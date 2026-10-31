"""Generate upstream *training* fixtures: models trained deterministically on one thread.

Each case writes ``tests/fixtures/train/<case>/``:
  - ``x.npy`` / ``y.npy`` (float64), ``params.json`` (``params`` + ``num_boost_round``)
  - ``w.npy`` (float32 sample weights, only for ``*_weighted`` cases)
  - ``group.npy`` (int32 query sizes, only for ranking cases)
  - ``model.txt`` (upstream ``model_to_string()`` after ``lgb.train``)

bankai must reproduce ``model.txt`` byte-for-byte (see AGENTS.md, parity tolerances).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "train"
UPSTREAM_VERSION = (ROOT / "UPSTREAM_VERSION").read_text().strip()

# Pin everything that would otherwise depend on timing or thread count.
BASE: dict[str, Any] = {
    "deterministic": True,
    "num_threads": 1,
    "force_col_wise": True,
    "seed": 7,
    "verbose": -1,
}

Data = tuple[np.ndarray, np.ndarray]


def reg(rng: np.random.Generator) -> Data:
    x = rng.normal(size=(2000, 6))
    x[:, 3] = np.round(x[:, 3], 1)
    y = 2 * x[:, 0] + np.sin(3 * x[:, 1]) + x[:, 2] * x[:, 3] + rng.normal(scale=0.2, size=2000)
    return x, y


def reg_nan(rng: np.random.Generator) -> Data:
    x, y = reg(rng)
    x[rng.random(x.shape) < 0.15] = np.nan
    return x, y


def reg_zero(rng: np.random.Generator) -> Data:
    x, y = reg(rng)
    x[rng.random(x.shape) < 0.4] = 0.0
    return x, y


def counts(rng: np.random.Generator) -> Data:
    x, _ = reg(rng)
    y = rng.poisson(np.exp(0.5 * x[:, 0] + 0.3 * np.sin(2 * x[:, 1]))).astype(np.float64)
    return x, y


def positive(rng: np.random.Generator) -> Data:
    x, _ = reg(rng)
    y = rng.gamma(2.0, np.exp(0.4 * x[:, 0] - 0.2 * x[:, 2]) / 2.0)
    return x, y


def binary(rng: np.random.Generator) -> Data:
    x, y = reg(rng)
    return x, (y + rng.normal(scale=1.0, size=y.shape) > 0.5).astype(np.float64)


def multiclass(rng: np.random.Generator) -> Data:
    x, _ = reg(rng)
    logits = np.stack([x[:, 0], x[:, 1] * x[:, 2], -x[:, 0] + np.sin(x[:, 3])], axis=1)
    logits += rng.gumbel(size=logits.shape)
    return x, np.argmax(logits, axis=1).astype(np.float64)


def probability(rng: np.random.Generator) -> Data:
    x, y = reg(rng)
    return x, 1.0 / (1.0 + np.exp(-0.7 * y))


def categorical(rng: np.random.Generator) -> Data:
    """Regression data where columns 0, 4 and 5 are integer categories (4, 12 and 30 levels)."""
    x, y = reg(rng)
    x[:, 0] = rng.integers(0, 4, size=len(y))
    x[:, 4] = rng.integers(0, 12, size=len(y))
    x[:, 5] = rng.integers(0, 30, size=len(y))
    effect = np.array([0.5, -1.0, 2.0, 0.0])[x[:, 0].astype(int)]
    effect += np.where(x[:, 4] % 3 == 0, 1.5, -0.5) + 0.05 * x[:, 5]
    return x, y + effect


def categorical_binary(rng: np.random.Generator) -> Data:
    x, y = categorical(rng)
    return x, (y > np.median(y)).astype(np.float64)


def categorical_nan(rng: np.random.Generator) -> Data:
    x, y = categorical(rng)
    x[rng.random(x.shape[0]) < 0.1, 4] = np.nan
    return x, y


def relevance(rng: np.random.Generator) -> Data:
    """Graded relevance 0..4 from a noisy linear signal; 200 queries of 10 rows."""
    x, y = reg(rng)
    edges = np.quantile(y + rng.normal(scale=0.5, size=y.shape), [0.4, 0.65, 0.85, 0.95])
    return x, np.searchsorted(edges, y).astype(np.float64)


GROUP_SIZE = 10

# Columns that are categorical in each categorical case.
CATEGORICAL_COLS = [0, 4, 5]


CASES: dict[str, tuple[Callable[[np.random.Generator], Data], dict[str, Any], int]] = {
    "l2_default": (reg, {"objective": "regression"}, 20),
    "l2_small": (
        reg,
        {"objective": "regression", "num_leaves": 7, "min_data_in_leaf": 5, "learning_rate": 0.3},
        15,
    ),
    "l2_nan": (reg_nan, {"objective": "regression", "num_leaves": 15}, 15),
    "l2_zero": (reg_zero, {"objective": "regression", "num_leaves": 15}, 15),
    "l2_zero_as_missing": (
        reg_zero,
        {"objective": "regression", "num_leaves": 15, "zero_as_missing": True},
        15,
    ),
    "l2_regularized": (
        reg,
        {
            "objective": "regression",
            "lambda_l1": 0.5,
            "lambda_l2": 2.0,
            "min_gain_to_split": 0.01,
            "min_sum_hessian_in_leaf": 5.0,
        },
        15,
    ),
    "l2_max_delta_smooth": (
        reg,
        {"objective": "regression", "max_delta_step": 0.5, "path_smooth": 3.0},
        15,
    ),
    "l2_max_depth": (reg, {"objective": "regression", "max_depth": 3}, 15),
    "l2_no_boost_from_average": (reg, {"objective": "regression", "boost_from_average": False}, 10),
    "l2_sqrt": (counts, {"objective": "regression", "reg_sqrt": True, "num_leaves": 15}, 10),
    "l1": (reg, {"objective": "regression_l1", "num_leaves": 15}, 15),
    "l1_no_boost_from_average": (
        reg,
        {"objective": "regression_l1", "num_leaves": 15, "boost_from_average": False},
        10,
    ),
    "quantile": (reg, {"objective": "quantile", "alpha": 0.3, "num_leaves": 15}, 15),
    "quantile_high": (reg, {"objective": "quantile", "alpha": 0.9, "num_leaves": 15}, 10),
    "mape": (positive, {"objective": "mape", "num_leaves": 15}, 15),
    "huber": (reg, {"objective": "huber", "alpha": 0.9, "num_leaves": 15}, 15),
    "fair": (reg, {"objective": "fair", "fair_c": 0.5, "num_leaves": 15}, 15),
    "poisson": (counts, {"objective": "poisson", "num_leaves": 15}, 15),
    "gamma": (positive, {"objective": "gamma", "num_leaves": 15}, 15),
    "tweedie": (counts, {"objective": "tweedie", "tweedie_variance_power": 1.3}, 15),
    "binary": (binary, {"objective": "binary", "num_leaves": 15}, 15),
    "binary_unbalance": (binary, {"objective": "binary", "is_unbalance": True}, 10),
    "binary_sigmoid": (
        binary,
        {"objective": "binary", "sigmoid": 0.7, "scale_pos_weight": 2.5, "num_leaves": 7},
        10,
    ),
    "multiclass": (multiclass, {"objective": "multiclass", "num_class": 3, "num_leaves": 7}, 10),
    "multiclassova": (
        multiclass,
        {"objective": "multiclassova", "num_class": 3, "num_leaves": 7},
        10,
    ),
    "xentropy": (probability, {"objective": "cross_entropy", "num_leaves": 15}, 15),
    "xentlambda": (probability, {"objective": "cross_entropy_lambda", "num_leaves": 15}, 15),
    "lambdarank": (relevance, {"objective": "lambdarank", "num_leaves": 15}, 15),
    "lambdarank_options": (
        relevance,
        {
            "objective": "lambdarank",
            "num_leaves": 7,
            "lambdarank_norm": False,
            "lambdarank_truncation_level": 4,
            "label_gain": [0, 1, 3, 4, 10],
            "sigmoid": 1.5,
        },
        10,
    ),
    "rank_xendcg": (relevance, {"objective": "rank_xendcg", "num_leaves": 15}, 15),
    "feature_fraction": (
        reg,
        {"objective": "regression", "num_leaves": 15, "feature_fraction": 0.5},
        15,
    ),
    "feature_fraction_bynode": (
        reg,
        {"objective": "regression", "num_leaves": 15, "feature_fraction_bynode": 0.4},
        15,
    ),
    "feature_fraction_both": (
        reg,
        {
            "objective": "regression",
            "num_leaves": 15,
            "feature_fraction": 0.7,
            "feature_fraction_bynode": 0.5,
            "feature_fraction_seed": 9,
        },
        15,
    ),
    "bagging": (
        reg,
        {
            "objective": "regression",
            "num_leaves": 15,
            "bagging_fraction": 0.6,
            "bagging_freq": 2,
            "bagging_seed": 5,
        },
        15,
    ),
    "bagging_no_freq": (
        reg,
        {"objective": "regression", "num_leaves": 15, "bagging_fraction": 0.6},
        10,
    ),
    "bagging_balanced": (
        binary,
        {
            "objective": "binary",
            "num_leaves": 15,
            "bagging_freq": 1,
            "pos_bagging_fraction": 0.5,
            "neg_bagging_fraction": 0.8,
        },
        12,
    ),
    "bagging_feature_fraction": (
        reg,
        {
            "objective": "regression",
            "num_leaves": 15,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "feature_fraction": 0.6,
        },
        12,
    ),
    "goss": (
        reg,
        {
            "objective": "regression",
            "num_leaves": 15,
            "data_sample_strategy": "goss",
            "learning_rate": 0.1,
            "top_rate": 0.2,
            "other_rate": 0.1,
        },
        25,
    ),
    "goss_dense_sample": (
        reg,
        {
            "objective": "regression",
            "num_leaves": 7,
            "data_sample_strategy": "goss",
            "learning_rate": 0.2,
            "top_rate": 0.3,
            "other_rate": 0.4,
            "bagging_seed": 17,
        },
        12,
    ),
    "cat_default": (
        categorical,
        {"objective": "regression", "num_leaves": 15, "categorical_cols": CATEGORICAL_COLS},
        15,
    ),
    "cat_onehot": (
        categorical,
        {
            "objective": "regression",
            "num_leaves": 15,
            "max_cat_to_onehot": 32,
            "categorical_cols": CATEGORICAL_COLS,
        },
        15,
    ),
    "cat_regularized": (
        categorical,
        {
            "objective": "regression",
            "num_leaves": 15,
            "cat_smooth": 5.0,
            "cat_l2": 3.0,
            "min_data_per_group": 40,
            "max_cat_threshold": 8,
            "lambda_l2": 1.0,
            "categorical_cols": CATEGORICAL_COLS,
        },
        15,
    ),
    "cat_extra_trees": (
        categorical,
        {
            "objective": "regression",
            "num_leaves": 15,
            "extra_trees": True,
            "categorical_cols": CATEGORICAL_COLS,
        },
        15,
    ),
    "cat_nan": (
        categorical_nan,
        {"objective": "regression", "num_leaves": 15, "categorical_cols": CATEGORICAL_COLS},
        15,
    ),
    "interaction": (
        reg,
        {"objective": "regression", "num_leaves": 15, "interaction_constraints": "[0,1,2],[3,4,5]"},
        15,
    ),
    "interaction_sampled": (
        reg,
        {
            "objective": "regression",
            "num_leaves": 15,
            "interaction_constraints": "[0,1],[2,3,4],[5]",
            "feature_fraction": 0.7,
            "feature_fraction_bynode": 0.6,
            "feature_fraction_seed": 13,
        },
        15,
    ),
    "interaction_bynode": (
        reg,
        {
            "objective": "regression",
            "num_leaves": 15,
            "interaction_constraints": "[0,2,4],[1,3,5]",
            "feature_fraction_bynode": 0.5,
        },
        15,
    ),
    "monotone": (
        reg,
        {"objective": "regression", "num_leaves": 15, "monotone_constraints": "1,0,-1,1,0,0"},
        15,
    ),
    "monotone_penalty": (
        reg,
        {
            "objective": "regression",
            "num_leaves": 31,
            "monotone_constraints": "-1,1,0,0,0,1",
            "monotone_penalty": 1.5,
            "min_data_in_leaf": 10,
        },
        15,
    ),
    "monotone_zero": (
        reg,
        {"objective": "regression", "num_leaves": 15, "monotone_constraints": "0,0,0,0,0,0"},
        12,
    ),
    "rf": (
        reg,
        {
            "objective": "regression",
            "boosting": "rf",
            "num_leaves": 15,
            "bagging_freq": 1,
            "bagging_fraction": 0.6,
        },
        12,
    ),
    "rf_feature_fraction": (
        reg,
        {"objective": "regression", "boosting": "rf", "num_leaves": 15, "feature_fraction": 0.5},
        10,
    ),
    "rf_binary": (
        binary,
        {
            "objective": "binary",
            "boosting": "rf",
            "num_leaves": 15,
            "bagging_freq": 2,
            "bagging_fraction": 0.7,
        },
        10,
    ),
    "dart": (
        reg,
        {
            "objective": "regression",
            "boosting": "dart",
            "num_leaves": 15,
            "learning_rate": 0.2,
            "drop_rate": 0.2,
        },
        15,
    ),
    "dart_uniform": (
        reg,
        {
            "objective": "regression",
            "boosting": "dart",
            "num_leaves": 15,
            "learning_rate": 0.2,
            "uniform_drop": True,
            "drop_rate": 0.3,
            "max_drop": 3,
        },
        15,
    ),
    "dart_xgboost": (
        reg,
        {
            "objective": "regression",
            "boosting": "dart",
            "num_leaves": 15,
            "learning_rate": 0.3,
            "xgboost_dart_mode": True,
            "skip_drop": 0.3,
            "drop_seed": 9,
        },
        15,
    ),
    "dart_binary": (
        binary,
        {
            "objective": "binary",
            "boosting": "dart",
            "num_leaves": 7,
            "learning_rate": 0.25,
            "drop_rate": 0.4,
        },
        12,
    ),
    "cat_bagging": (
        categorical,
        {
            "objective": "regression",
            "num_leaves": 15,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "bagging_seed": 4,
            "categorical_cols": CATEGORICAL_COLS,
        },
        12,
    ),
    "cat_binary": (
        categorical_binary,
        {
            "objective": "binary",
            "num_leaves": 15,
            "categorical_cols": CATEGORICAL_COLS,
        },
        12,
    ),
    "rank_xendcg_seed": (
        relevance,
        {"objective": "rank_xendcg", "num_leaves": 7, "objective_seed": 11},
        10,
    ),
}

# Same data and params as the named case, trained with random sample weights.
WEIGHTED = [
    "l2_default",
    "monotone",
    "cat_default",
    "bagging",
    "goss",
    "l1",
    "quantile",
    "mape",
    "huber",
    "fair",
    "poisson",
    "gamma",
    "tweedie",
    "binary",
    "multiclass",
    "multiclassova",
    "xentropy",
    "xentlambda",
    "lambdarank",
    "rank_xendcg",
]


def write_case(
    case: str,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    params: dict[str, Any],
    rounds: int,
) -> None:
    ranking = params["objective"] in ("lambdarank", "rank_xendcg")
    group = np.full(len(y) // GROUP_SIZE, GROUP_SIZE, dtype=np.int32) if ranking else None
    params = dict(params)
    cats = params.pop("categorical_cols", None)
    dataset = lgb.Dataset(x, y, weight=w, group=group, categorical_feature=cats or "auto")
    booster = lgb.train(params, dataset, num_boost_round=rounds)
    out = OUT / case
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x.npy", x)
    np.save(out / "y.npy", y)
    if w is not None:
        np.save(out / "w.npy", w)
    if group is not None:
        np.save(out / "group.npy", group)
    (out / "params.json").write_text(
        json.dumps(
            {"params": params, "num_boost_round": rounds, "categorical": cats or []},
            indent=1,
        )
    )
    (out / "model.txt").write_text(booster.model_to_string())
    print(f"wrote {out} ({booster.num_trees()} trees)")


def main() -> None:
    expected = UPSTREAM_VERSION.lstrip("v")
    if lgb.__version__ != expected:
        raise SystemExit(f"need lightgbm=={expected}, found {lgb.__version__}")
    for case, (make, extra, rounds) in CASES.items():
        rng = np.random.default_rng(0)
        x, y = make(rng)
        params = {**BASE, **extra}
        write_case(case, x, y, None, params, rounds)
        if case in WEIGHTED:
            w = np.random.default_rng(1).uniform(0.2, 2.0, size=len(y)).astype(np.float32)
            write_case(f"{case}_weighted", x, y, w, params, rounds)


if __name__ == "__main__":
    main()
