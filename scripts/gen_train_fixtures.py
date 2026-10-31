"""Generate upstream *training* fixtures: models trained deterministically on one thread.

Each case writes ``tests/fixtures/train/<case>/``:
  - ``x.npy`` / ``y.npy`` (float64), ``params.json`` (``params`` + ``num_boost_round``)
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
}


def main() -> None:
    expected = UPSTREAM_VERSION.lstrip("v")
    if lgb.__version__ != expected:
        raise SystemExit(f"need lightgbm=={expected}, found {lgb.__version__}")
    for case, (make, extra, rounds) in CASES.items():
        rng = np.random.default_rng(0)
        x, y = make(rng)
        params = {**BASE, **extra}
        booster = lgb.train(params, lgb.Dataset(x, y), num_boost_round=rounds)
        out = OUT / case
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "x.npy", x)
        np.save(out / "y.npy", y)
        (out / "params.json").write_text(
            json.dumps({"params": params, "num_boost_round": rounds}, indent=1)
        )
        (out / "model.txt").write_text(booster.model_to_string())
        print(f"wrote {out} ({booster.num_trees()} trees)")


if __name__ == "__main__":
    main()
