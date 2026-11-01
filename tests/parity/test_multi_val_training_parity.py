"""P2/P3 acceptance: training with a multi-value feature group (MultiValBin) matches upstream.

Exclusive Feature Bundling pools sparse features it cannot bundle into one multi-value
group, whose histograms upstream builds row-wise from a ``MultiValBin``
(src/io/train_share_states.cpp, src/io/multi_val_sparse_bin.hpp, v4.7.0). Every case trains
both libraries on the same data on this machine and compares the model text exactly.
"""

from __future__ import annotations

import sys
from typing import Any

import lightgbm
import numpy as np
import pytest
import scipy.sparse

import bankai_lightgbm as lgb

BASE: dict[str, Any] = {
    "deterministic": True,
    "num_threads": 1,
    "force_col_wise": True,
    "seed": 7,
    "verbose": -1,
    "num_leaves": 15,
}

# Uniform drops at rate 1 with max_drop >= rounds: upstream's DART is deterministic (see
# test_training_parity._DART_RNG_INDEPENDENT).
DART = {"boosting": "dart", "skip_drop": 0.0, "uniform_drop": True, "drop_rate": 1.0}


def _mixed(n: int = 2500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two dense features plus sparse ones that end up in a multi-value group."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12))
    x[:, 2:][rng.random((n, 10)) < 0.75] = 0.0
    x[rng.random(x.shape) < 0.02] = np.nan
    x[:, 9] = rng.integers(0, 7, n) * (rng.random(n) < 0.3)  # sparse categorical codes
    y = (
        np.nan_to_num(x[:, 0])
        + np.nan_to_num(x[:, 3]) * 1.5
        - np.nan_to_num(x[:, 5])
        + (x[:, 9] == 3)
        + rng.normal(size=n) * 0.1
    )
    return x, y


def _very_sparse(n: int = 3000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Only sparse features: the multi-value group is the whole dataset."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 60))
    x[rng.random(x.shape) < 0.9] = 0.0
    y = x[:, :10].sum(axis=1) + rng.normal(size=n) * 0.05
    return x, y


def _as(fmt: str, x: np.ndarray) -> Any:
    return x if fmt == "dense" else scipy.sparse.csr_matrix(x).asformat(fmt)


def _models(
    data: Any, y: np.ndarray, params: dict[str, Any], rounds: int = 8, **ds_kw: Any
) -> tuple[str, str]:
    out = []
    for lib in (lgb, lightgbm):
        booster = lib.train({**BASE, **params}, lib.Dataset(data, y, **ds_kw), rounds)
        out.append(str(booster.model_to_string()))
    return out[0], out[1]


CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "binary": {"objective": "binary"},
    "huber": {"objective": "huber", "alpha": 0.8},
    "bagging": {"bagging_fraction": 0.6, "bagging_freq": 1},
    "goss": {"data_sample_strategy": "goss", "top_rate": 0.3, "other_rate": 0.2},
    "rf": {"boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1},
    "dart": {**DART, "max_drop": 50},
    "feature_fraction": {"feature_fraction": 0.5},
    "feature_fraction_bynode": {"feature_fraction_bynode": 0.5},
    "extra_trees": {"extra_trees": True},
    "zero_as_missing": {"zero_as_missing": True},
    "no_missing": {"use_missing": False},
    "regularized": {"lambda_l1": 0.5, "lambda_l2": 2.0, "min_gain_to_split": 0.01},
    "path_smooth": {"path_smooth": 2.0, "max_depth": 4},
    "small_bins": {"max_bin": 15, "min_data_in_bin": 10},
    "large_leaves": {"min_data_in_leaf": 200},
    "monotone": {"monotone_constraints": [1, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0]},
    "interaction": {"interaction_constraints": [[0, 3, 5], [1, 2, 4, 6, 7, 8, 9, 10, 11]]},
}


_DARWIN_PATH_SMOOTH = pytest.mark.xfail(
    sys.platform == "darwin",
    strict=True,
    reason=(
        "macOS arm64 only: path_smooth on the multi-value path picks a different split in one "
        "node (decision_type and one leaf value differ from upstream), not triaged yet; "
        "see ROADMAP P2 MultiValBin and AGENTS.md section 4 (madd/FMA)"
    ),
)


@pytest.mark.parametrize("fmt", ["dense", "csr", "csc"])
@pytest.mark.parametrize(
    "name",
    [name for name in CONFIGS],
)
def test_multi_value_training_matches_upstream(fmt: str, name: str) -> None:
    x, y = _mixed()
    if CONFIGS[name].get("objective") == "binary":
        y = (y > np.median(y)).astype(float)
    ours, theirs = _models(_as(fmt, x), y, CONFIGS[name])
    assert ours == theirs


@pytest.mark.parametrize("fmt", ["dense", "csr"])
def test_all_features_in_one_multi_value_group(fmt: str) -> None:
    x, y = _very_sparse()
    for params in ({}, {"feature_fraction": 0.3}, {"bagging_fraction": 0.5, "bagging_freq": 2}):
        ours, theirs = _models(_as(fmt, x), y, params)
        assert ours == theirs, params


def test_weighted_and_categorical_multi_value_group() -> None:
    x, y = _mixed()
    w = np.random.default_rng(5).uniform(0.2, 3.0, len(y))
    ours, theirs = _models(x, y, {}, weight=w, categorical_feature=[9])
    assert ours == theirs


def test_multiclass_with_multi_value_group() -> None:
    x, y = _mixed()
    labels = np.digitize(y, np.quantile(y, [0.3, 0.7])).astype(float)
    ours, theirs = _models(x, labels, {"objective": "multiclass", "num_class": 3})
    assert ours == theirs


def test_lambdarank_with_multi_value_group() -> None:
    x, y = _mixed(2400)
    rel = np.digitize(y, np.quantile(y, [0.5, 0.8, 0.95])).astype(float)
    group = np.full(120, 20)
    ours, theirs = _models(x, rel, {"objective": "lambdarank"}, group=group)
    assert ours == theirs


def test_multi_value_training_with_validation_and_early_stopping() -> None:
    x, y = _mixed(3500)
    outcomes = []
    for lib in (lgb, lightgbm):
        train = lib.Dataset(x[:2500], y[:2500])
        valid = lib.Dataset(x[2500:], y[2500:], reference=train)
        booster = lib.train(
            {**BASE, "metric": "l2", "learning_rate": 0.3},
            train,
            60,
            valid_sets=[valid],
            callbacks=[lib.early_stopping(3, verbose=False)],
        )
        outcomes.append((booster.model_to_string(), booster.best_iteration))
    assert outcomes[0] == outcomes[1]


def test_multi_value_training_is_exact_with_two_threads() -> None:
    """The multi-value histogram follows upstream's data blocks and merge order, so the
    result does not depend on scheduling and matches upstream with several threads too."""
    x, y = _very_sparse(6000)
    ours, theirs = _models(x, y, {"num_threads": 2})
    assert ours == theirs
