"""Quantized training (`use_quantized_grad`) must match upstream's model text (T2.7).

Upstream discretizes gradients and Hessians to integers, builds integer histograms, and
searches splits on the integer sums (`src/treelearner/serial_tree_learner.cpp`,
`src/io/dense_bin.hpp` `ConstructHistogramIntInner`, `feature_histogram.hpp`
`FindBestThresholdSequentiallyInt`). With `stochastic_rounding=false` the discretization is
deterministic nearest rounding, so the whole path is reproducible and platform independent.

These tests pin the deterministic, single-thread, numeric-only slice: the trained trees must be
identical to upstream (features, thresholds, leaf counts) and the leaf values must match within
the training tolerance of AGENTS.md section 4.
"""

from __future__ import annotations

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

BASE = {
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
    "use_quantized_grad": True,
    "stochastic_rounding": False,
    "quant_train_renew_leaf": False,
}


def _data(seed: int = 5, n: int = 1200, f: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(scale=0.2, size=n)
    return X, y


def _trees(model_text: str) -> str:
    """The tree section only; the trailing parameters block echoes the flags."""
    return model_text.split("\nparameters:")[0]


@pytest.mark.parametrize("bins", [4, 8, 16])
def test_quantized_regression_matches_upstream(bins: int) -> None:
    X, y = _data()
    params = {**BASE, "num_grad_quant_bins": bins}
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=10)
    ours = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10)
    assert _trees(ours.model_to_string()) == _trees(theirs.model_to_string())


def test_quantized_changes_the_model() -> None:
    """Quantized training is a distinct path: it must not coincide with full precision."""
    X, y = _data()
    full = lgb.train({**BASE, "use_quantized_grad": False}, lgb.Dataset(X, y), num_boost_round=10)
    quant = lgb.train({**BASE, "num_grad_quant_bins": 4}, lgb.Dataset(X, y), num_boost_round=10)
    assert _trees(full.model_to_string()) != _trees(quant.model_to_string())
