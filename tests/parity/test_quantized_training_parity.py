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


def _binary_data(seed: int = 5, n: int = 1200, f: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A binary-classification target, so the objective's Hessian varies per row."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    logit = X[:, 0] - 0.5 * X[:, 1] + X[:, 2] * X[:, 3]
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(np.float64)
    return X, y


def _categorical_data(
    seed: int = 5, n: int = 1500, cats: int = 2, nums: int = 3
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Mixed numeric + integer-coded categorical features, with a categorical-driven target."""
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(n, nums))
    cat = np.column_stack([rng.integers(0, 6 + 4 * k, size=n) for k in range(cats)]).astype(float)
    X = np.column_stack([num, cat])
    cat_idx = list(range(nums, nums + cats))
    effect = sum((cat[:, k] % 3 - 1) for k in range(cats))
    y = num[:, 0] + 0.7 * effect + rng.normal(scale=0.2, size=n)
    return X, y, cat_idx


def _sparse_data(seed: int = 3, n: int = 2000, f: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Mostly-zero columns, so Exclusive Feature Bundling packs them into a multi-value group."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    X[rng.random(X.shape) < 0.75] = 0.0  # 75% zeros -> sparse groups bundled by EFB
    y = X[:, 0] + 2.0 * X[:, 1] - X[:, 2] * X[:, 3] + rng.normal(scale=0.2, size=n)
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


@pytest.mark.parametrize("bins", [4, 8, 16])
def test_quantized_binary_matches_upstream(bins: int) -> None:
    """A variable-Hessian objective: the row count of a side is estimated from the integer
    Hessian sum (`cnt_factor != 1`), not counted, so this exercises a path the constant-Hessian
    regression case does not."""
    X, y = _binary_data()
    params = {**BASE, "objective": "binary", "num_grad_quant_bins": bins}
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=10)
    ours = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10)
    assert _trees(ours.model_to_string()) == _trees(theirs.model_to_string())


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_quantized_renew_leaf_matches_upstream(objective: str) -> None:
    """`quant_train_renew_leaf` keeps the quantized split structure but recomputes every leaf
    output from the full-precision gradients (upstream `RenewIntGradTreeOutput`)."""
    X, y = _binary_data() if objective == "binary" else _data()
    params = {
        **BASE,
        "objective": objective,
        "num_grad_quant_bins": 8,
        "quant_train_renew_leaf": True,
    }
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=10)
    ours = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10)
    assert _trees(ours.model_to_string()) == _trees(theirs.model_to_string())


@pytest.mark.parametrize("bins", [4, 8, 16])
def test_quantized_categorical_matches_upstream(bins: int) -> None:
    """Categorical features use the integer categorical split (one-hot and many-vs-many by the
    gradient/Hessian ratio). Trees must match upstream tree-by-tree."""
    X, y, cat_idx = _categorical_data()
    params = {**BASE, "num_grad_quant_bins": bins, "min_data_in_leaf": 20}
    theirs = lightgbm.train(
        params, lightgbm.Dataset(X, y, categorical_feature=cat_idx), num_boost_round=10
    )
    ours = lgb.train(params, lgb.Dataset(X, y, categorical_feature=cat_idx), num_boost_round=10)
    assert _trees(ours.model_to_string()) == _trees(theirs.model_to_string())


@pytest.mark.parametrize("bins", [4, 8, 16])
def test_quantized_sparse_matches_upstream(bins: int) -> None:
    """Sparse columns bundle into a multi-value feature group; the integer histogram is built
    over its per-row bins (upstream `ConstructHistogramsMultiVal`). Trees must match upstream."""
    X, y = _sparse_data()
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
