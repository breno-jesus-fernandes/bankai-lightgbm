"""Cost-efficient gradient boosting (ROADMAP T3.4, upstream `CostEfficientGradientBoosting`).

CEGB subtracts a cost penalty from each candidate split's gain before the split is chosen:
``cegb_tradeoff`` scales ``cegb_penalty_split`` per data point in the leaf, a one-off
``cegb_penalty_feature_coupled`` the first time a feature is used in any split, and a lazy
``cegb_penalty_feature_lazy`` per data point that has not yet touched the feature. The penalties
change which splits win, so the trees differ from penalty-free training and must match upstream
byte for byte. Deterministic single-threaded training keeps the comparison free of thread order.
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
}


def _data(seed: int = 3, n: int = 400, f: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, f))
    y = 2.0 * X[:, 0] + X[:, 2] ** 2 - X[:, 4] + rng.normal(scale=0.1, size=n)
    return X, y


def _trees(model_text: str) -> str:
    """The tree section only: the trailing parameters block echoes the cegb values themselves."""
    return model_text.split("\nparameters:")[0]


CASES = [
    {"cegb_penalty_split": 0.5},
    {"cegb_penalty_split": 2.0, "cegb_tradeoff": 0.5},
    {"cegb_penalty_feature_coupled": [1.0, 2.0, 1.0, 2.0, 1.0]},
    {"cegb_penalty_feature_coupled": [0.5, 1.0, 0.5, 1.0, 0.5], "cegb_tradeoff": 2.0},
    {"cegb_penalty_feature_lazy": [0.1, 0.2, 0.3, 0.4, 0.5]},
    {"cegb_penalty_feature_lazy": [0.05, 0.1, 0.15, 0.2, 0.25], "cegb_tradeoff": 2.0},
    {
        "cegb_penalty_split": 0.3,
        "cegb_penalty_feature_coupled": [1.0, 0.0, 2.0, 0.0, 1.0],
        "cegb_penalty_feature_lazy": [0.1, 0.0, 0.2, 0.0, 0.1],
        "cegb_tradeoff": 1.5,
    },
]


@pytest.mark.parametrize("case", CASES)
def test_cegb_matches_upstream_tree_by_tree(case: dict) -> None:
    X, y = _data()
    params = {**BASE, **case}
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=12)
    ours = lgb.train(params, lgb.Dataset(X, y), num_boost_round=12)
    assert _trees(ours.model_to_string()) == _trees(theirs.model_to_string())


def test_cegb_changes_the_trees() -> None:
    """A harsh penalty must actually change the trees (guards against silently ignoring cegb)."""
    X, y = _data()
    plain = lgb.train(BASE, lgb.Dataset(X, y), num_boost_round=12)
    penalized = lgb.train(
        {**BASE, "cegb_penalty_feature_coupled": [50.0, 100.0, 10.0, 25.0, 30.0]},
        lgb.Dataset(X, y),
        num_boost_round=12,
    )
    assert _trees(plain.model_to_string()) != _trees(penalized.model_to_string())


def test_cegb_penalty_feature_coupled_wrong_size_errors() -> None:
    """A per-feature penalty vector of the wrong length is rejected like upstream."""
    X, y = _data()
    params = {**BASE, "cegb_penalty_feature_coupled": [1.0, 2.0]}  # only 2 of 5 features
    with pytest.raises(lightgbm.basic.LightGBMError):
        lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=2)
    with pytest.raises(lgb.basic.LightGBMError):
        lgb.train(params, lgb.Dataset(X, y), num_boost_round=2)
