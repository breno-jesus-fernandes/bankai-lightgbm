"""``force_row_wise=true`` must train to upstream's model text (ROADMAP P3, row-wise layout).

Row-wise and column-wise histograms accumulate each bin over the same rows in the same
order, so the models are expected to match byte for byte, as upstream's do. Deterministic,
single-threaded training keeps the comparison free of thread-order effects.
"""

from __future__ import annotations

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

BASE = {
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
}


def _dense(seed: int = 3, n: int = 1500, f: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    X[rng.random(size=X.shape) < 0.1] = np.nan
    y = 2.0 * np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) ** 2 + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.mark.parametrize(
    "extra",
    [
        {"objective": "regression"},
        {"objective": "binary"},
        {"objective": "regression", "bagging_fraction": 0.7, "bagging_freq": 1},
    ],
)
def test_row_wise_model_matches_upstream(extra: dict) -> None:
    X, y = _dense()
    if extra["objective"] == "binary":
        y = (y > np.median(y)).astype(float)
    params = {**BASE, **extra, "force_row_wise": True}
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=10)
    ours = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10)
    assert ours.model_to_string() == theirs.model_to_string()


def _trees(model_text: str) -> str:
    """The tree section only: the trailing parameters block echoes the layout flag itself."""
    return model_text.split("\nparameters:")[0]


def test_row_wise_trees_match_col_wise() -> None:
    X, y = _dense()
    col = lgb.train({**BASE, "force_col_wise": True}, lgb.Dataset(X, y), num_boost_round=10)
    row = lgb.train({**BASE, "force_row_wise": True}, lgb.Dataset(X, y), num_boost_round=10)
    assert _trees(row.model_to_string()) == _trees(col.model_to_string())


def test_force_row_wise_false_still_trains() -> None:
    X, y = _dense()
    booster = lgb.train({**BASE, "force_row_wise": False}, lgb.Dataset(X, y), num_boost_round=2)
    assert booster.num_trees() == 2
