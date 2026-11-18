"""Auto layout choice (ROADMAP T2.2, upstream ``Dataset::GetShareStates`` selection rules).

With neither ``force_col_wise`` nor ``force_row_wise`` set, upstream times a column-wise and a
row-wise histogram build and keeps the faster one. The two layouts accumulate the same bins over
the same rows, so the *model* is identical whichever is chosen: auto training must match upstream
auto training and must equal bankai's own forced-column and forced-row models byte for byte (tree
section). Setting both force flags at once is a conflict upstream rejects with a ``LightGBMError``.

Deterministic single-threaded training keeps the comparison free of thread-order effects.
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


def _trees(model_text: str) -> str:
    """The tree section only: the trailing parameters block echoes the layout flags themselves."""
    return model_text.split("\nparameters:")[0]


def test_auto_layout_matches_upstream() -> None:
    """Auto mode (neither force flag set) trains to upstream's model byte for byte."""
    X, y = _dense()
    params = {**BASE, "objective": "regression"}
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=10)
    ours = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10)
    assert ours.model_to_string() == theirs.model_to_string()


def test_auto_layout_trees_equal_both_forced() -> None:
    """The auto model equals both forced layouts: the choice never changes the trees."""
    X, y = _dense()
    auto = lgb.train({**BASE}, lgb.Dataset(X, y), num_boost_round=10)
    col = lgb.train({**BASE, "force_col_wise": True}, lgb.Dataset(X, y), num_boost_round=10)
    row = lgb.train({**BASE, "force_row_wise": True}, lgb.Dataset(X, y), num_boost_round=10)
    assert _trees(auto.model_to_string()) == _trees(col.model_to_string())
    assert _trees(auto.model_to_string()) == _trees(row.model_to_string())


def test_force_both_layouts_raises_like_upstream() -> None:
    """Setting both force flags is the conflict upstream rejects (``Dataset::GetShareStates``)."""
    X, y = _dense()
    params = {**BASE, "force_col_wise": True, "force_row_wise": True}

    with pytest.raises(lightgbm.basic.LightGBMError) as up:
        lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=2)
    assert "force_col_wise" in str(up.value) and "force_row_wise" in str(up.value)

    with pytest.raises(lgb.basic.LightGBMError) as ours:
        lgb.train(params, lgb.Dataset(X, y), num_boost_round=2)
    msg = str(ours.value)
    assert "force_col_wise" in msg and "force_row_wise" in msg
