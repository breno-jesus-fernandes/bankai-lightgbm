"""``force_row_wise=true`` must raise instead of training col-wise silently.

Upstream selects the row-wise histogram layout with this parameter. bankai only has the
column-wise layout (ROADMAP P3, "Row-wise training layout"), so until that lands the
combination raises ``LightGBMError``, like every other unsupported path in the package.
"""

from __future__ import annotations

import numpy as np
import pytest

import bankai_lightgbm as lgb

PARAMS = {
    "objective": "regression",
    "num_leaves": 7,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
}


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] + rng.normal(scale=0.1, size=200)
    return X, y


def test_train_with_force_row_wise_raises() -> None:
    X, y = _data()
    with pytest.raises(lgb.LightGBMError, match="force_row_wise"):
        lgb.train({**PARAMS, "force_row_wise": True}, lgb.Dataset(X, y), num_boost_round=2)


def test_sklearn_with_force_row_wise_raises() -> None:
    X, y = _data()
    model = lgb.LGBMRegressor(
        n_estimators=2, num_leaves=7, force_row_wise=True, verbose=-1, n_jobs=1
    )
    with pytest.raises(lgb.LightGBMError, match="force_row_wise"):
        model.fit(X, y)


def test_force_row_wise_false_still_trains() -> None:
    X, y = _data()
    booster = lgb.train({**PARAMS, "force_row_wise": False}, lgb.Dataset(X, y), num_boost_round=2)
    assert booster.num_trees() == 2
