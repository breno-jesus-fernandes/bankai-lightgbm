"""Linear leaves (``linear_tree``) must match upstream prediction for prediction.

upstream: src/treelearner/linear_tree_learner.cpp (CalculateLinear)

The leaf fit solves the same normal equations as upstream, but with a complete-pivot LU
solve instead of Eigen's full-pivot inverse, so the last bits can differ. Predictions are
held to rtol 1e-7, a few orders above that rounding and far below any real divergence.
"""

from __future__ import annotations

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

BASE = {
    "objective": "regression",
    "num_leaves": 7,
    "min_data_in_leaf": 10,
    "min_data_in_bin": 1,
    "learning_rate": 0.1,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 4,
    "linear_tree": True,
}


def _data(with_nan: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(600, 3))
    y = (
        2.0 * x[:, 0]
        - x[:, 1]
        + np.where(x[:, 2] > 0, 1.5 * x[:, 2], -x[:, 2])
        + rng.normal(scale=0.1, size=600)
    )
    if with_nan:
        x[::9, 0] = np.nan
        x[::13, 2] = np.nan
    return x, y


@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("linear_lambda", [0.0, 0.5])
def test_linear_tree_predictions_match_upstream(with_nan: bool, linear_lambda: float) -> None:
    x, y = _data(with_nan)
    params = {**BASE, "linear_lambda": linear_lambda}
    up = lightgbm.train(params, lightgbm.Dataset(x, y), num_boost_round=8)
    ours = lgb.train(params, lgb.Dataset(x, y), num_boost_round=8)
    assert ours.num_trees() == up.num_trees()
    np.testing.assert_allclose(ours.predict(x), up.predict(x), rtol=1e-7, atol=1e-9)
