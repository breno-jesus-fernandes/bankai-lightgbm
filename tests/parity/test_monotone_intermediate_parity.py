"""monotone_constraints_method=intermediate must match upstream tree by tree.

upstream: src/treelearner/monotone_constraints.hpp (IntermediateLeafConstraints)
"""

from __future__ import annotations

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

PARAMS = {
    "objective": "regression",
    "num_leaves": 31,
    "min_data_in_leaf": 5,
    "min_data_in_bin": 1,
    "learning_rate": 0.1,
    "monotone_constraints": [1, -1, 0, 1],
    "monotone_constraints_method": "intermediate",
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 5,
}


def _data(with_nan_and_categorical: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    x = rng.normal(size=(1500, 4))
    y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + np.sin(3 * x[:, 3]) + rng.normal(scale=0.3, size=1500)
    if with_nan_and_categorical:
        # column 2 is categorical (unconstrained), column 3 gets missing values
        x[:, 2] = rng.integers(0, 5, size=1500).astype(np.float64)
        y = y + 0.5 * (x[:, 2] == 1)
        x[rng.random(1500) < 0.1, 3] = np.nan
    return x, y


@pytest.mark.parametrize("penalty", [0.0, 2.0])
@pytest.mark.parametrize("with_nan_and_categorical", [False, True])
def test_intermediate_monotone_matches_upstream_tree_by_tree(
    penalty: float, with_nan_and_categorical: bool
) -> None:
    x, y = _data(with_nan_and_categorical)
    params = {**PARAMS, "monotone_penalty": penalty}
    cat = [2] if with_nan_and_categorical else "auto"
    up = lightgbm.Booster(params=params, train_set=lightgbm.Dataset(x, y, categorical_feature=cat))
    ours = lgb.Booster(params=params, train_set=lgb.Dataset(x, y, categorical_feature=cat))
    for _ in range(20):
        up.update()
        ours.update()
    assert ours.num_trees() == up.num_trees()
    np.testing.assert_allclose(ours.predict(x), up.predict(x), rtol=1e-9, atol=0)
