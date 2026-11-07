"""``reset_parameter`` during training must match upstream.

Sampling and feature-fraction changes take effect from the next iteration, so the model
trained with a reset in the middle must equal upstream's, tree by tree.
"""

from __future__ import annotations

import lightgbm
import numpy as np

import bankai_lightgbm as lgb

PARAMS = {
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "bagging_freq": 0,
    "bagging_fraction": 1.0,
    "feature_fraction": 1.0,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
}

RESET = {"bagging_freq": 1, "bagging_fraction": 0.6, "feature_fraction": 0.7}


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(500, 6))
    y = x[:, 0] - 0.5 * x[:, 2] + rng.normal(scale=0.2, size=500)
    return x, y


def test_reset_parameter_mid_training_matches_upstream() -> None:
    x, y = _data()
    up = lightgbm.Booster(params=PARAMS, train_set=lightgbm.Dataset(x, y, free_raw_data=False))
    ours = lgb.Booster(params=PARAMS, train_set=lgb.Dataset(x, y, free_raw_data=False))
    for _ in range(5):
        up.update()
        ours.update()
    up.reset_parameter(RESET)
    ours.reset_parameter(RESET)
    for _ in range(5):
        up.update()
        ours.update()
    assert ours.num_trees() == up.num_trees()
    np.testing.assert_allclose(ours.predict(x), up.predict(x), rtol=1e-9, atol=1e-12)
