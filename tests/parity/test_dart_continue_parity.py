"""DART continued training (``init_model``) must match upstream.

Upstream's DART drops only the trees trained in the current run (indices after the init model's
iterations) and keeps the init model's trees fixed. The model text and the predictions must match
``lightgbm==4.7.0`` exactly on one thread with a fixed drop seed.
"""

from __future__ import annotations

import lightgbm
import numpy as np

import bankai_lightgbm as lgb

PARAMS = {
    "boosting_type": "dart",
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "drop_rate": 0.3,
    "skip_drop": 0.2,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
    "drop_seed": 11,
}


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 6))
    y = x[:, 0] * 2.0 + np.sin(x[:, 1]) + rng.normal(scale=0.1, size=400)
    return x, y


def test_dart_continued_training_matches_upstream() -> None:
    x, y = _data()
    up_init = lightgbm.train(
        PARAMS, lightgbm.Dataset(x, y, free_raw_data=False), num_boost_round=10
    )
    up = lightgbm.train(
        PARAMS,
        lightgbm.Dataset(x, y, free_raw_data=False),
        num_boost_round=15,
        init_model=up_init,
    )

    our_init = lgb.train(PARAMS, lgb.Dataset(x, y, free_raw_data=False), num_boost_round=10)
    ours = lgb.train(
        PARAMS,
        lgb.Dataset(x, y, free_raw_data=False),
        num_boost_round=15,
        init_model=our_init,
    )

    assert ours.num_trees() == up.num_trees()
    np.testing.assert_allclose(ours.predict(x), up.predict(x), rtol=1e-9, atol=1e-12)
