"""``Booster.params`` carries the training Dataset params, as upstream ``Booster.__init__`` does."""

from __future__ import annotations

import lightgbm
import numpy as np

import bankai_lightgbm as lgb

PARAMS = {"objective": "binary", "num_leaves": 7, "verbose": -1, "num_threads": 1}


def test_booster_params_include_the_training_dataset_params() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = (X[:, 0] > 0).astype(float)
    dataset_params = {"max_bin": 63}

    upstream = lightgbm.Booster(
        params=PARAMS, train_set=lightgbm.Dataset(X, y, params=dataset_params)
    )
    ours = lgb.Booster(params=PARAMS, train_set=lgb.Dataset(X, y, params=dataset_params))

    assert ours.params["max_bin"] == upstream.params["max_bin"] == 63
    assert ours.params == upstream.params
