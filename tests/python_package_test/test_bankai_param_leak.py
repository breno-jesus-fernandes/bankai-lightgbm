"""Training parameters must not leak into later calls that reuse the same Dataset (upstream: engine.train reverses its Dataset update)."""

import numpy as np

import bankai_lightgbm as lgb


def test_goss_from_an_earlier_train_does_not_leak_into_a_later_bagging_train():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 5))
    y = X[:, 0] + rng.standard_normal(500)
    train_set = lgb.Dataset(X, y)
    base = {"objective": "regression", "verbose": -1, "num_threads": 1}

    lgb.train({**base, "data_sample_strategy": "goss"}, train_set, num_boost_round=2)
    # upstream: the GOSS setting is not carried over, so bagging is accepted.
    bst = lgb.train({**base, "bagging_freq": 1, "bagging_fraction": 0.8}, train_set, num_boost_round=2)
    assert bst.num_trees() == 2
    assert "data_sample_strategy" not in train_set.params
