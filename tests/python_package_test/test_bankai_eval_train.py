"""``Booster.eval_train`` returns the training data's metrics (upstream ``Booster.eval_train``)."""

import numpy as np
import pytest

import bankai_lightgbm as lgb


def _data():
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(200, 4))
    y = (X[:, 0] + 0.1 * rng.uniform(size=200) > 0.5).astype(float)
    return X, y


def test_eval_train_matches_eval_on_the_training_set():
    X, y = _data()
    train = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "metric": "binary_logloss", "verbose": -1, "num_threads": 1}
    # Build the Booster directly: lgb.train frees the training Dataset before it returns.
    bst = lgb.Booster(params, train)
    for _ in range(5):
        bst.update()

    results = bst.eval_train()

    assert len(results) == 1
    dataset_name, metric_name, value, higher_better = results[0]
    assert dataset_name == "training"
    assert metric_name == "binary_logloss"
    assert higher_better is False
    expected = bst.eval(train, "training")[0]
    assert value == pytest.approx(expected[2])


def test_eval_train_without_training_set_raises():
    X, y = _data()
    bst = lgb.train(
        {"objective": "binary", "verbose": -1, "num_threads": 1},
        lgb.Dataset(X, label=y),
        num_boost_round=1,
    )
    loaded = lgb.Booster(model_str=bst.model_to_string())

    with pytest.raises(lgb.basic.LightGBMError, match="no training Dataset"):
        loaded.eval_train()
