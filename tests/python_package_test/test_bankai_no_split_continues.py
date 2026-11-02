"""Training keeps running after an iteration finds no splits (upstream engine.train behaviour).

upstream: python-package/lightgbm/engine.py (v4.7.0) train() calls ``booster.update`` and ignores its
``is_finished`` return value, so later rounds still evaluate the validation sets.
"""

import numpy as np

import bankai_lightgbm as lgb


def test_no_split_training_still_evaluates_every_round():
    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    train_set = lgb.Dataset(X, y, categorical_feature=[1])
    valid_set = lgb.Dataset(X, y, categorical_feature=[1], reference=train_set)
    history: dict = {}
    lgb.train(
        {"objective": "binary", "metric": "binary_error", "verbose": -1},
        train_set,
        num_boost_round=10,
        valid_sets=[valid_set],
        callbacks=[lgb.record_evaluation(history)],
    )
    # upstream records one value per round, even though the first round found no split
    assert len(history["valid_0"]["binary_error"]) == 10
