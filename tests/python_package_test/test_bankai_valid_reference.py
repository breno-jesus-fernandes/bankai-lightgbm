"""Validation sets are binned with the training set's bin mappers (upstream engine.py v4.7.0).

upstream: python-package/lightgbm/engine.py (v4.7.0) train() calls
``valid_data._update_params(params).set_reference(train_set)`` before the Booster is created.
"""

import numpy as np
import pytest

import bankai_lightgbm as lgb
from bankai_lightgbm import basic


@pytest.fixture
def captured_native_log():
    messages: list[str] = []

    class Capture:
        def info(self, msg: str) -> None:
            messages.append(msg)

        def warning(self, msg: str) -> None:
            messages.append(msg)

    saved = (basic._LOGGER, basic._INFO_METHOD_NAME, basic._WARNING_METHOD_NAME)
    lgb.register_logger(Capture())
    yield messages
    basic._LOGGER, basic._INFO_METHOD_NAME, basic._WARNING_METHOD_NAME = saved


def _params():
    return {"objective": "binary", "metric": "binary_logloss", "verbose": -1, "num_leaves": 8, "seed": 1}


def test_validation_set_uses_training_bin_mappers():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    # Validation rows follow a shifted, wider distribution, so their own bins would differ.
    Xv = rng.normal(loc=0.3, scale=1.5, size=(300, 4))
    yv = (Xv[:, 0] + 0.5 * Xv[:, 1] > 0).astype(int)
    train_set = lgb.Dataset(X, y)
    valid_set = lgb.Dataset(Xv, yv)
    lgb.train(_params(), train_set, num_boost_round=3, valid_sets=[valid_set])
    assert valid_set.reference is train_set
    assert [valid_set.feature_num_bin(i) for i in range(4)] == [train_set.feature_num_bin(i) for i in range(4)]


def test_separately_built_validation_set_does_not_repeat_native_warning(captured_native_log):
    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    train_set = lgb.Dataset(X, y, categorical_feature=[1])
    valid_set = lgb.Dataset(X, y, categorical_feature=[1])
    lgb.train({**_params(), "metric": "auc", "verbose": 1}, train_set, num_boost_round=1, valid_sets=[valid_set])
    warnings_seen = [m for m in captured_native_log if "There are no meaningful features" in m]
    assert len(warnings_seen) == 1
