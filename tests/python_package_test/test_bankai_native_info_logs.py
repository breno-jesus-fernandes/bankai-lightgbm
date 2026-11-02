"""Native Info lines emitted while building and starting a binary training run match upstream.

upstream: src/objective/binary_objective.hpp (v4.7.0) Init and BoostFromScore,
src/treelearner/feature_histogram.hpp DynamicChangeSize ("Total Bins"),
src/treelearner/serial_tree_learner.cpp Init ("Number of data points in the train set").
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


def test_binary_training_logs_class_counts_bins_and_boost_from_score(captured_native_log):
    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    lgb.train(
        {"objective": "binary", "verbose": 1},
        lgb.Dataset(X, y, categorical_feature=[1]),
        num_boost_round=1,
    )
    assert captured_native_log[:5] == [
        "[LightGBM] [Warning] There are no meaningful features which satisfy the provided configuration. "
        "Decreasing Dataset parameters min_data_in_bin or min_data_in_leaf and re-constructing Dataset "
        "might resolve this warning.",
        "[LightGBM] [Info] Number of positive: 2, number of negative: 2",
        "[LightGBM] [Info] Total Bins 0",
        "[LightGBM] [Info] Number of data points in the train set: 4, number of used features: 0",
        "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000",
    ]
