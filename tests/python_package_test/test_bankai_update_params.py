"""Changing Dataset parameters after construction (upstream ``Dataset._update_params``)."""

import numpy as np
import pytest

import bankai_lightgbm as lgb


def _data():
    rng = np.random.default_rng(0)
    return rng.uniform(size=(100, 2)), rng.uniform(size=100)


def test_binning_parameter_change_is_rejected_when_raw_data_is_freed():
    X, y = _data()
    ds = lgb.Dataset(X, y, params={"max_bin": 100, "verbose": -1}).construct()
    with pytest.raises(lgb.basic.LightGBMError, match="Cannot change max_bin"):
        lgb.train({"max_bin": 150, "verbose": -1}, ds, num_boost_round=1)


def test_binning_parameter_change_is_allowed_when_raw_data_is_kept():
    X, y = _data()
    ds = lgb.Dataset(X, y, params={"max_bin": 100, "verbose": -1}, free_raw_data=False).construct()
    lgb.train({"max_bin": 150, "verbose": -1}, ds, num_boost_round=1)
