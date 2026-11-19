"""SIMD dense histogram accumulation retains upstream's deterministic tree structure."""

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb


@pytest.mark.parametrize("objective", ["regression", "binary"])
@pytest.mark.parametrize("max_bin", [15, 255, 511])
def test_simd_histogram_model_matches_upstream(objective: str, max_bin: int) -> None:
    rng = np.random.default_rng(61)
    x = rng.normal(size=(1801, 7))
    x[rng.random(x.shape) < 0.07] = np.nan
    y = np.nan_to_num(x[:, 0]) * 2 + rng.normal(scale=0.2, size=x.shape[0])
    if objective == "binary":
        y = (y > np.median(y)).astype(float)
    params = {
        "objective": objective,
        "max_bin": max_bin,
        "min_data_in_bin": 1,
        "num_leaves": 13,
        "min_data_in_leaf": 7,
        "bagging_fraction": 0.73,
        "bagging_freq": 1,
        "force_col_wise": True,
        "deterministic": True,
        "num_threads": 1,
        "seed": 61,
        "verbose": -1,
    }
    ours = lgb.train(params, lgb.Dataset(x, y), num_boost_round=7)
    theirs = lightgbm.train(params, lightgbm.Dataset(x, y), num_boost_round=7)
    assert ours.model_to_string() == theirs.model_to_string()
