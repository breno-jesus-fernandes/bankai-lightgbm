"""Bounded histogram caches must evict/rebuild with upstream's LRU policy."""

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb


@pytest.mark.parametrize("pool_mb", [0.000001, 0.05, 0.2, 100.0])
@pytest.mark.parametrize("row_wise", [False, True])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_histogram_pool_matches_upstream(pool_mb: float, row_wise: bool, objective: str) -> None:
    rng = np.random.default_rng(73)
    x = rng.normal(size=(4097, 12))
    x[:, 3:][rng.random((len(x), 9)) < 0.8] = 0.0
    x[rng.random(x.shape) < 0.03] = np.nan
    y = np.nan_to_num(x[:, 0]) * 2 + np.nan_to_num(x[:, 5]) + rng.normal(scale=0.1, size=len(x))
    if objective == "binary":
        y = (y > np.median(y)).astype(float)
    params = {
        "objective": objective,
        "histogram_pool_size": pool_mb,
        "max_bin": 63,
        "num_leaves": 31,
        "min_data_in_leaf": 7,
        "num_threads": 1,
        "deterministic": True,
        "seed": 73,
        "verbose": -1,
        "force_row_wise": row_wise,
        "force_col_wise": not row_wise,
        "bagging_fraction": 0.73,
        "bagging_freq": 1,
        "feature_fraction": 0.8,
    }
    ours = lgb.train(params, lgb.Dataset(x, y), num_boost_round=12)
    theirs = lightgbm.train(params, lightgbm.Dataset(x, y), num_boost_round=12)
    assert ours.model_to_string() == theirs.model_to_string()


@pytest.mark.parametrize("threads", [1, 2, 3])
@pytest.mark.parametrize("method", ["basic", "intermediate", "advanced"])
@pytest.mark.parametrize("row_wise", [False, True])
def test_histogram_pool_eviction_with_monotone_constraints(
    threads: int, method: str, row_wise: bool
) -> None:
    rng = np.random.default_rng(79)
    x = rng.normal(size=(2501, 6))
    y = x[:, 0] * 2 + x[:, 1] * 0.3 + rng.normal(scale=0.1, size=len(x))
    params = {
        "hist_pool_size": 0.000001,
        "max_bin": 63,
        "num_leaves": 31,
        "min_data_in_leaf": 7,
        "monotone_constraints": [1, 0, 0, 0, 0, 0],
        "monotone_constraints_method": method,
        "num_threads": threads,
        "deterministic": True,
        "seed": 79,
        "verbose": -1,
        "force_row_wise": row_wise,
        "force_col_wise": not row_wise,
    }
    ours = lgb.train(params, lgb.Dataset(x, y), num_boost_round=12)
    theirs = lightgbm.train(params, lightgbm.Dataset(x, y), num_boost_round=12)
    assert ours.model_to_string() == theirs.model_to_string()


@pytest.mark.parametrize("extra_trees", [False, True])
def test_histogram_pool_reset_preserves_sampling_streams(extra_trees: bool) -> None:
    rng = np.random.default_rng(83)
    x = rng.normal(size=(1001, 6))
    y = x[:, 0] + x[:, 2] * 0.3 + rng.normal(scale=0.1, size=len(x))
    params = {
        "histogram_pool_size": 0.000001,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "max_bin": 63,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "feature_fraction": 0.6,
        "extra_trees": extra_trees,
        "force_col_wise": True,
        "deterministic": True,
        "num_threads": 1,
        "seed": 83,
        "verbose": -1,
    }
    models = []
    for lib in (lgb, lightgbm):
        booster = lib.Booster(params=params, train_set=lib.Dataset(x, y, free_raw_data=False))
        for _ in range(6):
            booster.update()
        booster.reset_parameter({"histogram_pool_size": 100.0})
        for _ in range(6):
            booster.update()
        models.append(booster)
    np.testing.assert_array_equal(
        models[0].predict(x, pred_leaf=True), models[1].predict(x, pred_leaf=True)
    )
    np.testing.assert_allclose(models[0].predict(x), models[1].predict(x), rtol=1e-12, atol=1e-12)
