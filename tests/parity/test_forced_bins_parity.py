"""``forcedbins_filename`` sets per-feature upper bin bounds; the model must match upstream."""

from __future__ import annotations

from pathlib import Path

import lightgbm
import numpy as np

import bankai_lightgbm as lgb

FORCED_BINS = Path(__file__).resolve().parents[2] / "examples" / "regression" / "forced_bins.json"

PARAMS = {
    "objective": "regression_l1",
    "max_bin": 5,
    "num_leaves": 2,
    "min_data_in_leaf": 1,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 3,
}


def _data() -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((100, 2))
    x[:, 0] = np.arange(0, 1, 0.01)
    x[:, 1] = -np.arange(0, 1, 0.01)
    return x, np.arange(0, 1, 0.01)


def test_forced_bins_model_matches_upstream() -> None:
    x, y = _data()
    params = {**PARAMS, "forcedbins_filename": str(FORCED_BINS)}

    upstream = lightgbm.train(params, lightgbm.Dataset(x, label=y), num_boost_round=20)
    ours = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=20)

    assert ours.model_to_string() == upstream.model_to_string()


def test_forced_bins_bin_counts_match_upstream() -> None:
    x, y = _data()
    params = {**PARAMS, "forcedbins_filename": str(FORCED_BINS)}
    upstream_ds = lightgbm.Dataset(x, label=y, params=params).construct()
    ours_ds = lgb.Dataset(x, label=y, params=params).construct()
    for feature in range(2):
        assert ours_ds.feature_num_bin(feature) == upstream_ds.feature_num_bin(feature)
