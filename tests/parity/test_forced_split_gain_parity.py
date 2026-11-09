"""Forced splits must record upstream's gain in every tree, not only the first (ROADMAP P3).

Upstream's root `LeafSplits::Init(gradients, hessians)` sets no weight, so the forced split
at the root reads the weight the previous tree left behind as its parent output
(`src/treelearner/leaf_splits.hpp` v4.7.0, `SerialTreeLearner::BeforeTrain`). The recorded
`split_gain` of that forced split depends on it from the second tree on, so the whole model
text must match, not only the predictions.
"""

from __future__ import annotations

import json

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

BASE = {
    "objective": "regression",
    "num_leaves": 7,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
}


def _data(seed: int = 9, n: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 6, size=n).astype(float)
    num = rng.normal(size=n)
    X = np.column_stack([cat, num])
    y = np.where(np.isin(cat, [1.0, 4.0]), 2.0, -1.0) + 0.5 * num + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.mark.parametrize("forced", [
    {"feature": 1, "threshold": 0.0},
    {"feature": 1, "threshold": 0.5, "left": {"feature": 1, "threshold": -0.5}},
])
def test_forced_split_gain_matches_upstream_in_every_tree(tmp_path, forced: dict) -> None:
    X, y = _data()
    path = tmp_path / "forced.json"
    path.write_text(json.dumps(forced))
    params = {**BASE, "forcedsplits_filename": str(path)}
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y, categorical_feature=[0]), 4)
    ours = lgb.train(params, lgb.Dataset(X, y, categorical_feature=[0]), 4)
    assert ours.model_to_string() == theirs.model_to_string()
