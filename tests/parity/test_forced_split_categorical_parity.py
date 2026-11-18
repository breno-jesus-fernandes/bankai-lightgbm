"""Forced splits on a categorical feature must match upstream (ROADMAP P3, PROGRESS T3.3).

upstream: src/treelearner/serial_tree_learner.cpp (`ForceSplits` reads the JSON "threshold"
as a real value and maps it to a bin with `BinThreshold`/`ValueToBin`, with no type branch),
src/treelearner/feature_histogram.hpp (`GatherInfoForThresholdCategorical`): the forced bin
is a one-vs-rest categorical split (that bin goes left, `default_left = false`,
`cat_threshold = {bin}`), using `lambda_l2` without `cat_l2`.

The whole model text must match, not only the predictions: the recorded `split_gain` of the
forced split depends on the carried root weight from the second tree on (see
`test_forced_split_gain_parity.py`).
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


@pytest.mark.parametrize(
    "forced",
    [
        {"feature": 0, "threshold": 1},
        {"feature": 0, "threshold": 4},
        {"feature": 0, "threshold": 1, "left": {"feature": 0, "threshold": 4}},
        {"feature": 0, "threshold": 2, "right": {"feature": 1, "threshold": 0.0}},
    ],
)
def test_forced_split_on_categorical_matches_upstream(tmp_path, forced: dict) -> None:
    x, y = _data()
    path = tmp_path / "forced.json"
    path.write_text(json.dumps(forced))
    params = {**BASE, "forcedsplits_filename": str(path)}
    theirs = lightgbm.train(params, lightgbm.Dataset(x, y, categorical_feature=[0]), 4)
    ours = lgb.train(params, lgb.Dataset(x, y, categorical_feature=[0]), 4)
    # The first forced split must be categorical and on the forced feature.
    up_tree = theirs.dump_model()["tree_info"][0]["tree_structure"]
    assert up_tree["split_feature"] == 0 and up_tree["decision_type"] == "=="
    assert ours.model_to_string() == theirs.model_to_string()
