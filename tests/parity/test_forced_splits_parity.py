"""``forcedsplits_filename`` must match upstream: the first splits are the forced ones, in
breadth-first order, and the rest of each tree is grown normally.

upstream: src/treelearner/serial_tree_learner.cpp (ForceSplits, FindAllForceFeatures)
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

BASE = {
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "min_data_in_bin": 1,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 3,
    "force_col_wise": True,
}


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(800, 5))
    y = 2.0 * (x[:, 0] > 0.5) + x[:, 2] - 0.5 * x[:, 1] + rng.normal(scale=0.1, size=800)
    return x, y


FORCED = {
    "two_level": {"feature": 0, "threshold": 0.5, "right": {"feature": 2, "threshold": 10.0}},
    "both_children": {
        "feature": 0,
        "threshold": 0.5,
        "left": {"feature": 1, "threshold": -0.2},
        "right": {"feature": 2, "threshold": 0.3},
    },
}


@pytest.mark.parametrize("case", sorted(FORCED))
@pytest.mark.parametrize("feature_fraction", [1.0, 0.6])
@pytest.mark.parametrize("with_nan", [False, True])
def test_forced_splits_match_upstream(
    tmp_path: Path, case: str, feature_fraction: float, with_nan: bool
) -> None:
    x, y = _data()
    if with_nan:
        # Missing values in the forced features exercise the NaN-as-missing scan.
        x[::7, 0] = np.nan
        x[::5, 2] = np.nan
    split_file = tmp_path / "forced.json"
    split_file.write_text(json.dumps(FORCED[case]))
    params = {
        **BASE,
        "feature_fraction": feature_fraction,
        "forcedsplits_filename": str(split_file),
    }
    up = lightgbm.train(params, lightgbm.Dataset(x, y), num_boost_round=8)
    ours = lgb.train(params, lgb.Dataset(x, y), num_boost_round=8)
    assert ours.num_trees() == up.num_trees()
    np.testing.assert_allclose(ours.predict(x), up.predict(x), rtol=1e-9, atol=0)
    up_tree = up.dump_model()["tree_info"][0]["tree_structure"]
    our_tree = ours.dump_model()["tree_info"][0]["tree_structure"]
    assert our_tree["split_feature"] == up_tree["split_feature"] == 0
    assert our_tree["threshold"] == up_tree["threshold"]
