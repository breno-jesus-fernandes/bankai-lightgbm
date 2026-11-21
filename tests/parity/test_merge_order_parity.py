"""MergeFrom prepends the other ensemble, checked against upstream's actual C API."""

import lightgbm
import numpy as np
import pytest
from lightgbm.basic import _LIB, _safe_call

import bankai_lightgbm as lgb


@pytest.mark.parametrize("objective", ["regression", "binary", "multiclass"])
def test_merge_preserves_upstream_tree_order_and_iteration_windows(objective: str) -> None:
    rng = np.random.default_rng(117)
    x = rng.normal(size=(256, 4))
    y_a = 2 * x[:, 0] - x[:, 1] + rng.normal(scale=0.2, size=256)
    y_b = 3 * x[:, 2] + np.sin(x[:, 3]) + rng.normal(scale=0.3, size=256)
    params = {
        "objective": objective,
        "num_threads": 1,
        "deterministic": True,
        "verbosity": -1,
        "num_leaves": 7,
        "min_data_in_leaf": 3,
        "seed": 17,
    }
    if objective == "binary":
        y_a = (y_a > 0).astype(float)
        y_b = (y_b > 0).astype(float)
    elif objective == "multiclass":
        y_a = np.digitize(y_a, [-0.5, 0.5]).astype(float)
        y_b = np.digitize(y_b, [-0.5, 0.5]).astype(float)
        params["num_class"] = 3
    a = lightgbm.train(params, lightgbm.Dataset(x, y_a), num_boost_round=2)
    b = lightgbm.train(params, lightgbm.Dataset(x, y_b), num_boost_round=3)
    ours = lgb.Booster(model_str=a.model_to_string())
    other = lgb.Booster(model_str=b.model_to_string())
    # lightgbm's Python API exposes no public merge, so call its defining C API directly.
    _safe_call(_LIB.LGBM_BoosterMerge(a._handle, b._handle))
    ours.merge(other)
    assert ours.num_trees() == a.num_trees()
    np.testing.assert_array_equal(ours.predict(x, pred_leaf=True), a.predict(x, pred_leaf=True))
    for start, count in [(0, 1), (1, 2), (3, 2), (0, -1)]:
        np.testing.assert_array_equal(
            ours.predict(x, start_iteration=start, num_iteration=count, raw_score=True),
            a.predict(x, start_iteration=start, num_iteration=count, raw_score=True),
        )
