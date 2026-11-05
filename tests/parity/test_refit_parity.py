"""``Booster.refit`` must match upstream LightGBM.

The same upstream model is refitted on the same new data with the same ``decay_rate`` by
``lightgbm`` and ``bankai_lightgbm``. The refitted tree structures must be identical and
the leaf values must agree within ``rtol=1e-12`` (deterministic, single-threaded training).
"""

from __future__ import annotations

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

PARAMS = {
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
}


def _data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 2.0 * X[:, 0] + X[:, 1] ** 2 + rng.normal(scale=0.1, size=n)
    return X, y


def _structure_and_leaves(dump: dict) -> tuple[list, list]:
    structures, leaves = [], []
    for tree in dump["tree_info"]:
        structures.append(tree["num_leaves"])
        leaves.append(_leaf_values(tree["tree_structure"]))
    return structures, leaves


def _leaf_values(node: dict) -> list[float]:
    if "leaf_value" in node:
        return [node["leaf_value"]]
    return _leaf_values(node["left_child"]) + _leaf_values(node["right_child"])


@pytest.mark.parametrize("objective", ["regression", "binary"])
@pytest.mark.parametrize("decay_rate", [0.9, 0.0, 1.0])
def test_refit_matches_upstream(objective: str, decay_rate: float) -> None:
    X, y = _data(1, 500)
    X_new, y_new = _data(2, 300)
    if objective == "binary":
        y = (y > 0).astype(float)
        y_new = (y_new > 0).astype(float)
    params = {**PARAMS, "objective": objective}
    upstream = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=10)
    bankai_model = lgb.Booster(model_str=upstream.model_to_string())

    want = upstream.refit(X_new, y_new, decay_rate=decay_rate)
    got = bankai_model.refit(X_new, y_new, decay_rate=decay_rate)

    want_structures, want_leaves = _structure_and_leaves(want.dump_model())
    got_structures, got_leaves = _structure_and_leaves(got.dump_model())
    assert got_structures == want_structures
    for got_tree, want_tree in zip(got_leaves, want_leaves, strict=True):
        np.testing.assert_allclose(got_tree, want_tree, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(
        got.predict(X_new, pred_leaf=True), want.predict(X_new, pred_leaf=True)
    )
    np.testing.assert_allclose(got.predict(X_new), want.predict(X_new), rtol=1e-12, atol=0)
