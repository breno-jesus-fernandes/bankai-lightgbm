"""``bagging_by_query`` must match upstream: whole queries are bagged, not single rows.

upstream: src/boosting/bagging.hpp (query branch), src/boosting/gbdt.cpp (Bagging before
GetGradientsWithSampledQueries). Lambdarank gradients are per query, so the model trained
with query bagging must equal upstream's, tree by tree.
"""

from __future__ import annotations

import lightgbm
import numpy as np

import bankai_lightgbm as lgb

PARAMS = {
    "objective": "lambdarank",
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "min_data_in_bin": 1,
    "bagging_by_query": True,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "bagging_seed": 3,
    "eval_at": [5],
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
}


def _ranking_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    sizes = rng.integers(3, 12, size=60)
    n = int(sizes.sum())
    x = rng.normal(size=(n, 5))
    score = x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.3, size=n)
    y = np.clip(np.round(score), 0, 4).astype(np.float64)
    return x, y, sizes


def test_query_bagging_matches_upstream_tree_by_tree() -> None:
    x, y, group = _ranking_data()
    up = lightgbm.Booster(
        params=PARAMS, train_set=lightgbm.Dataset(x, y, group=group, free_raw_data=False)
    )
    ours = lgb.Booster(params=PARAMS, train_set=lgb.Dataset(x, y, group=group, free_raw_data=False))
    for _ in range(10):
        up.update()
        ours.update()
    assert ours.num_trees() == up.num_trees()
    np.testing.assert_allclose(ours.predict(x), up.predict(x), rtol=1e-9, atol=0)
