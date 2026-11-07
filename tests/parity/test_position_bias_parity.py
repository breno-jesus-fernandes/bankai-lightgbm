"""Unbiased LambdaMART: position bias factors must match upstream tree by tree.

upstream: src/objective/rank_objective.hpp (LambdarankNDCG::GetGradientsWithSampledQueries,
UpdatePositionBiasFactors)
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
    "learning_rate": 0.1,
    "lambdarank_position_bias_regularization": 0.5,
    "eval_at": [3],
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 9,
}


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    sizes = rng.integers(4, 10, size=50)
    n = int(sizes.sum())
    x = rng.normal(size=(n, 4))
    relevance = np.clip(np.round(x[:, 0] + 0.4 * x[:, 1] + rng.normal(scale=0.2, size=n)), 0, 3)
    # Position of each row inside its query; the bias depends on it.
    position = np.concatenate([np.arange(s) for s in sizes]).astype(np.int64)
    position = np.minimum(position, 4)
    return x, relevance.astype(np.float64), sizes, position


def test_position_bias_matches_upstream_tree_by_tree() -> None:
    x, y, group, position = _data()
    up = lightgbm.Booster(
        params=PARAMS,
        train_set=lightgbm.Dataset(x, y, group=group, position=position, free_raw_data=False),
    )
    ours = lgb.Booster(
        params=PARAMS,
        train_set=lgb.Dataset(x, y, group=group, position=position, free_raw_data=False),
    )
    for _ in range(12):
        up.update()
        ours.update()
    assert ours.num_trees() == up.num_trees()
    np.testing.assert_allclose(ours.predict(x), up.predict(x), rtol=1e-9, atol=0)
