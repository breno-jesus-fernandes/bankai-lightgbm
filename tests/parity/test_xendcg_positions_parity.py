"""rank_xendcg accepts a position field and matches upstream (PROGRESS T4.1).

Upstream quirk (v4.7.0): `RankXENDCG` does not override `UpdatePositionBiasFactors`, so the
shared `RankingObjective` base keeps every position-bias factor at 0 and the score adjustment
is a no-op. Positions are therefore accepted but **inert** for rank_xendcg — only lambdarank
learns the bias (`src/objective/rank_objective.hpp`). bankai previously raised
`NotImplementedError`; now it accepts positions and trains identically to upstream, which means
identically to training with no positions at all.
"""

from __future__ import annotations

import lightgbm
import numpy as np

import bankai_lightgbm as lgb

PARAMS = {
    "objective": "rank_xendcg",
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
    position = np.concatenate([np.arange(s) for s in sizes]).astype(np.int64)
    position = np.minimum(position, 4)
    return x, relevance.astype(np.float64), sizes, position


def _train(lib, x, y, group, position):
    train_set = lib.Dataset(x, y, group=group, position=position, free_raw_data=False)
    booster = lib.Booster(params=PARAMS, train_set=train_set)
    for _ in range(12):
        booster.update()
    return booster


def test_xendcg_with_positions_matches_upstream_tree_by_tree() -> None:
    x, y, group, position = _data()
    up = _train(lightgbm, x, y, group, position)
    ours = _train(lgb, x, y, group, position)
    assert ours.num_trees() == up.num_trees()
    np.testing.assert_allclose(ours.predict(x), up.predict(x), rtol=1e-9, atol=0)


def test_xendcg_positions_are_inert() -> None:
    # Upstream learns no bias for xendcg, so positions must not change the model at all.
    x, y, group, position = _data()
    with_positions = _train(lgb, x, y, group, position)
    no_positions = lgb.Booster(
        params=PARAMS, train_set=lgb.Dataset(x, y, group=group, free_raw_data=False)
    )
    for _ in range(12):
        no_positions.update()
    np.testing.assert_array_equal(with_positions.predict(x), no_positions.predict(x))
