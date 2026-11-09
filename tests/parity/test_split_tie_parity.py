"""Equal split gains must break ties the way upstream does (SPEC-NUM-7, ROADMAP P3).

Upstream's `SplitInfo::operator>` picks the higher gain, and on an exact tie the smaller feature
index. Duplicate columns make every split of the copies tie exactly, so the chosen feature is the
lowest index among the copies, and the model text shows it. The model must match upstream byte
for byte, with and without bagging, and for a column that is a copy of the label-correlated one.
"""

from __future__ import annotations

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


def _tied_columns(seed: int = 11, n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 2))
    # Features 0, 2 and 3 are copies of one column, so their best splits have identical gain.
    X = np.column_stack([base[:, 0], base[:, 1], base[:, 0], base[:, 0]])
    y = 1.5 * base[:, 0] - base[:, 1] + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.mark.parametrize("extra", [{}, {"bagging_fraction": 0.7, "bagging_freq": 1}])
def test_tied_gains_choose_the_lowest_feature_like_upstream(extra: dict) -> None:
    X, y = _tied_columns()
    params = {**BASE, **extra}
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=8)
    ours = lgb.train(params, lgb.Dataset(X, y), num_boost_round=8)
    assert ours.model_to_string() == theirs.model_to_string()


def test_tied_gains_split_on_the_lowest_index_among_copies() -> None:
    X, y = _tied_columns()
    theirs = lightgbm.train(BASE, lightgbm.Dataset(X, y), num_boost_round=1)
    first_split_feature = theirs.dump_model()["tree_info"][0]["tree_structure"]["split_feature"]
    # Features 0, 2 and 3 tie; upstream takes the lowest index, 0.
    assert first_split_feature == 0
    ours = lgb.train(BASE, lgb.Dataset(X, y), num_boost_round=1)
    assert ours.dump_model()["tree_info"][0]["tree_structure"]["split_feature"] == 0
