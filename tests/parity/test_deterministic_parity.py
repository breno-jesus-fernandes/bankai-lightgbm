"""`deterministic` must train to upstream's model text (SPEC-NUM-2, ROADMAP P3).

Upstream's `config.cpp` reads `deterministic` only to warn that GPU runs may differ. The CPU
learners do not branch on it (upstream's `config.h` note asks users to set `force_col_wise` or
`force_row_wise` with it). The tests pin that: the flag on CPU must give the same model as
upstream, and bankai must not change the model when it is set. The 1-thread structure rule of
SPEC-NUM-2 is what the comparison checks.
"""

from __future__ import annotations

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

BASE = {
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "num_threads": 1,
    "seed": 7,
}


def _data(seed: int = 5, n: int = 1200, f: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(scale=0.2, size=n)
    return X, y


@pytest.mark.parametrize("layout", [{}, {"force_col_wise": True}, {"force_row_wise": True}])
def test_deterministic_model_matches_upstream(layout: dict) -> None:
    X, y = _data()
    params = {**BASE, **layout, "deterministic": True}
    theirs = lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=10)
    ours = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10)
    assert ours.model_to_string() == theirs.model_to_string()


def test_deterministic_does_not_change_the_cpu_model() -> None:
    X, y = _data()
    off = lgb.train({**BASE, "deterministic": False}, lgb.Dataset(X, y), num_boost_round=10)
    on = lgb.train({**BASE, "deterministic": True}, lgb.Dataset(X, y), num_boost_round=10)
    # The trailing parameters block echoes the flag itself; compare the trees only.
    assert (
        off.model_to_string().split("\nparameters:")[0]
        == on.model_to_string().split("\nparameters:")[0]
    )
