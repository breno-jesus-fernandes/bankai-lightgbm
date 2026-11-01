"""``num_iteration <= 0`` predicts all iterations from ``start_iteration``, as upstream does.

upstream: src/application/predictor.hpp (v4.7.0) and python-package/lightgbm/basic.py
``Booster.predict``: a non-positive ``num_iteration`` means "to the end of the model".
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

import bankai_lightgbm as bl

PARAMS = {
    "objective": "regression",
    "verbose": -1,
    "num_leaves": 7,
    "seed": 1,
    "deterministic": True,
    "num_threads": 1,
}
PREDICT_KINDS = {
    "raw": {"raw_score": True},
    "leaf": {"pred_leaf": True},
    "contrib": {"pred_contrib": True},
}


@pytest.fixture(scope="module")
def models() -> tuple[bl.Booster, lgb.Booster, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] + rng.normal(scale=0.1, size=200)
    ours = bl.train(PARAMS, bl.Dataset(X, y), 20)
    reference = lgb.train(PARAMS, lgb.Dataset(X, y), 20)
    return ours, reference, X


@pytest.mark.parametrize("kind", sorted(PREDICT_KINDS))
@pytest.mark.parametrize("num_iteration", [0, -1])
def test_non_positive_num_iteration_predicts_to_end_of_model(models, kind, num_iteration):
    ours, reference, X = models
    kwargs = PREDICT_KINDS[kind]
    got = ours.predict(X, start_iteration=10, num_iteration=num_iteration, **kwargs)
    want = reference.predict(X, start_iteration=10, num_iteration=num_iteration, **kwargs)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


def test_leaf_prediction_with_no_iterations_left_is_empty(models):
    """Leaf indices of a start past the last tree are an (n, 0) array, not a panic."""
    ours, reference, X = models
    got = ours.predict(X, start_iteration=25, pred_leaf=True)
    want = reference.predict(X, start_iteration=25, pred_leaf=True)
    assert want.shape == (X.shape[0], 0)
    assert got.shape == want.shape
