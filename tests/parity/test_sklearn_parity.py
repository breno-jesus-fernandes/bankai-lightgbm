"""sklearn estimators: same data and params as upstream `lightgbm.LGBM*`, outputs compared."""

from __future__ import annotations

import inspect

import lightgbm as up
import numpy as np
import pytest

import bankai_lightgbm as bk

PARAMS = dict(
    n_estimators=30,
    num_leaves=15,
    min_child_samples=5,
    learning_rate=0.1,
    random_state=7,
    n_jobs=1,
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
)


def _data(n: int = 800, f: int = 6, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f))
    y = x[:, 0] - 2.0 * x[:, 1] + 0.1 * rng.normal(size=n)
    return x, y


def test_public_names_are_exported() -> None:
    for name in ["LGBMModel", "LGBMRegressor", "LGBMClassifier", "LGBMRanker"]:
        assert hasattr(bk, name), name


@pytest.mark.parametrize("name", ["LGBMModel", "LGBMRegressor", "LGBMClassifier", "LGBMRanker"])
def test_init_signature_matches_upstream(name: str) -> None:
    got = inspect.signature(getattr(bk, name).__init__)
    want = inspect.signature(getattr(up, name).__init__)
    assert list(got.parameters) == list(want.parameters)
    for p in want.parameters:
        assert got.parameters[p].default == want.parameters[p].default, p


def test_regressor_predictions_match_upstream() -> None:
    x, y = _data()
    want = up.LGBMRegressor(**PARAMS).fit(x, y).predict(x)
    got = bk.LGBMRegressor(**PARAMS).fit(x, y).predict(x)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def test_classifier_probabilities_match_upstream() -> None:
    x, y = _data()
    labels = (y > 0).astype(int)
    want_m = up.LGBMClassifier(**PARAMS).fit(x, labels)
    got_m = bk.LGBMClassifier(**PARAMS).fit(x, labels)
    np.testing.assert_array_equal(got_m.classes_, want_m.classes_)
    np.testing.assert_allclose(
        got_m.predict_proba(x), want_m.predict_proba(x), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_array_equal(got_m.predict(x), want_m.predict(x))


def test_ranker_predictions_match_upstream() -> None:
    x, y = _data(n=800)
    group = [40] * 20
    relevance = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75])).astype(int)
    want = up.LGBMRanker(**PARAMS).fit(x, relevance, group=group).predict(x)
    got = bk.LGBMRanker(**PARAMS).fit(x, relevance, group=group).predict(x)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
