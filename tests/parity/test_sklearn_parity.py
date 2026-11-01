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


def test_eval_set_and_early_stopping_match_upstream() -> None:
    x, y = _data(n=900)
    xtr, ytr, xva, yva = x[:600], y[:600], x[600:], y[600:]
    params = dict(PARAMS, n_estimators=200)
    want = up.LGBMRegressor(**params).fit(
        xtr,
        ytr,
        eval_set=[(xva, yva)],
        eval_metric="l2",
        callbacks=[up.early_stopping(5, verbose=False)],
    )
    got = bk.LGBMRegressor(**params).fit(
        xtr,
        ytr,
        eval_set=[(xva, yva)],
        eval_metric="l2",
        callbacks=[bk.early_stopping(5, verbose=False)],
    )
    assert got.best_iteration_ == want.best_iteration_
    assert list(got.evals_result_) == list(want.evals_result_)
    np.testing.assert_allclose(
        got.evals_result_["valid_0"]["l2"], want.evals_result_["valid_0"]["l2"], rtol=1e-12
    )
    np.testing.assert_allclose(got.predict(xva), want.predict(xva), rtol=1e-12, atol=1e-15)


def test_multiclass_matches_upstream() -> None:
    x, y = _data(n=900)
    labels = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))
    want = up.LGBMClassifier(**PARAMS).fit(x, labels)
    got = bk.LGBMClassifier(**PARAMS).fit(x, labels)
    assert got.n_classes_ == want.n_classes_ == 3
    np.testing.assert_allclose(got.predict_proba(x), want.predict_proba(x), rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(got.predict(x), want.predict(x))


def test_feature_importances_match_upstream() -> None:
    x, y = _data()
    want = up.LGBMRegressor(**PARAMS).fit(x, y)
    got = bk.LGBMRegressor(**PARAMS).fit(x, y)
    np.testing.assert_array_equal(got.feature_importances_, want.feature_importances_)
    assert got.n_features_in_ == want.n_features_in_ == x.shape[1]


def test_pandas_feature_names_match_upstream() -> None:
    pd = pytest.importorskip("pandas")
    x, y = _data()
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    want = up.LGBMRegressor(**PARAMS).fit(df, y)
    got = bk.LGBMRegressor(**PARAMS).fit(df, y)
    assert list(got.feature_names_in_) == list(want.feature_names_in_)
    np.testing.assert_allclose(got.predict(df), want.predict(df), rtol=1e-12, atol=1e-15)


def test_mismatched_lengths_raise_like_upstream() -> None:
    # sklearn validates the inputs before LightGBM sees them, so both raise the same error.
    x, y = _data()
    with pytest.raises(ValueError):
        up.LGBMRegressor(**PARAMS).fit(x, y[:-1])
    with pytest.raises(ValueError):
        bk.LGBMRegressor(**PARAMS).fit(x, y[:-1])


def test_pandas_model_text_matches_upstream() -> None:
    pd = pytest.importorskip("pandas")
    x, y = _data()
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    want = up.LGBMRegressor(**PARAMS).fit(df, y).booster_.model_to_string()
    got = bk.LGBMRegressor(**PARAMS).fit(df, y).booster_.model_to_string()
    trees = lambda text: text.split("\nparameters:")[0]  # noqa: E731
    assert trees(got) == trees(want)
