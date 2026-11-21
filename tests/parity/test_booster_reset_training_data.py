"""Parity for replacing a live Booster's training Dataset."""

from types import ModuleType

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm


@pytest.mark.parametrize("boosting", ["gbdt", "dart"])
@pytest.mark.parametrize("feature_fraction", [1.0, 0.6])
def test_update_with_aligned_training_dataset(boosting: str, feature_fraction: float) -> None:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(128, 5))
    labels = x @ np.arange(1.0, 6.0)
    params = {
        "objective": "regression",
        "boosting": boosting,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "min_data_in_leaf": 4,
        "feature_pre_filter": False,
        "feature_fraction": feature_fraction,
        "verbosity": -1,
    }
    models = []
    for lib in (lightgbm, bankai_lightgbm):
        original = lib.Dataset(x, label=labels, params=params)
        booster = lib.Booster(params, original)
        assert not booster.update()
        replacement = original.create_valid(x[::2], label=-labels[::2], params=params)
        assert not booster.update(train_set=replacement)
        assert booster.current_iteration() == 2
        models.append(booster.predict(x, raw_score=True, num_threads=1))

    np.testing.assert_allclose(models[1], models[0], rtol=1e-12, atol=0)


@pytest.mark.parametrize("lib", [lightgbm, bankai_lightgbm])
def test_update_rejects_different_bin_mappers(lib: ModuleType) -> None:
    x = np.arange(128, dtype=np.float64).reshape(-1, 1)
    params = {"objective": "regression", "num_threads": 1, "verbosity": -1}
    original = lib.Dataset(x, label=x[:, 0], params=params).construct()
    booster = lib.Booster(params, original)
    booster.update()

    replacement = lib.Dataset(x * 10, label=x[:, 0], params=params).construct()
    with pytest.raises(lib.basic.LightGBMError, match="different bin mappers"):
        booster.update(train_set=replacement)


def test_rf_update_with_aligned_training_dataset() -> None:
    rng = np.random.default_rng(83)
    x = rng.normal(size=(256, 5))
    y = 10 + x @ np.arange(1.0, 6.0)
    params = {
        "objective": "regression",
        "boosting": "rf",
        "metric": "l2",
        "num_threads": 1,
        "min_data_in_leaf": 4,
        "feature_fraction": 0.6,
        "verbosity": -1,
    }
    predictions = []
    evaluations = []
    for lib in (lightgbm, bankai_lightgbm):
        original = lib.Dataset(x, label=y, params=params)
        booster = lib.Booster(params, original)
        booster.update()
        booster.update()
        replacement = original.create_valid(x[::2], label=-y[::2], params=params)
        booster.update(train_set=replacement)
        assert booster.current_iteration() == 3
        predictions.append(booster.predict(x, num_threads=1))
        evaluations.append(booster.eval_train()[0][2])

    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(evaluations[1], evaluations[0], rtol=1e-12, atol=0)
