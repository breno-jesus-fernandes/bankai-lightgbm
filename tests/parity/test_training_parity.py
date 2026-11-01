"""Training through the public API must produce upstream's model text byte for byte.

Every training fixture under ``tests/fixtures/train`` is trained with ``lgb.train`` here
and with ``lightgbm.train`` (same data, params, weights, groups and categoricals).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

TRAIN = Path(__file__).resolve().parent.parent / "fixtures" / "train"


def _cases() -> list[str]:
    return sorted(p.name for p in TRAIN.iterdir() if (p / "model.txt").exists())


def _dataset_kwargs(case: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"label": np.load(case / "y.npy")}
    if (case / "w.npy").exists():
        kwargs["weight"] = np.load(case / "w.npy")
    if (case / "group.npy").exists():
        kwargs["group"] = np.load(case / "group.npy")
    cats = json.loads((case / "params.json").read_text()).get("categorical") or []
    kwargs["categorical_feature"] = cats if cats else "auto"
    return kwargs


@pytest.mark.parametrize("case", _cases())
def test_training_matches_upstream_text(case: str) -> None:
    folder = TRAIN / case
    meta = json.loads((folder / "params.json").read_text())
    params: dict[str, Any] = meta["params"]
    rounds: int = meta["num_boost_round"]
    x = np.load(folder / "x.npy")
    kwargs = _dataset_kwargs(folder)

    ours = lgb.train(
        dict(params), lgb.Dataset(x, params=dict(params), **kwargs), num_boost_round=rounds
    )
    theirs = lightgbm.train(
        dict(params),
        lightgbm.Dataset(x, params=dict(params), **kwargs),
        num_boost_round=rounds,
    )
    assert ours.model_to_string() == theirs.model_to_string()


def test_dataset_introspection() -> None:
    x = np.random.default_rng(0).normal(size=(50, 4))
    ds = lgb.Dataset(x, label=np.zeros(50), params={"max_bin": 16, "verbose": -1})
    assert ds.num_data() == 50
    assert ds.num_feature() == 4


def test_booster_update_matches_two_round_upstream_training() -> None:
    rng = np.random.default_rng(481)
    x = rng.normal(size=(80, 3))
    y = x[:, 0] * 0.7 - x[:, 1] * 0.2 + (x[:, 2] > 0).astype(float)
    params: dict[str, Any] = {
        "objective": "regression",
        "verbosity": -1,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "learning_rate": 0.2,
        "boost_from_average": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 481,
        "num_threads": 1,
    }
    train_set = lgb.Dataset(x, label=y, params=dict(params))
    incremental = lgb.Booster(params=dict(params), train_set=train_set)
    assert incremental.update() is False
    assert incremental.update() is False

    expected = lightgbm.train(
        dict(params),
        lightgbm.Dataset(x, label=y, params=dict(params)),
        num_boost_round=2,
    )
    assert incremental.current_iteration() == 2
    # Split-gain diagnostics can differ by a few f32 rounding bits across builds.
    np.testing.assert_allclose(incremental.predict(x), expected.predict(x), rtol=1e-12, atol=1e-15)

    trained_once = lgb.train(
        dict(params), lgb.Dataset(x, label=y, params=dict(params)), num_boost_round=1
    )
    assert trained_once.update() is False
    np.testing.assert_allclose(trained_once.predict(x), expected.predict(x), rtol=1e-12, atol=1e-15)


def test_early_stopping_callback_matches_upstream() -> None:
    rng = np.random.default_rng(17)
    x_train = rng.normal(size=(100, 3))
    y_train = x_train[:, 0] + 0.1 * x_train[:, 1]
    x_valid = rng.normal(size=(60, 3))
    y_valid = -x_valid[:, 0] - 0.1 * x_valid[:, 1]
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "num_leaves": 7,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 31,
        "learning_rate": 0.3,
        "boost_from_average": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 17,
        "num_threads": 1,
    }
    train_set = lgb.Dataset(x_train, label=y_train, params=dict(params))
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, params=dict(params))
    ours = lgb.train(
        dict(params),
        train_set,
        num_boost_round=30,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(2, verbose=False)],
    )
    theirs_train_set = lightgbm.Dataset(x_train, label=y_train, params=dict(params))
    theirs_valid_set = lightgbm.Dataset(
        x_valid, label=y_valid, reference=theirs_train_set, params=dict(params)
    )
    theirs = lightgbm.train(
        dict(params),
        theirs_train_set,
        num_boost_round=30,
        valid_sets=[theirs_valid_set],
        valid_names=["valid"],
        callbacks=[lightgbm.early_stopping(2, verbose=False)],
    )

    assert ours.best_iteration == theirs.best_iteration
    assert ours.current_iteration() == theirs.current_iteration()
    np.testing.assert_allclose(
        ours.predict(x_valid), theirs.predict(x_valid), rtol=1e-12, atol=1e-15
    )
    assert ours.best_score.keys() == theirs.best_score.keys()
    np.testing.assert_allclose(
        ours.best_score["valid"]["l2"], theirs.best_score["valid"]["l2"], rtol=1e-12
    )
    assert ours.eval_valid() == theirs.eval_valid() == []

    automatic = lgb.train(
        {**params, "early_stopping_round": 2},
        lgb.Dataset(x_train, label=y_train, params=dict(params)),
        num_boost_round=30,
        valid_sets=[lgb.Dataset(x_valid, label=y_valid, params=dict(params))],
        valid_names=["valid"],
    )
    assert automatic.best_iteration == theirs.best_iteration
    assert automatic.current_iteration() == theirs.current_iteration()


def test_record_and_log_evaluation_callbacks(caplog: Any) -> None:
    rng = np.random.default_rng(32)
    x_train = rng.normal(size=(50, 2))
    y_train = x_train[:, 0] - x_train[:, 1]
    x_valid = rng.normal(size=(30, 2))
    y_valid = x_valid[:, 0] - x_valid[:, 1]
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "seed": 32,
        "num_threads": 1,
    }
    recorded: dict[str, dict[str, list[float]]] = {}
    caplog.set_level(logging.INFO, logger="bankai_lightgbm")
    ours = lgb.train(
        dict(params),
        lgb.Dataset(x_train, label=y_train, params=dict(params)),
        num_boost_round=3,
        valid_sets=[lgb.Dataset(x_valid, label=y_valid, params=dict(params))],
        valid_names=["valid"],
        callbacks=[lgb.record_evaluation(recorded), lgb.log_evaluation(period=1)],
    )
    expected_results: dict[str, dict[str, list[float]]] = {}
    lightgbm.train(
        dict(params),
        lightgbm.Dataset(x_train, label=y_train, params=dict(params)),
        num_boost_round=3,
        valid_sets=[lightgbm.Dataset(x_valid, label=y_valid, params=dict(params))],
        valid_names=["valid"],
        callbacks=[
            lightgbm.record_evaluation(expected_results),
            lightgbm.log_evaluation(period=1),
        ],
    )

    assert recorded.keys() == expected_results.keys()
    np.testing.assert_allclose(recorded["valid"]["l2"], expected_results["valid"]["l2"])
    assert "[1]\tvalid's l2:" in caplog.text
    assert ours.current_iteration() == 3


def test_reset_parameter_learning_rate_matches_upstream() -> None:
    rng = np.random.default_rng(91)
    x = rng.normal(size=(70, 3))
    y = 0.8 * x[:, 0] - 0.25 * x[:, 1]
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "None",
        "verbosity": -1,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "learning_rate": 0.3,
        "boost_from_average": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 91,
        "num_threads": 1,
    }
    schedule = [0.3, 0.1, 0.05]
    ours = lgb.train(
        dict(params),
        lgb.Dataset(x, label=y, params=dict(params)),
        num_boost_round=len(schedule),
        callbacks=[lgb.reset_parameter(learning_rate=schedule)],
    )
    theirs = lightgbm.train(
        dict(params),
        lightgbm.Dataset(x, label=y, params=dict(params)),
        num_boost_round=len(schedule),
        callbacks=[lightgbm.reset_parameter(learning_rate=schedule)],
    )

    np.testing.assert_allclose(ours.predict(x), theirs.predict(x), rtol=1e-12, atol=1e-15)
    ours_rate = next(
        line for line in ours.model_to_string().splitlines() if line.startswith("[learning_rate:")
    )
    theirs_rate = next(
        line for line in theirs.model_to_string().splitlines() if line.startswith("[learning_rate:")
    )
    assert ours_rate == theirs_rate

    callable_schedule = lgb.train(
        dict(params),
        lgb.Dataset(x, label=y, params=dict(params)),
        num_boost_round=len(schedule),
        callbacks=[lgb.reset_parameter(learning_rate=lambda iteration: schedule[iteration])],
    )
    np.testing.assert_allclose(
        callable_schedule.predict(x), ours.predict(x), rtol=1e-12, atol=1e-15
    )


def test_cv_metrics_match_upstream() -> None:
    rng = np.random.default_rng(120)
    x = rng.normal(size=(90, 3))
    y = (x[:, 0] - 0.4 * x[:, 1] > 0).astype(float)
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "learning_rate": 0.2,
        "boost_from_average": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 120,
        "num_threads": 1,
    }
    ours_history: dict[str, dict[str, list[float]]] = {}
    ours = lgb.cv(
        dict(params),
        lgb.Dataset(x, label=y, params=dict(params)),
        num_boost_round=4,
        nfold=3,
        stratified=False,
        seed=120,
        eval_train_metric=True,
        callbacks=[lgb.record_evaluation(ours_history)],
        return_cvbooster=True,
    )
    theirs_history: dict[str, dict[str, list[float]]] = {}
    theirs = lightgbm.cv(
        dict(params),
        lightgbm.Dataset(x, label=y, params=dict(params)),
        num_boost_round=4,
        nfold=3,
        stratified=False,
        seed=120,
        eval_train_metric=True,
        callbacks=[lightgbm.record_evaluation(theirs_history)],
        return_cvbooster=True,
    )

    assert ours.keys() == theirs.keys()
    assert ours_history.keys() == theirs_history.keys()
    for dataset_name in theirs_history:
        assert ours_history[dataset_name].keys() == theirs_history[dataset_name].keys()
        for metric_name in theirs_history[dataset_name]:
            np.testing.assert_allclose(
                ours_history[dataset_name][metric_name],
                theirs_history[dataset_name][metric_name],
                rtol=1e-12,
                atol=1e-15,
            )
    for name in theirs:
        if name == "cvbooster":
            continue
        np.testing.assert_allclose(ours[name], theirs[name], rtol=1e-12, atol=1e-15)
    cvbooster = ours["cvbooster"]
    assert isinstance(cvbooster, lgb.CVBooster)
    assert cvbooster.current_iteration() == [4, 4, 4]
    restored = lgb.CVBooster().model_from_string(cvbooster.model_to_string())
    assert restored.current_iteration() == [4, 4, 4]


def test_cv_early_stopping_matches_upstream() -> None:
    rng = np.random.default_rng(221)
    x = rng.normal(size=(60, 3))
    y = (x[:, 0] > 0).astype(float)
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "learning_rate": 0.2,
        "boost_from_average": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 221,
        "num_threads": 1,
        "early_stopping_round": 2,
        "early_stopping_min_delta": 1.0,
    }
    ours = lgb.cv(
        dict(params),
        lgb.Dataset(x, label=y, params=dict(params)),
        num_boost_round=10,
        nfold=3,
        stratified=False,
        seed=221,
        return_cvbooster=True,
    )
    theirs = lightgbm.cv(
        dict(params),
        lightgbm.Dataset(x, label=y, params=dict(params)),
        num_boost_round=10,
        nfold=3,
        stratified=False,
        seed=221,
        return_cvbooster=True,
    )

    assert ours["cvbooster"].best_iteration == theirs["cvbooster"].best_iteration == 1
    assert len(ours["valid binary_logloss-mean"]) == len(theirs["valid binary_logloss-mean"]) == 1
    np.testing.assert_allclose(
        ours["valid binary_logloss-mean"],
        theirs["valid binary_logloss-mean"],
        rtol=1e-12,
        atol=1e-15,
    )


def test_custom_evaluation_matches_upstream_train_and_cv() -> None:
    rng = np.random.default_rng(315)
    x = rng.normal(size=(72, 3))
    y = (x[:, 0] - 0.3 * x[:, 1] > 0).astype(float)
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "None",
        "verbosity": -1,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "learning_rate": 0.2,
        "boost_from_average": False,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 315,
        "num_threads": 1,
    }

    def custom_mae(predictions: np.ndarray, dataset: Any) -> list[tuple[str, float, bool]]:
        residual = predictions - dataset.get_label()
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        return [("custom_mae", mae, False), ("custom_rmse", rmse, False)]

    ours_valid = lgb.Dataset(x, label=y, params=dict(params))
    theirs_valid = lightgbm.Dataset(x, label=y, params=dict(params))
    ours = lgb.train(
        dict(params),
        lgb.Dataset(x, label=y, params=dict(params)),
        num_boost_round=3,
        valid_sets=[ours_valid],
        valid_names=["valid"],
        feval=[custom_mae],
        keep_training_booster=True,
    )
    theirs = lightgbm.train(
        dict(params),
        lightgbm.Dataset(x, label=y, params=dict(params)),
        num_boost_round=3,
        valid_sets=[theirs_valid],
        valid_names=["valid"],
        feval=[custom_mae],
        keep_training_booster=True,
    )
    ours_results = ours.eval_valid(feval=custom_mae)
    theirs_results = theirs.eval_valid(feval=custom_mae)
    assert [(row[0], row[1], row[3]) for row in ours_results] == [
        (row[0], row[1], row[3]) for row in theirs_results
    ]
    np.testing.assert_allclose(
        [row[2] for row in ours_results], [row[2] for row in theirs_results], rtol=1e-12
    )

    ours_cv = lgb.cv(
        dict(params),
        lgb.Dataset(x, label=y, params=dict(params)),
        num_boost_round=3,
        nfold=3,
        stratified=False,
        seed=315,
        feval=[custom_mae],
    )
    theirs_cv = lightgbm.cv(
        dict(params),
        lightgbm.Dataset(x, label=y, params=dict(params)),
        num_boost_round=3,
        nfold=3,
        stratified=False,
        seed=315,
        feval=[custom_mae],
    )
    assert ours_cv.keys() == theirs_cv.keys()
    for name in theirs_cv:
        np.testing.assert_allclose(ours_cv[name], theirs_cv[name], rtol=1e-12, atol=1e-15)


def test_train_rejects_unimplemented_options() -> None:
    x = np.random.default_rng(0).normal(size=(40, 2))
    ds = lgb.Dataset(x, label=x[:, 0], params={"verbose": -1})
    # Upstream silently ignores unknown init_model objects; bankai raises instead.
    with pytest.raises(TypeError, match="init_model"):
        lgb.train({"verbose": -1}, ds, init_model=object())
    referenced = lgb.Dataset(x, label=x[:, 0], reference=ds)
    assert referenced.construct().num_data() == len(x)
    with pytest.raises(ValueError, match="greater than zero"):
        lgb.train({"verbose": -1}, ds, num_boost_round=0)


@pytest.mark.parametrize("boosting", ["gbdt", "dart", "rf"])
def test_custom_regression_objective_matches_upstream(boosting: str) -> None:
    rng = np.random.default_rng(501)
    x = rng.normal(size=(80, 3))
    y = x[:, 0] - 0.4 * x[:, 1]

    def objective(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        return predictions - dataset.get_label(), np.ones_like(predictions)

    params: dict[str, Any] = {
        "objective": objective,
        "boosting": boosting,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "metric": "None",
        "verbosity": -1,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
    }
    if boosting == "rf":
        for library in (lgb, lightgbm):
            with pytest.raises(library.basic.LightGBMError, match="RF mode do not support custom"):
                library.train(params, library.Dataset(x, label=y), num_boost_round=3)
        return
    ours = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=3)
    theirs = lightgbm.train(params, lightgbm.Dataset(x, label=y), num_boost_round=3)
    np.testing.assert_allclose(ours.predict(x), theirs.predict(x), rtol=1e-12, atol=1e-15)


def test_custom_multiclass_cv_matches_upstream() -> None:
    rng = np.random.default_rng(812)
    x = rng.normal(size=(90, 4))
    y = np.argmax(x[:, :3], axis=1)

    def objective(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        assert predictions.shape == (len(dataset.get_label()), 3)
        probabilities = np.exp(predictions - predictions.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        targets = np.eye(3)[dataset.get_label().astype(int)]
        return probabilities - targets, 1.5 * probabilities * (1 - probabilities)

    def evaluate(predictions: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
        return "raw_mean", float(np.mean(predictions**2)), False

    params = {
        "objective": objective,
        "num_class": 3,
        "metric": "None",
        "verbosity": -1,
        "num_threads": 1,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "deterministic": True,
        "force_col_wise": True,
    }
    ours = lgb.cv(
        params,
        lgb.Dataset(x, label=y),
        num_boost_round=3,
        nfold=3,
        stratified=False,
        seed=17,
        feval=evaluate,
        return_cvbooster=True,
    )
    theirs = lightgbm.cv(
        params,
        lightgbm.Dataset(x, label=y),
        num_boost_round=3,
        nfold=3,
        stratified=False,
        seed=17,
        feval=evaluate,
        return_cvbooster=True,
    )
    for key in theirs:
        if key != "cvbooster":
            np.testing.assert_allclose(ours[key], theirs[key], rtol=1e-12, atol=1e-15)
    for our_fold, their_fold in zip(
        ours["cvbooster"].boosters, theirs["cvbooster"].boosters, strict=True
    ):
        np.testing.assert_allclose(
            our_fold.predict(x, raw_score=True),
            their_fold.predict(x, raw_score=True),
            rtol=1e-12,
            atol=1e-15,
        )


@pytest.mark.parametrize("invalid", ["unequal", "short", "matrix", "tensor"])
def test_invalid_custom_gradients_match_upstream(invalid: str) -> None:
    x = np.random.default_rng(91).normal(size=(40, 3))
    params = {
        "objective": "none",
        "verbosity": -1,
        "min_data_in_leaf": 2,
        "num_threads": 1,
        "metric": "None",
    }
    errors = []
    for library in (lightgbm, lgb):
        booster = library.Booster(
            params=params, train_set=library.Dataset(x, label=x[:, 0], params=params)
        )

        def valid(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
            return predictions - dataset.get_label(), np.ones_like(predictions)

        booster.update(fobj=valid)
        before = booster.predict(x, raw_score=True)
        iteration = booster.current_iteration()

        def objective(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
            if invalid == "unequal":
                return np.ones(40), np.ones(39)
            if invalid == "short":
                return np.ones(39), np.ones(39)
            shape = (20, 2) if invalid == "matrix" else (40, 1, 1)
            return np.ones(shape), np.ones(shape)

        with pytest.raises((ValueError, TypeError)) as caught:
            booster.update(fobj=objective)
        errors.append((type(caught.value), str(caught.value)))
        assert booster.current_iteration() == iteration
        np.testing.assert_array_equal(booster.predict(x, raw_score=True), before)
        booster.update(fobj=valid)
        assert booster.current_iteration() == iteration + 1
    assert errors[0] == errors[1]


@pytest.mark.parametrize("initial_objective", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("variant", ["gbdt", "dart", "goss", "rf"])
def test_switch_to_custom_objective_matches_upstream(initial_objective: str, variant: str) -> None:
    x = np.random.default_rng(415).normal(size=(80, 3))
    y = (x[:, 0] > 0).astype(float) if initial_objective == "binary" else x[:, 0]
    params = {
        "objective": initial_objective,
        "verbosity": -1,
        "num_threads": 1,
        "min_data_in_leaf": 2,
        "num_leaves": 5,
        "min_data_in_bin": 1,
    }
    if initial_objective == "multiclass":
        y = np.argmax(x, axis=1)
        params["num_class"] = 3
    if variant == "goss":
        params["data_sample_strategy"] = "goss"
    else:
        params["boosting"] = variant
    if variant == "rf":
        params.update(bagging_fraction=0.8, bagging_freq=1)
    models = []
    for library in (lightgbm, lgb):
        booster = library.train(
            params, library.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True
        )

        def objective(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
            hessian = np.linspace(0.5, 1.5, len(predictions))
            if initial_objective == "multiclass":
                return predictions - np.eye(3)[dataset.get_label().astype(int)], np.repeat(
                    hessian[:, None], 3, axis=1
                )
            return predictions - dataset.get_label(), hessian

        if variant == "rf":
            with pytest.raises(library.basic.LightGBMError, match="RF mode do not support custom"):
                booster.update(fobj=objective)
            continue
        booster.update(fobj=objective)
        before = booster.predict(x, raw_score=True)
        iteration = booster.current_iteration()
        with pytest.raises(
            library.basic.LightGBMError, match="Cannot update due to null objective function"
        ):
            booster.update()
        assert booster.current_iteration() == iteration
        np.testing.assert_array_equal(booster.predict(x, raw_score=True), before)
        booster.update(fobj=objective)
        models.append(booster)
    if variant != "rf":
        np.testing.assert_allclose(
            models[1].predict(x), models[0].predict(x), rtol=1e-12, atol=1e-15
        )


@pytest.mark.parametrize("alias", ["objective", "application", "app", "objective_type", "loss"])
@pytest.mark.parametrize("entry", ["train", "cv"])
def test_custom_objective_aliases(alias: str, entry: str) -> None:
    x = np.random.default_rng(31).normal(size=(80, 3))
    y = x[:, 0]

    def objective(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        return predictions - dataset.get_label(), np.ones_like(predictions)

    def evaluate(predictions: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
        return "raw_mean", float(np.mean(predictions)), False

    params = {
        alias: objective,
        "metric": "None",
        "verbosity": -1,
        "num_threads": 1,
        "min_data_in_leaf": 2,
    }
    results = []
    for library in (lightgbm, lgb):
        dataset = library.Dataset(x, label=y)
        if entry == "train":
            values: dict[str, Any] = {}
            model = library.train(
                params,
                dataset,
                num_boost_round=3,
                valid_sets=[dataset],
                feval=evaluate,
                callbacks=[library.record_evaluation(values)],
            )
            np.testing.assert_allclose(
                next(iter(values.values()))["raw_mean"][-1],
                np.mean(model.predict(x, raw_score=True)),
                rtol=1e-12,
            )
            results.append(model.predict(x))
        else:
            result = library.cv(
                params, dataset, num_boost_round=3, nfold=2, stratified=False, feval=evaluate
            )
            results.append(result["valid raw_mean-mean"])
    np.testing.assert_allclose(results[1], results[0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("categorical", [False, True])
def test_reference_dataset_training_matches_upstream(categorical: bool) -> None:
    rng = np.random.default_rng(911)
    source = rng.normal(size=(120, 3))
    x = rng.normal(loc=1.5, size=(70, 3))
    y = x[:, 0] - x[:, 1]
    if categorical:
        source[:, 1] = np.arange(len(source)) % 4
        x[:, 1] = np.arange(len(x)) % 5
    params = {
        "verbosity": -1,
        "num_threads": 1,
        "max_bin": 7,
        "min_data_in_bin": 1,
        "min_data_in_leaf": 2,
        "num_leaves": 5,
        "objective": "regression",
    }
    predictions = []
    for library in (lightgbm, lgb):
        reference = library.Dataset(
            source,
            label=source[:, 0],
            params=params,
            categorical_feature=[1] if categorical else [],
        ).construct()
        dataset = library.Dataset(
            x, label=y, weight=np.linspace(0.5, 1.5, len(y)), reference=reference, params=params
        ).construct()
        predictions.append(library.train(params, dataset, num_boost_round=3).predict(x))
    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("rows", [[8, 0, 5, 2, 6], [2, 0, 2, 8]])
def test_subset_metadata_matches_upstream(rows: list[int]) -> None:
    x = np.random.default_rng(71).normal(size=(12, 3))
    y = np.arange(12, dtype=float)
    weights = np.linspace(0.5, 2.0, 12)
    subsets = []
    params = {"verbosity": -1, "min_data_in_leaf": 1, "min_data_in_bin": 1}
    for library in (lightgbm, lgb):
        source = library.Dataset(x, label=y, weight=weights, group=[4, 3, 5], params=params)
        subset = source.subset(rows).construct()
        subsets.append(subset)
    np.testing.assert_array_equal(subsets[1].label, subsets[0].label)
    np.testing.assert_array_equal(
        np.asarray(subsets[1].weight, dtype=np.float32), subsets[0].get_weight()
    )
    np.testing.assert_array_equal(subsets[1].group, subsets[0].get_group())
    predictions = []
    for library, subset in zip((lightgbm, lgb), subsets, strict=True):
        booster = library.train(
            {**params, "objective": "lambdarank", "num_threads": 1}, subset, num_boost_round=3
        )
        predictions.append(booster.predict(x))
    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("rows", [[-1], [12], [], [0, 12]])
def test_invalid_subset_indices_match_upstream(rows: list[int]) -> None:
    x = np.arange(36, dtype=float).reshape(12, 3)
    messages = []
    for library in (lightgbm, lgb):
        dataset = library.Dataset(
            x, label=np.arange(12), params={"verbosity": -1, "min_data_in_leaf": 1}
        )
        with pytest.raises(library.basic.LightGBMError) as caught:
            dataset.subset(rows).construct()
        message = str(caught.value)
        # Upstream embeds its build-machine path and line number in CHECK errors.
        messages.append(message.split(" at ")[0])
        assert dataset.num_data() == 12
        assert dataset.subset([0, 1]).construct().num_data() == 2
    assert messages[0] == messages[1]


@pytest.mark.parametrize("field,values", [("weight", [1.0] * 11), ("group", [4, 7])])
def test_invalid_dataset_metadata_matches_upstream(field: str, values: list[float]) -> None:
    x = np.arange(36, dtype=float).reshape(12, 3)
    errors = []
    for library in (lightgbm, lgb):
        with pytest.raises(library.basic.LightGBMError) as caught:
            library.Dataset(
                x,
                label=np.arange(12),
                **{field: values},
                params={"verbosity": -1, "min_data_in_leaf": 1},
            ).construct()
        errors.append(str(caught.value))
    assert errors[0] == errors[1]


def test_nested_subset_with_empty_query_matches_upstream() -> None:
    x = np.random.default_rng(21).normal(size=(12, 3))
    subsets = []
    for library in (lightgbm, lgb):
        source = library.Dataset(
            x,
            label=np.arange(12),
            weight=np.arange(12) + 1,
            group=[4, 0, 8],
            params={"verbosity": -1, "min_data_in_leaf": 1},
        )
        subset = source.subset([10, 1, 4, 3, 6]).construct().subset([4, 0, 2]).construct()
        subsets.append(subset)
    np.testing.assert_array_equal(subsets[1].label, subsets[0].label)
    np.testing.assert_array_equal(
        np.asarray(subsets[1].weight, dtype=np.float32), subsets[0].get_weight()
    )
    np.testing.assert_array_equal(subsets[1].group, subsets[0].get_group())


def test_negative_query_sizes_match_upstream_for_regression() -> None:
    # upstream: src/io/metadata.cpp (v4.7.0) SetQueriesFromIterator only checks that the
    # query sizes sum to #data; regression ignores the query boundaries.
    x = np.random.default_rng(5).normal(size=(12, 3))
    y = np.arange(12, dtype=float)
    params = {"verbosity": -1, "min_data_in_leaf": 1, "num_threads": 1}
    predictions = []
    for library in (lightgbm, lgb):
        dataset = library.Dataset(x, label=y, group=[-1, 13], params=params).construct()
        booster = library.train({**params, "objective": "regression"}, dataset, num_boost_round=3)
        predictions.append(booster.predict(x))
    np.testing.assert_array_equal(
        lightgbm.Dataset(x, label=y, group=[-1, 13]).construct().get_group(), [-1, 13]
    )
    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("group", [[-1, 13], [13, -1], [-1, 1, 12]])
def test_negative_query_sizes_reject_ranking_with_clear_error(group: list[int]) -> None:
    # Upstream aborts the process (std::length_error) here, so there is no upstream message
    # to match; the Python layer must raise LightGBMError rather than crash or train silently.
    x = np.random.default_rng(6).normal(size=(12, 3))
    labels = np.random.default_rng(7).integers(0, 3, size=12).astype(float)
    params = {"verbosity": -1, "min_data_in_leaf": 1, "num_threads": 1}
    dataset = lgb.Dataset(x, label=labels, group=group, params=params)
    with pytest.raises(lgb.basic.LightGBMError, match="negative"):
        lgb.train({**params, "objective": "lambdarank"}, dataset, num_boost_round=1)


def _init_score_params() -> dict[str, Any]:
    return {"verbosity": -1, "num_threads": 1, "min_data_in_leaf": 2, "num_leaves": 5}


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_init_score_training_matches_upstream(objective: str) -> None:
    # upstream: src/boosting/gbdt.cpp (v4.7.0) BoostFromAverage skips the automatic initial
    # score when the training score updater has an init score; the trees start from it.
    rng = np.random.default_rng(31)
    x = rng.normal(size=(60, 3))
    y = (x[:, 0] > 0).astype(float) if objective == "binary" else x[:, 0] - x[:, 1]
    init = 0.3 * x[:, 1] + 0.5
    params = {**_init_score_params(), "objective": objective}
    results = []
    for library in (lightgbm, lgb):
        dataset = library.Dataset(x, label=y, init_score=init, params=params).construct()
        booster = library.train(params, dataset, num_boost_round=4)
        results.append((booster.model_to_string(), booster.predict(x, raw_score=True)))
    assert results[1][0] == results[0][0]
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12, atol=1e-15)


def test_multiclass_init_score_matches_upstream() -> None:
    # Two-dimensional init scores are class-major after upstream's Fortran-order ravel.
    rng = np.random.default_rng(32)
    x = rng.normal(size=(90, 3))
    y = np.argmax(x, axis=1).astype(float)
    init = rng.normal(scale=0.5, size=(90, 3))
    params = {**_init_score_params(), "objective": "multiclass", "num_class": 3}
    results = []
    for library in (lightgbm, lgb):
        dataset = library.Dataset(x, label=y, init_score=init, params=params).construct()
        booster = library.train(params, dataset, num_boost_round=3)
        results.append(booster.predict(x, raw_score=True))
    np.testing.assert_allclose(results[1], results[0], rtol=1e-12, atol=1e-15)


def test_validation_init_score_changes_metrics_like_upstream() -> None:
    rng = np.random.default_rng(33)
    x = rng.normal(size=(80, 3))
    y = x[:, 0] + rng.normal(scale=0.1, size=80)
    x_valid = rng.normal(size=(30, 3))
    y_valid = x_valid[:, 0]
    valid_init = 0.2 * x_valid[:, 2]
    params = {**_init_score_params(), "objective": "regression", "metric": "l2"}
    curves = []
    for library in (lightgbm, lgb):
        train = library.Dataset(x, label=y, params=params)
        valid = library.Dataset(
            x_valid, label=y_valid, init_score=valid_init, reference=train, params=params
        )
        record: dict[str, Any] = {}
        library.train(
            params,
            train,
            num_boost_round=3,
            valid_sets=[valid],
            valid_names=["valid"],
            callbacks=[library.record_evaluation(record)],
        )
        curves.append(record["valid"]["l2"])
    np.testing.assert_allclose(curves[1], curves[0], rtol=1e-12, atol=1e-15)


def test_subset_with_init_score_trains_like_upstream() -> None:
    rng = np.random.default_rng(34)
    x = rng.normal(size=(40, 3))
    y = x[:, 0] - 0.5 * x[:, 1]
    init = np.linspace(-1.0, 1.0, 40)
    rows = [30, 2, 17, 2, 9, 39, 0, 11]
    params = {**_init_score_params(), "objective": "regression"}
    predictions = []
    for library in (lightgbm, lgb):
        source = library.Dataset(x, label=y, init_score=init, params=params).construct()
        subset = source.subset(rows).construct()
        booster = library.train(params, subset, num_boost_round=3)
        predictions.append(booster.predict(x[rows], raw_score=True))
    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-12, atol=1e-15)


def test_init_score_size_error_matches_upstream() -> None:
    x = np.arange(36, dtype=float).reshape(12, 3)
    messages = []
    for library in (lightgbm, lgb):
        with pytest.raises(library.basic.LightGBMError) as caught:
            library.Dataset(
                x, label=np.arange(12), init_score=np.ones(11), params={"verbosity": -1}
            ).construct()
        messages.append(str(caught.value).split(" at ")[0])
    assert messages[0] == messages[1]


def test_random_forest_rejects_init_score_like_upstream() -> None:
    # upstream: src/boosting/rf.hpp (v4.7.0) RF::Init: CHECK_EQ(init_score, nullptr).
    rng = np.random.default_rng(35)
    x = rng.normal(size=(60, 3))
    y = x[:, 0]
    params = {
        "verbosity": -1,
        "objective": "regression",
        "boosting": "rf",
        "bagging_freq": 1,
        "bagging_fraction": 0.5,
        "num_threads": 1,
    }
    dataset = lgb.Dataset(x, label=y, init_score=np.ones(60), params=params)
    with pytest.raises(lgb.LightGBMError, match="init_score"):
        lgb.train(params, dataset, num_boost_round=1)


@pytest.mark.parametrize("source", ["booster", "file"])
def test_init_model_continuation_matches_upstream(source: str, tmp_path: Path) -> None:
    # upstream: GBDT::MergeFrom keeps the initial model's trees first, and the new trees are
    # trained on its raw predictions (passed as init_score). The parameters block records the
    # rounds of the continued call only.
    rng = np.random.default_rng(41)
    x = rng.normal(size=(60, 3))
    y = x[:, 0] - x[:, 1]
    params = {**_init_score_params(), "objective": "regression"}
    results = []
    for library in (lightgbm, lgb):
        base = library.train(params, library.Dataset(x, label=y, params=params), num_boost_round=3)
        if source == "file":
            path = tmp_path / f"{library.__name__}.txt"
            base.save_model(str(path))
            init_model: Any = str(path)
        else:
            init_model = base
        continued = library.train(
            params,
            library.Dataset(x, label=y, params=params),
            num_boost_round=2,
            init_model=init_model,
        )
        results.append((continued.model_to_string(), continued.predict(x)))
    assert results[1][0] == results[0][0]
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12, atol=1e-15)


def test_multiclass_init_model_continuation_matches_upstream() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(90, 3))
    y = np.argmax(x, axis=1).astype(float)
    params = {**_init_score_params(), "objective": "multiclass", "num_class": 3}
    results = []
    for library in (lightgbm, lgb):
        base = library.train(params, library.Dataset(x, label=y, params=params), num_boost_round=2)
        continued = library.train(
            params, library.Dataset(x, label=y, params=params), num_boost_round=2, init_model=base
        )
        results.append(continued.predict(x, raw_score=True))
    np.testing.assert_allclose(results[1], results[0], rtol=1e-12, atol=1e-15)


def test_init_model_with_validation_callbacks_matches_upstream() -> None:
    rng = np.random.default_rng(43)
    x = rng.normal(size=(80, 3))
    y = x[:, 0] + rng.normal(scale=0.1, size=80)
    x_valid = rng.normal(size=(30, 3))
    y_valid = x_valid[:, 0]
    params = {**_init_score_params(), "objective": "regression", "metric": "l2"}
    curves = []
    for library in (lightgbm, lgb):
        base = library.train(params, library.Dataset(x, label=y, params=params), num_boost_round=2)
        train = library.Dataset(x, label=y, params=params)
        valid = library.Dataset(x_valid, label=y_valid, reference=train, params=params)
        record: dict[str, Any] = {}
        library.train(
            params,
            train,
            num_boost_round=3,
            init_model=base,
            valid_sets=[valid],
            valid_names=["valid"],
            callbacks=[library.record_evaluation(record)],
        )
        curves.append(record["valid"]["l2"])
    np.testing.assert_allclose(curves[1], curves[0], rtol=1e-12, atol=1e-15)
