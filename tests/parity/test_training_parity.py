"""Training through the public API must produce upstream's model text byte for byte.

Every training fixture under ``tests/fixtures/train`` is trained with ``lgb.train`` here
and with ``lightgbm.train`` (same data, params, weights, groups and categoricals).
"""

from __future__ import annotations

import json
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


def test_train_without_keep_training_booster_cannot_continue_like_upstream() -> None:
    rng = np.random.default_rng(482)
    x = rng.normal(size=(80, 3))
    for library in (lgb, lightgbm):
        booster = library.train({"verbosity": -1}, library.Dataset(x, label=x[:, 0]), 1)
        # upstream fails with AttributeError (free_dataset drops train_set); bankai raises
        # LightGBMError("Cannot continue training a Booster loaded from a model ...").
        with pytest.raises((AttributeError, library.basic.LightGBMError)):
            booster.update()


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

    # upstream train() returns a reparsed model without its Dataset unless
    # keep_training_booster=True (engine.py v4.7.0), so only then can training continue.
    trained_once = lgb.train(
        dict(params),
        lgb.Dataset(x, label=y, params=dict(params)),
        num_boost_round=1,
        keep_training_booster=True,
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


def test_record_and_log_evaluation_callbacks(capsys: Any) -> None:
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
    ours = lgb.train(
        dict(params),
        lgb.Dataset(x_train, label=y_train, params=dict(params)),
        num_boost_round=3,
        valid_sets=[lgb.Dataset(x_valid, label=y_valid, params=dict(params))],
        valid_names=["valid"],
        callbacks=[lgb.record_evaluation(recorded), lgb.log_evaluation(period=1)],
    )
    ours_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
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
    # upstream: log_evaluation prints each line through the same logger as the native library.
    expected_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
    assert ours_lines[0].startswith("[1]\tvalid's l2: ")
    assert ours_lines == expected_lines
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
    # upstream: engine.py (v4.7.0) train() checks the resolved num_iterations.
    with pytest.raises(ValueError) as want:
        lightgbm.train({"verbose": -1}, lightgbm.Dataset(x, label=x[:, 0]), num_boost_round=0)
    with pytest.raises(ValueError) as got:
        lgb.train({"verbose": -1}, ds, num_boost_round=0)
    assert str(got.value) == str(want.value)


# Upstream's DART never initializes ``is_update_score_cur_iter_`` (src/boosting/dart.hpp
# (v4.7.0)), and with a custom objective the first ``GetTrainingScore`` reads it to decide
# whether to draw from the drop RNG, so upstream's own trees vary between processes for
# default DART settings. With ``skip_drop=0`` and uniform drops at ``drop_rate=1`` every draw
# passes its threshold, so the outcome no longer depends on that draw, and trees are
# dropped on every iteration (which is what these tests need to exercise).
_DART_RNG_INDEPENDENT: dict[str, Any] = {"skip_drop": 0.0, "uniform_drop": True, "drop_rate": 1.0}


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
    if boosting == "dart":
        params.update(_DART_RNG_INDEPENDENT)
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


@pytest.mark.parametrize("strategy", ["bagging", "goss"])
def test_rollback_one_iter_matches_upstream(strategy: str) -> None:
    # upstream: GBDT::RollbackOneIter subtracts the last iteration's shrunken trees from the
    # training scores, drops them, and training continues from the rolled back scores.
    rng = np.random.default_rng(51)
    x = rng.normal(size=(120, 3))
    y = x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.2, size=120)
    params = {
        **_init_score_params(),
        "objective": "regression",
        "data_sample_strategy": strategy,
        "bagging_freq": 1 if strategy == "bagging" else 0,
        "bagging_fraction": 0.7,
        "bagging_seed": 3,
        "seed": 7,
    }
    results = []
    for library in (lightgbm, lgb):
        booster = library.train(
            params,
            library.Dataset(x, label=y, params=params),
            num_boost_round=4,
            keep_training_booster=True,
        )
        booster.rollback_one_iter()
        after_rollback = booster.predict(x, raw_score=True)
        booster.update()
        results.append(
            (booster.current_iteration(), after_rollback, booster.predict(x, raw_score=True))
        )
    assert results[1][0] == results[0][0] == 4
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(results[1][2], results[0][2], rtol=1e-12, atol=1e-15)


def test_set_leaf_output_matches_upstream() -> None:
    # upstream: GBDT::SetLeafValue changes the model's leaf output; later iterations keep
    # training from the trained scores, not from the edited leaf.
    rng = np.random.default_rng(52)
    x = rng.normal(size=(100, 3))
    y = x[:, 0] - 0.5 * x[:, 1]
    params = {**_init_score_params(), "objective": "regression"}
    results = []
    for library in (lightgbm, lgb):
        booster = library.train(
            params,
            library.Dataset(x, label=y, params=params),
            num_boost_round=3,
            keep_training_booster=True,
        )
        booster.set_leaf_output(1, 2, 0.75)
        edited = booster.predict(x, raw_score=True)
        booster.update()
        results.append((edited, booster.predict(x, raw_score=True), booster.get_leaf_output(1, 2)))
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12, atol=1e-15)
    assert results[1][2] == results[0][2] == 0.75


def test_shuffle_models_matches_upstream() -> None:
    # upstream: GBDT::ShuffleModels draws the iteration order with Random(17) over
    # [start, end); trees move in blocks of num_tree_per_iteration.
    rng = np.random.default_rng(53)
    x = rng.normal(size=(100, 3))
    y = x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.2, size=100)
    params = {**_init_score_params(), "objective": "regression"}
    results = []
    for library in (lightgbm, lgb):
        booster = library.train(
            params,
            library.Dataset(x, label=y, params=params),
            num_boost_round=5,
            keep_training_booster=True,
        )
        booster.shuffle_models(1, 4)
        shuffled = (booster.model_to_string(), booster.predict(x, raw_score=True))
        booster.update()
        results.append((shuffled, booster.predict(x, raw_score=True)))
    assert results[1][0][0] == results[0][0][0]
    np.testing.assert_allclose(results[1][0][1], results[0][0][1], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12, atol=1e-15)


def test_dataset_getters_match_upstream() -> None:
    x = np.random.default_rng(54).normal(size=(12, 3))
    y = np.arange(12, dtype=float)
    weight = np.linspace(0.5, 2.0, 12)
    group = [4, 3, 5]
    init = np.arange(36, dtype=float).reshape(12, 3)
    params = {"verbosity": -1, "min_data_in_leaf": 1}
    values = []
    for library in (lightgbm, lgb):
        dataset = library.Dataset(
            x, label=y, weight=weight, group=group, init_score=init, params=params
        ).construct()
        values.append(
            (
                np.asarray(dataset.get_weight(), dtype=np.float64),
                np.asarray(dataset.get_group()),
                np.asarray(dataset.get_init_score(), dtype=np.float64),
            )
        )
    for ours, theirs in zip(values[1], values[0], strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_dataset_setters_after_construct_match_upstream() -> None:
    # upstream: set_label/set_weight/set_group update an already constructed Dataset.
    x = np.random.default_rng(55).normal(size=(12, 3))
    params = {"verbosity": -1, "min_data_in_leaf": 1, "num_threads": 1}
    new_label = (np.arange(12) % 3).astype(float)
    new_weight = np.linspace(0.5, 2.0, 12)
    predictions = []
    for library in (lightgbm, lgb):
        dataset = library.Dataset(x, label=np.arange(12), params=params).construct()
        dataset.set_label(new_label)
        dataset.set_weight(new_weight)
        dataset.set_group([6, 6])
        np.testing.assert_array_equal(dataset.get_label(), new_label.astype(np.float32))
        booster = library.train({**params, "objective": "lambdarank"}, dataset, num_boost_round=2)
        predictions.append(booster.predict(x, raw_score=True))
    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("seed", [9, 10, 11])
def test_extra_trees_matches_upstream(seed: int) -> None:
    # upstream: FeatureHistogram random thresholds (rand.NextInt per feature), seeded by `seed`.
    # Coverage for an already-ported path: this passes on the first run.
    rng = np.random.default_rng(61)
    x = rng.normal(size=(300, 4))
    y = x[:, 0] - x[:, 1] + rng.normal(scale=0.2, size=300)
    params = {
        **_init_score_params(),
        "objective": "regression",
        "extra_trees": True,
        "seed": seed,
        "min_data_in_leaf": 5,
        "num_leaves": 15,
    }
    results = []
    for library in (lightgbm, lgb):
        booster = library.train(
            params, library.Dataset(x, label=y, params=params), num_boost_round=5
        )
        results.append((booster.model_to_string(), booster.predict(x)))
    assert results[1][0] == results[0][0]
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=1e-12, atol=1e-15)


def test_bagging_by_query_rejects_rank_xendcg_until_ported() -> None:
    # Upstream samples whole queries for bagging_by_query=true, and rank_xendcg draws a
    # random number per query, so only the sampled queries get gradients upstream. bankai
    # does not mirror that yet, so it must refuse rather than silently diverge. Lambdarank
    # is supported: see test_bagging_by_query_parity.py.
    rng = np.random.default_rng(62)
    x = rng.normal(size=(12, 3))
    params = {
        "verbosity": -1,
        "objective": "rank_xendcg",
        "bagging_by_query": True,
        "bagging_freq": 1,
        "bagging_fraction": 0.5,
        "min_data_in_leaf": 1,
    }
    dataset = lgb.Dataset(x, label=np.arange(12) % 3, group=[4, 4, 4], params=params)
    with pytest.raises(lgb.LightGBMError, match="bagging_by_query"):
        lgb.train(params, dataset, num_boost_round=1)


def _num_iterations_line(model: str) -> str:
    return next(line for line in model.splitlines() if line.startswith("[num_iterations:"))


def test_model_records_num_boost_round_with_validation_sets() -> None:
    """upstream: engine.py (v4.7.0) _choose_num_iterations puts it in params on every path."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(400, 4))
    y = x[:, 0] + rng.normal(size=400) * 0.1
    lines = []
    for library in (lgb, lightgbm):
        train = library.Dataset(x[:300], y[:300])
        valid = library.Dataset(x[300:], y[300:], reference=train)
        booster = library.train(
            {"verbose": -1, "num_threads": 1, "deterministic": True},
            train,
            num_boost_round=7,
            valid_sets=[valid],
        )
        lines.append(_num_iterations_line(booster.model_to_string()))
    assert lines[0] == lines[1] == "[num_iterations: 7]"


@pytest.mark.parametrize(
    "extra",
    [{"n_estimators": 5}, {"num_iterations": 4, "num_trees": 6}, {"num_round": 3}],
)
def test_num_iterations_in_params_overrides_num_boost_round(extra: dict[str, Any]) -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(300, 3))
    y = x[:, 0]
    results = []
    for library in (lgb, lightgbm):
        params = {"verbose": -1, "num_threads": 1, **extra}
        booster = library.train(params, library.Dataset(x, y), num_boost_round=9)
        results.append((booster.num_trees(), _num_iterations_line(booster.model_to_string())))
    assert results[0] == results[1]


def test_conflicting_num_iterations_aliases_warn_like_upstream() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(200, 3))
    params = {"verbose": -1, "n_estimators": 3, "num_rounds": 4}
    messages = []
    for library in (lgb, lightgbm):
        with pytest.warns(UserWarning, match="Found conflicting values for num_iterations") as rec:
            library.train(dict(params), library.Dataset(x, x[:, 0]), num_boost_round=2)
        messages.append(next(str(w.message) for w in rec if "conflicting" in str(w.message)))
    # upstream lists the aliases from ``_ConfigAliases.get`` (a set), so their order depends
    # on Python's per-process string hashing; compare the pairs as a set, the rest exactly.
    prefix = "Found conflicting values for num_iterations provided via 'params': "

    def split(message: str) -> tuple[set[str], str]:
        pairs, rest = message[len(prefix) :].split(". ", 1)
        return set(pairs.split(", ")), rest

    assert all(m.startswith(prefix) for m in messages)
    assert split(messages[0]) == split(messages[1])


@pytest.mark.parametrize("alias", ["linear_tree", "linear_trees"])
def test_linear_tree_training_matches_upstream(alias: str) -> None:
    # Upstream fits a linear model in every leaf. bankai trains the same linear leaves
    # (ROADMAP P3 `linear_tree`); the full multi-round comparison lives in
    # tests/parity/test_linear_tree_parity.py, this checks the alias path end to end.
    rng = np.random.default_rng(64)
    x = rng.normal(size=(200, 3))
    params = {"verbosity": -1, "num_threads": 1, "deterministic": True, "seed": 1, alias: True}
    ours = lgb.train(params, lgb.Dataset(x, label=x[:, 0]), num_boost_round=2)
    theirs = lightgbm.train(params, lightgbm.Dataset(x, label=x[:, 0]), num_boost_round=2)
    assert "is_linear=1" in ours.model_to_string()
    np.testing.assert_allclose(ours.predict(x), theirs.predict(x), rtol=1e-7, atol=1e-12)


def _early_stopping_outcome(
    library: Any, params: dict[str, Any], rounds: int, callback_kw: dict[str, Any], **train_kw: Any
) -> tuple[int, int, dict[str, Any]]:
    rng = np.random.default_rng(65)
    x = rng.normal(size=(600, 4))
    y = x[:, 0] + 0.5 * np.sin(3 * x[:, 1]) + rng.normal(size=600) * 0.3
    train = library.Dataset(x[:400], y[:400])
    valid = library.Dataset(x[400:], y[400:], reference=train)
    valid_sets = train_kw.pop("valid_sets", None)
    booster = library.train(
        {"verbose": -1, "num_threads": 1, "deterministic": True, "seed": 3, **params},
        train,
        num_boost_round=rounds,
        valid_sets=[valid]
        if valid_sets is None
        else [train if v == "train" else valid for v in valid_sets],
        callbacks=[library.early_stopping(**callback_kw)],
        **train_kw,
    )
    best = {k: dict(v) for k, v in booster.best_score.items()}
    return booster.best_iteration, booster.num_trees(), best


@pytest.mark.parametrize(
    ("params", "rounds", "callback_kw"),
    [
        # never stops: upstream still reports the best iteration on the final round
        ({"metric": "l2"}, 15, {"stopping_rounds": 50, "verbose": False}),
        # several metrics: the first one that goes stale decides
        (
            {"metric": ["l2", "l1"], "learning_rate": 0.5},
            60,
            {"stopping_rounds": 2, "verbose": False},
        ),
        (
            {"metric": ["l1", "l2"], "learning_rate": 0.5},
            60,
            {"stopping_rounds": 2, "verbose": False},
        ),
        (
            {"metric": ["l2", "l1"], "learning_rate": 0.5},
            60,
            {"stopping_rounds": 2, "verbose": False, "first_metric_only": True},
        ),
        (
            {"metric": ["l2", "l1"], "learning_rate": 0.5},
            60,
            {"stopping_rounds": 2, "verbose": False, "min_delta": [0.01, 0.0]},
        ),
        # disabled with stopping_rounds <= 0
        ({"metric": "l2"}, 8, {"stopping_rounds": 0, "verbose": False}),
    ],
)
def test_early_stopping_scenarios_match_upstream(
    params: dict[str, Any], rounds: int, callback_kw: dict[str, Any]
) -> None:
    """upstream: callback.py (v4.7.0) _EarlyStoppingCallback."""
    ours = _early_stopping_outcome(lgb, params, rounds, callback_kw)
    theirs = _early_stopping_outcome(lightgbm, params, rounds, callback_kw)
    assert ours == theirs


def test_early_stopping_is_disabled_for_dart_like_upstream() -> None:
    for library in (lgb, lightgbm):
        with pytest.warns(UserWarning, match="Early stopping is not available in dart mode"):
            best_iteration, num_trees, _ = _early_stopping_outcome(
                library,
                {"metric": "l2", "boosting": "dart"},
                6,
                {"stopping_rounds": 1, "verbose": False},
            )
        assert (best_iteration, num_trees) == (0, 6)


def test_early_stopping_with_only_the_training_set_is_disabled_like_upstream() -> None:
    outcomes = []
    for library in (lgb, lightgbm):
        with pytest.warns(UserWarning, match="Only training set found, disabling early stopping."):
            outcomes.append(
                _early_stopping_outcome(
                    library,
                    {"metric": "l2"},
                    6,
                    {"stopping_rounds": 1, "verbose": False},
                    valid_sets=["train"],
                )
            )
    assert outcomes[0] == outcomes[1]


def test_early_stopping_rounds_type_error_like_upstream() -> None:
    errors = []
    for library in (lgb, lightgbm):
        with pytest.raises(TypeError) as err:
            _early_stopping_outcome(library, {"metric": "l2"}, 3, {"stopping_rounds": 1.5})
        errors.append(str(err.value))
    assert errors[0] == errors[1]


def _capture_fobj_inputs(
    library: Any, params: dict[str, Any], x: np.ndarray, y: np.ndarray, rounds: int, **ds_kw: Any
) -> tuple[list[np.ndarray], str]:
    seen: list[np.ndarray] = []

    def objective(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        seen.append(np.array(predictions, copy=True))
        return predictions - dataset.get_label(), np.ones_like(predictions)

    booster = library.train(
        {**params, "objective": objective}, library.Dataset(x, label=y, **ds_kw), rounds
    )
    return seen, booster.model_to_string()


# max_drop >= rounds keeps uniform drop_rate = min(1, max_drop / iter) at 1, so the drop RNG
# never decides (see _DART_RNG_INDEPENDENT); with max_drop=1 upstream itself returns two
# different models across processes (13/16 vs 3/16 runs; bankai reproduces the majority).
@pytest.mark.parametrize("max_drop", [5, 50])
@pytest.mark.parametrize("xgboost_dart_mode", [False, True])
def test_dart_custom_objective_sees_dropped_training_score(
    max_drop: int, xgboost_dart_mode: bool
) -> None:
    """upstream: Booster.update(fobj) passes LGBM_BoosterGetPredict(0), i.e.
    DART::GetTrainingScore after this iteration's drop (src/boosting/dart.hpp v4.7.0)."""
    rng = np.random.default_rng(66)
    x = rng.normal(size=(120, 3))
    y = x[:, 0] - 0.4 * x[:, 1]
    params = {
        "boosting": "dart",
        "verbosity": -1,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "num_leaves": 5,
        "min_data_in_leaf": 2,
        "metric": "None",
        "max_drop": max_drop,
        "xgboost_dart_mode": xgboost_dart_mode,
        **_DART_RNG_INDEPENDENT,
    }
    ours, ours_model = _capture_fobj_inputs(lgb, params, x, y, 5)
    theirs, theirs_model = _capture_fobj_inputs(lightgbm, params, x, y, 5)
    assert len(ours) == len(theirs) == 5
    for i, (got, want) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15, err_msg=f"iteration {i}")
    assert ours_model == theirs_model


def test_custom_objective_sees_init_score_like_upstream() -> None:
    """The training score handed to ``fobj`` starts from the Dataset's init_score."""
    rng = np.random.default_rng(67)
    x = rng.normal(size=(100, 3))
    y = x[:, 0]
    init = rng.normal(size=100)
    params = {"verbosity": -1, "num_threads": 1, "deterministic": True, "metric": "None"}
    ours, ours_model = _capture_fobj_inputs(lgb, params, x, y, 3, init_score=init)
    theirs, theirs_model = _capture_fobj_inputs(lightgbm, params, x, y, 3, init_score=init)
    for got, want in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
    assert ours_model == theirs_model


def test_multiclass_custom_objective_input_shape_like_upstream() -> None:
    rng = np.random.default_rng(68)
    x = rng.normal(size=(90, 4))
    y = np.argmax(x[:, :3], axis=1).astype(float)
    params = {
        "verbosity": -1,
        "num_threads": 1,
        "deterministic": True,
        "metric": "None",
        "num_class": 3,
    }

    def run(library: Any) -> list[np.ndarray]:
        seen: list[np.ndarray] = []

        def objective(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
            seen.append(np.array(predictions, copy=True))
            onehot = np.eye(3)[dataset.get_label().astype(int)]
            p = np.exp(predictions - predictions.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            return p - onehot, p * (1 - p)

        library.train({**params, "objective": objective}, library.Dataset(x, label=y), 3)
        return seen

    ours, theirs = run(lgb), run(lightgbm)
    for got, want in zip(ours, theirs, strict=True):
        assert got.shape == want.shape == (90, 3)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


def test_nested_interaction_constraints_keep_their_groups() -> None:
    """upstream: basic.py (v4.7.0) _to_string writes inner lists as ``[a,b]``; flattening
    them merged every group into one, which silently removed the constraint."""
    rng = np.random.default_rng(69)
    x = rng.normal(size=(600, 3))
    y = x[:, 0] * x[:, 1] + 0.5 * x[:, 2]
    params = {
        "verbosity": -1,
        "num_threads": 1,
        "deterministic": True,
        "interaction_constraints": [[0], [1, 2]],
        "num_leaves": 8,
    }
    models = [
        lib.train(params, lib.Dataset(x, label=y), num_boost_round=5).model_to_string()
        for lib in (lgb, lightgbm)
    ]
    assert models[0] == models[1]
    assert "[interaction_constraints: [0],[1,2]]" in models[0]


@pytest.mark.parametrize("valid", [False, True])
def test_train_returns_the_reparsed_model_like_upstream(valid: bool) -> None:
    """upstream: engine.py (v4.7.0) train() ends with model_from_string(model_to_string())
    unless keep_training_booster; a one-leaf tree then prints an empty leaf_weight, as
    Tree(const char*) returns before parsing it."""
    rng = np.random.default_rng(71)
    x = rng.normal(size=(300, 3))
    y = x[:, 0]
    params = {"verbosity": -1, "num_threads": 1, "min_data_in_leaf": 250, "metric": "l2"}
    texts = []
    for lib in (lgb, lightgbm):
        train = lib.Dataset(x, label=y)
        kw = {"valid_sets": [lib.Dataset(x, label=y, reference=train)]} if valid else {}
        texts.append(lib.train(params, train, num_boost_round=2, **kw).model_to_string())
    assert texts[0] == texts[1]
    assert "num_leaves=1\n" in texts[0] and "leaf_weight=\n" in texts[0]
    for lib in (lgb, lightgbm):
        kept = lib.train(params, lib.Dataset(x, label=y), 2, keep_training_booster=True)
        assert "leaf_weight=0\n" in kept.model_to_string()


@pytest.mark.parametrize(
    "bad",
    [
        {"label_nan": True},
        {"label_inf": True},
        {"weight_nan": True},
        {"weight_huge": True},
    ],
    ids=["label-nan", "label-inf", "weight-nan", "weight-huge"],
)
def test_non_finite_labels_and_weights_are_sanitized_like_upstream(bad: dict[str, bool]) -> None:
    """upstream: src/io/metadata.cpp (v4.7.0) stores labels and weights through
    Common::AvoidInf (NaN -> 0, clamped to +-1e38 as float)."""
    rng = np.random.default_rng(70)
    x = rng.normal(size=(500, 4))
    y = x[:, 0] + rng.normal(size=500) * 0.1
    w = rng.uniform(0.5, 2.0, size=500)
    if bad.get("label_nan"):
        y[::17] = np.nan
    if bad.get("label_inf"):
        y[5], y[9] = np.inf, -np.inf
    if bad.get("weight_nan"):
        w[::13] = np.nan
    if bad.get("weight_huge"):
        w[3] = 1e300
    params = {"verbosity": -1, "num_threads": 1, "deterministic": True, "num_leaves": 7}
    outputs = []
    for lib in (lgb, lightgbm):
        ds = lib.Dataset(x, label=y, weight=w if "weight" in next(iter(bad)) else None)
        booster = lib.train(params, ds, num_boost_round=3)
        outputs.append((booster.model_to_string(), ds.get_label()))
    assert outputs[0][0] == outputs[1][0]
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
