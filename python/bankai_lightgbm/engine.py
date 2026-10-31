"""Training entry point. Mirrors ``lightgbm.engine``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bankai_lightgbm.basic import Booster, Dataset, _to_bool
from bankai_lightgbm.callback import (
    CallbackEnv,
    EarlyStopException,
    _EarlyStoppingCallback,
    early_stopping,
)

__all__ = ["train"]


def train(
    params: dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    *,
    valid_sets: Dataset | list[Dataset] | None = None,
    valid_names: list[str] | None = None,
    feval: Any = None,
    init_model: Any = None,
    feature_name: list[str] | str = "auto",
    categorical_feature: list[int | str] | str = "auto",
    keep_training_booster: bool = False,
    callbacks: list[Callable[[CallbackEnv], None]] | None = None,
) -> Booster:
    """Train a booster on ``train_set`` for ``num_boost_round`` iterations.

    Validation metrics and early stopping callbacks are evaluated after each iteration.
    """
    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    for name, value in (("feval", feval), ("init_model", init_model)):
        if value:
            raise NotImplementedError(f"{name} is not implemented yet (roadmap P5)")
    params = dict(params)
    if feature_name != "auto":
        train_set.feature_name = feature_name
    if categorical_feature != "auto":
        train_set.categorical_feature = categorical_feature
    train_set.params.update(params)
    booster = Booster(params=params, train_set=train_set)
    if valid_sets is None and not callbacks:
        booster._train(num_boost_round)
        return booster

    if valid_sets is None:
        eval_sets = []
    elif isinstance(valid_sets, Dataset):
        eval_sets = [valid_sets]
    else:
        eval_sets = list(valid_sets)
    if valid_names is not None and len(valid_names) != len(eval_sets):
        raise ValueError("Length of valid_names does not match valid_sets")
    names = valid_names or [f"valid_{i}" for i in range(len(eval_sets))]
    for valid_set, name in zip(eval_sets, names, strict=True):
        booster.add_valid(valid_set, name)

    callback_list: list[Callable[[CallbackEnv], None]] = list(callbacks or [])
    early_rounds_value = 0
    for alias in (
        "early_stopping_round",
        "early_stopping_rounds",
        "early_stopping",
        "n_iter_no_change",
    ):
        if alias in params:
            early_rounds_value = params[alias]
            break
    early_rounds = int(early_rounds_value or 0)
    if early_rounds > 0 and not any(
        isinstance(callback, _EarlyStoppingCallback) for callback in callback_list
    ):
        callback_list.append(
            early_stopping(
                early_rounds,
                first_metric_only=_to_bool(params.get("first_metric_only", False)),
                min_delta=float(params.get("early_stopping_min_delta", 0.0)),
            )
        )
    for callback in callback_list:
        before_training = getattr(callback, "before_training", None)
        if before_training is not None:
            model = before_training(booster)
            if model is not None:
                booster = model
    callback_list.sort(key=lambda callback: getattr(callback, "order", 0))
    before_callbacks = [
        callback for callback in callback_list if getattr(callback, "before_iteration", False)
    ]
    after_callbacks = [
        callback for callback in callback_list if not getattr(callback, "before_iteration", False)
    ]

    for iteration in range(num_boost_round):
        before_env = CallbackEnv(booster, params, iteration, 0, num_boost_round, None)
        for callback in before_callbacks:
            callback(before_env)
        finished = booster.update()
        evaluation_result_list = booster.eval_valid()
        booster.best_score = {}
        for data_name, metric_name, value, _ in evaluation_result_list:
            booster.best_score.setdefault(data_name, {})[metric_name] = value
        after_env = CallbackEnv(
            booster, params, iteration, 0, num_boost_round, evaluation_result_list
        )
        try:
            for callback in after_callbacks:
                callback(after_env)
        except EarlyStopException as stop:
            booster.best_iteration = stop.best_iteration + 1
            booster.best_score = {}
            for data_name, metric_name, value, _ in stop.best_score:
                booster.best_score.setdefault(data_name, {})[metric_name] = value
            break
        if finished:
            break
    if booster.best_iteration > 0 and not keep_training_booster:
        model_str = booster.model_to_string(num_iteration=booster.best_iteration)
        booster.model_from_string(model_str)
        booster.valid_sets = []
    return booster
