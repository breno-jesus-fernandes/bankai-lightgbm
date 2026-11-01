"""Training entry point. Mirrors ``lightgbm.engine``."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from bankai_lightgbm.basic import Booster, Dataset, LightGBMError, _to_bool
from bankai_lightgbm.callback import (
    CallbackEnv,
    EarlyStopException,
    _EarlyStoppingCallback,
    early_stopping,
)

__all__ = ["CVBooster", "cv", "train"]


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
    if init_model is not None:
        raise NotImplementedError("init_model is not implemented yet (roadmap P5)")
    params = dict(params)
    fobj = params.get("objective") if callable(params.get("objective")) else None
    if fobj is not None:
        params["objective"] = "none"
    if feature_name != "auto":
        train_set.feature_name = feature_name
    if categorical_feature != "auto":
        train_set.categorical_feature = categorical_feature
    train_set.params.update(params)
    booster = Booster(params=params, train_set=train_set)
    if valid_sets is None and not callbacks and fobj is None:
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
        finished = booster.update(fobj=fobj)
        evaluation_result_list = booster.eval_valid(feval=feval) or None
        booster.best_score = {}
        for data_name, metric_name, value, _ in evaluation_result_list or []:
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
            for result in stop.best_score:
                data_name, metric_name, value = result[0], result[1], result[2]
                booster.best_score.setdefault(data_name, {})[metric_name] = value
            break
        if finished:
            break
    if booster.best_iteration > 0 and not keep_training_booster:
        model_str = booster.model_to_string(num_iteration=booster.best_iteration)
        booster.model_from_string(model_str)
        booster.valid_sets = []
    return booster


class CVBooster:
    """Collection of fold boosters returned by :func:`cv`."""

    def __init__(self, boosters: list[Booster] | None = None) -> None:
        self.boosters = boosters or []
        self.best_iteration = -1

    def update(self, fobj: Any = None) -> list[bool]:
        """Update every fold model by one boosting iteration."""
        return [booster.update(fobj=fobj) for booster in self.boosters]

    def eval_valid(self, feval: Any = None) -> list[list[tuple[str, str, float, bool]]]:
        """Return validation metric results for each fold."""
        return [booster.eval_valid(feval=feval) for booster in self.boosters]

    def reset_parameter(self, params: dict[str, Any]) -> None:
        """Reset the same supported training parameters on every fold model."""
        for booster in self.boosters:
            booster.reset_parameter(params)

    def model_to_string(
        self,
        num_iteration: int | None = None,
        start_iteration: int = 0,
        importance_type: str = "split",
    ) -> str:
        """Serialize fold models in the upstream CVBooster JSON shape."""
        return json.dumps(
            {
                "boosters": [
                    booster.model_to_string(
                        num_iteration=num_iteration,
                        start_iteration=start_iteration,
                        importance_type=importance_type,
                    )
                    for booster in self.boosters
                ],
                "best_iteration": self.best_iteration,
            }
        )

    def model_from_string(self, model_str: str) -> CVBooster:
        """Replace fold models from a serialized CVBooster."""
        from bankai_lightgbm.basic import Booster as BoosterClass

        decoded = json.loads(model_str)
        self.boosters = [BoosterClass(model_str=value) for value in decoded["boosters"]]
        self.best_iteration = int(decoded["best_iteration"])
        return self

    def save_model(
        self,
        filename: str | Path,
        num_iteration: int | None = None,
        start_iteration: int = 0,
        importance_type: str = "split",
    ) -> CVBooster:
        """Write the serialized fold models to ``filename``."""
        Path(filename).write_text(
            self.model_to_string(num_iteration, start_iteration, importance_type)
        )
        return self

    def __getattr__(self, name: str) -> Any:
        def apply_to_boosters(*args: Any, **kwargs: Any) -> list[Any]:
            return [getattr(booster, name)(*args, **kwargs) for booster in self.boosters]

        return apply_to_boosters


def _make_cv_folds(
    train_set: Dataset,
    *,
    folds: Iterable[tuple[Any, Any]] | Any | None,
    nfold: int,
    stratified: bool,
    shuffle: bool,
    seed: int,
    params: dict[str, Any],
) -> list[tuple[list[int], list[int]]]:
    num_data = train_set.num_data()
    labels = np.asarray(train_set.label) if train_set.label is not None else None
    if folds is not None and hasattr(folds, "split"):
        group_sizes = train_set.group
        groups = None
        if group_sizes is not None:
            groups = np.repeat(np.arange(len(group_sizes)), np.asarray(group_sizes, dtype=int))
        raw_folds = folds.split(np.empty(num_data), y=labels, groups=groups)
    elif folds is not None:
        raw_folds = iter(folds)
    else:
        objective = str(params.get("objective", params.get("application", ""))).lower()
        ranking_objectives = {
            "lambdarank",
            "rank_xendcg",
            "xendcg",
            "xe_ndcg",
            "xe_ndcg_mart",
            "xendcg_mart",
        }
        splitters: tuple[Any, Any, Any] | None
        try:
            from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

            splitters = (GroupKFold, KFold, StratifiedKFold)
        except ImportError as error:
            if stratified or objective in ranking_objectives:
                raise LightGBMError(
                    "scikit-learn is required for stratified or ranking cv"
                ) from error
            splitters = None

        if objective in ranking_objectives:
            if train_set.group is None:
                raise ValueError("Ranking cross-validation requires group information")
            group_sizes = np.asarray(train_set.group, dtype=int)
            groups = np.repeat(np.arange(len(group_sizes)), group_sizes)
            if splitters is None:
                raise LightGBMError("scikit-learn is required for ranking cv")
            splitter = splitters[0](n_splits=nfold)
            raw_folds = splitter.split(np.empty(num_data), groups=groups)
        elif stratified:
            if labels is None:
                raise ValueError("Stratified cross-validation requires labels")
            if splitters is None:
                raise LightGBMError("scikit-learn is required for stratified cv")
            splitter = splitters[2](n_splits=nfold, shuffle=shuffle, random_state=seed)
            raw_folds = splitter.split(np.empty(num_data), labels)
        elif splitters is not None:
            splitter = splitters[1](
                n_splits=nfold, shuffle=shuffle, random_state=seed if shuffle else None
            )
            raw_folds = splitter.split(np.empty(num_data))
        else:
            if nfold < 2 or nfold > num_data:
                raise ValueError("nfold must be at least 2 and no greater than the row count")
            row_ids = (
                np.random.RandomState(seed).permutation(num_data)
                if shuffle
                else np.arange(num_data)
            )
            test_folds = np.array_split(row_ids, nfold)
            raw_folds = (
                (
                    np.concatenate([test_folds[i] for i in range(nfold) if i != fold]),
                    test_folds[fold],
                )
                for fold in range(nfold)
            )

    result: list[tuple[list[int], list[int]]] = []
    for train_indices, valid_indices in raw_folds:
        train_rows = np.sort(np.asarray(train_indices, dtype=np.int64))
        valid_rows = np.sort(np.asarray(valid_indices, dtype=np.int64))
        if train_rows.size == 0 or valid_rows.size == 0:
            raise ValueError("Every cross-validation fold must have training and validation rows")
        result.append((train_rows.tolist(), valid_rows.tolist()))
    if not result:
        raise ValueError("folds must contain at least one train/validation split")
    return result


def cv(
    params: dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    folds: Iterable[tuple[Any, Any]] | Any | None = None,
    nfold: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    metrics: str | list[str] | None = None,
    feval: Any = None,
    init_model: Any = None,
    fpreproc: Any = None,
    seed: int = 0,
    callbacks: list[Callable[[CallbackEnv], None]] | None = None,
    eval_train_metric: bool = False,
    return_cvbooster: bool = False,
) -> dict[str, Any]:
    """Perform seeded cross-validation and aggregate per-fold metrics."""
    if not isinstance(train_set, Dataset):
        raise TypeError("cv() only accepts a Dataset as train_set")
    if num_boost_round <= 0:
        raise ValueError("Number of boosting rounds must be greater than 0")
    for name, value in (("init_model", init_model), ("fpreproc", fpreproc)):
        if value is not None:
            raise NotImplementedError(f"{name} is not implemented yet")
    params = dict(params)
    fobj = params.get("objective") if callable(params.get("objective")) else None
    if fobj is not None:
        params["objective"] = "none"
    if metrics is not None:
        for alias in ("metric", "metrics", "metric_types"):
            params.pop(alias, None)
        params["metric"] = metrics
    folds_list = _make_cv_folds(
        train_set,
        folds=folds,
        nfold=nfold,
        stratified=stratified,
        shuffle=shuffle,
        seed=seed,
        params=params,
    )
    boosters: list[Booster] = []
    for train_rows, valid_rows in folds_list:
        fold_train = train_set.subset(train_rows, params)
        fold_valid = train_set.subset(valid_rows, params)
        booster = Booster(params=params, train_set=fold_train)
        if eval_train_metric:
            booster.add_valid(fold_train, "train")
        booster.add_valid(fold_valid, "valid")
        boosters.append(booster)
    cvbooster = CVBooster(boosters)

    callback_list = list(callbacks or [])
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
                verbose=int(params.get("verbosity", 1)) > 0,
                min_delta=float(params.get("early_stopping_min_delta", 0.0)),
            )
        )
    for callback in callback_list:
        before_training = getattr(callback, "before_training", None)
        if before_training is not None:
            before_training(cvbooster)
    callback_list.sort(key=lambda callback: getattr(callback, "order", 0))
    before_callbacks = [
        callback for callback in callback_list if getattr(callback, "before_iteration", False)
    ]
    after_callbacks = [
        callback for callback in callback_list if not getattr(callback, "before_iteration", False)
    ]

    results: dict[str, list[float]] = OrderedDict()
    for iteration in range(num_boost_round):
        before_env = CallbackEnv(cvbooster, params, iteration, 0, num_boost_round, None)
        for callback in before_callbacks:
            callback(before_env)
        cvbooster.update(fobj=fobj)
        fold_results = cvbooster.eval_valid(feval=feval)
        metric_values: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        metric_flags: dict[tuple[str, str], bool] = {}
        for fold_result in fold_results:
            for data_name, metric_name, value, higher_is_better in fold_result:
                key = (data_name, metric_name)
                metric_values.setdefault(key, []).append(value)
                metric_flags[key] = higher_is_better
        evaluation_result_list: list[tuple[str, str, float, bool, float]] = []
        for (data_name, metric_name), values in metric_values.items():
            mean = float(np.mean(values))
            std = float(np.std(values))
            results.setdefault(f"{data_name} {metric_name}-mean", []).append(mean)
            results.setdefault(f"{data_name} {metric_name}-stdv", []).append(std)
            evaluation_result_list.append(
                (data_name, metric_name, mean, metric_flags[(data_name, metric_name)], std)
            )
        after_env = CallbackEnv(
            cvbooster, params, iteration, 0, num_boost_round, evaluation_result_list or None
        )
        try:
            for callback in after_callbacks:
                callback(after_env)
        except EarlyStopException as stop:
            cvbooster.best_iteration = stop.best_iteration + 1
            for booster in boosters:
                booster.best_iteration = cvbooster.best_iteration
            results = OrderedDict(
                (name, values[: cvbooster.best_iteration]) for name, values in results.items()
            )
            break

    if return_cvbooster:
        results["cvbooster"] = cvbooster  # type: ignore[assignment]
    return dict(results)
