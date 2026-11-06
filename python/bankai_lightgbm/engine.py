"""Training entry point. Mirrors ``lightgbm.engine``."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from bankai_lightgbm.basic import (
    Booster,
    Dataset,
    LightGBMError,
    _choose_param_value,
    _ConfigAliases,
    _log_warning,
    _to_bool,
)
from bankai_lightgbm.callback import (
    CallbackEnv,
    EarlyStopException,
    _EarlyStoppingCallback,
    _should_enable_early_stopping,
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
    # upstream: python-package/lightgbm/engine.py (v4.7.0) train(), input checks
    if not isinstance(train_set, Dataset):
        raise TypeError(
            f"train() only accepts Dataset object, train_set has type '{type(train_set).__name__}'."
        )
    if isinstance(valid_sets, list):
        for i, valid_item in enumerate(valid_sets):
            if not isinstance(valid_item, Dataset):
                raise TypeError(
                    "Every item in valid_sets must be a Dataset object. "
                    f"Item {i} has type '{type(valid_item).__name__}'."
                )
    params = dict(params)
    # upstream: engine.py::_choose_param_value, objective aliases (v4.7.0).
    objective_aliases = ("objective", "app", "loss", "application", "objective_type")
    objective = next(
        (params[objective_key] for objective_key in objective_aliases if objective_key in params),
        None,
    )
    fobj = objective if callable(objective) else None
    if fobj is not None:
        for objective_key in objective_aliases:
            params.pop(objective_key, None)
        params["objective"] = "none"
    params = _choose_num_iterations(num_boost_round_kwarg=num_boost_round, params=params)
    num_boost_round = int(params["num_iterations"])
    # upstream: engine.py (v4.7.0) train() resolves early stopping from the global params first,
    # and drops it when it is None.
    params = _choose_param_value(
        main_param_name="early_stopping_round", params=params, default_value=None
    )
    if params["early_stopping_round"] is None:
        params.pop("early_stopping_round")
    _should_enable_early_stopping(
        params.get("early_stopping_round", 0)
    )  # type check before Booster
    if num_boost_round <= 0:
        raise ValueError(
            f"Number of boosting rounds must be greater than 0. Got {num_boost_round}."
        )
    if feature_name != "auto":
        train_set.feature_name = feature_name
    if categorical_feature != "auto":
        train_set.categorical_feature = categorical_feature
    # upstream: engine.py (v4.7.0) train(), `train_set._reverse_update_params()` in `finally`:
    # the training parameters are only applied to the Dataset while it trains.
    train_set._check_update_params(params)
    params_back_up = dict(train_set.params)
    train_set.params.update(params)
    # upstream: engine.py (v4.7.0) train(), validation params are reversed in `finally`.
    valid_params_back_up: list[tuple[Dataset, dict[str, Any]]] = []
    try:
        if train_set.position is not None and str(
            params.get("objective", params.get("application", ""))
        ).lower() in {
            "lambdarank",
            "rank_xendcg",
            "xendcg",
            "xe_ndcg",
            "xe_ndcg_mart",
            "xendcg_mart",
        }:
            raise NotImplementedError(
                "position bias for ranking objectives is not implemented yet (ROADMAP P4)"
            )
        booster = Booster(params=params, train_set=train_set)
        if init_model is not None:
            booster._continue_from(init_model)
        # upstream: engine.py (v4.7.0) train() resets it after constructing the Booster.
        booster.best_iteration = 0
        if valid_sets is None and not callbacks and fobj is None:
            booster._train(num_boost_round)
            return _finish_training(booster, keep_training_booster)

        if valid_sets is None:
            eval_sets = []
        elif isinstance(valid_sets, Dataset):
            eval_sets = [valid_sets]
        else:
            eval_sets = list(valid_sets)
        if isinstance(valid_names, str):
            # upstream: engine.py (v4.7.0) train() accepts a single name
            valid_names = [valid_names]
        # upstream: engine.py (v4.7.0) train(). The training set is evaluated first, under
        # ``valid_names[i]`` or "training"; other sets keep their original index in "valid_{i}".
        # A ``valid_names`` shorter than ``valid_sets`` only names the first sets, as upstream does.
        named: list[tuple[Dataset, str]] = []
        for i, valid_set in enumerate(eval_sets):
            given = valid_names[i] if valid_names is not None and i < len(valid_names) else None
            if valid_set is train_set:
                booster._train_data_name = given if given is not None else "training"
                named.insert(0, (valid_set, booster._train_data_name))
            else:
                named.append((valid_set, given if given is not None else f"valid_{i}"))
        for valid_set, name in named:
            if valid_set is not train_set:
                # upstream: engine.py (v4.7.0) train(): _update_params, then set_reference.
                valid_params_back_up.append((valid_set, dict(valid_set.params)))
                valid_set.params.update(params)
                valid_set.set_reference(train_set)
            booster.add_valid(valid_set, name)

        callback_list: list[Callable[[CallbackEnv], None]] = list(callbacks or [])
        if _should_enable_early_stopping(params.get("early_stopping_round", 0)) and not any(
            isinstance(callback, _EarlyStoppingCallback) for callback in callback_list
        ):
            callback_list.append(
                early_stopping(
                    params["early_stopping_round"],
                    first_metric_only=_to_bool(params.get("first_metric_only", False)),
                    min_delta=float(params.get("early_stopping_min_delta", 0.0)),
                    verbose=int(_choose_param_value("verbosity", params, 1)["verbosity"]) > 0,
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
            callback
            for callback in callback_list
            if not getattr(callback, "before_iteration", False)
        ]

        for iteration in range(num_boost_round):
            before_env = CallbackEnv(booster, params, iteration, 0, num_boost_round, None)
            for callback in before_callbacks:
                callback(before_env)
            # upstream: the is_finished return of update is ignored; every round runs.
            booster.update(fobj=fobj)
            # upstream: an empty list (no validation sets) is passed on, not None.
            evaluation_result_list = booster.eval_valid(feval=feval)
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
                for result in stop.best_score:
                    data_name, metric_name, value = result[0], result[1], result[2]
                    booster.best_score.setdefault(data_name, {})[metric_name] = value
                break
        return _finish_training(booster, keep_training_booster)
    finally:
        train_set.params = params_back_up
        for valid_set, back_up in valid_params_back_up:
            valid_set.params = back_up


def _finish_training(booster: Booster, keep_training_booster: bool) -> Booster:
    """upstream: engine.py (v4.7.0) train() ends with
    ``booster.model_from_string(booster.model_to_string()).free_dataset()`` unless
    ``keep_training_booster``: the returned model is the reparsed text, truncated to
    ``best_iteration`` (a one-leaf tree, for example, loses its ``leaf_weight``)."""
    if not keep_training_booster:
        booster.model_from_string(booster.model_to_string()).free_dataset()
    return booster


class CVBooster:
    """Collection of fold boosters returned by :func:`cv`."""

    def __init__(
        self,
        boosters: list[Booster] | None = None,
        model_file: str | Path | None = None,
    ) -> None:
        """Create a CVBooster from fold boosters, or load one from ``model_file``.

        upstream: python-package/lightgbm/engine.py (v4.7.0) ``CVBooster.__init__``.
        """
        self.boosters = boosters or []
        self.best_iteration = -1
        if model_file is not None:
            self.model_from_string(Path(model_file).read_text())

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

    # upstream: python-package/lightgbm/engine.py (v4.7.0) CVBooster.__getstate__/__setstate__.
    # Defined explicitly so pickle does not hit ``__getattr__`` before ``boosters`` exists.
    def __getstate__(self) -> dict[str, Any]:
        return vars(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)


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


def _choose_num_iterations(*, num_boost_round_kwarg: int, params: dict[str, Any]) -> dict[str, Any]:
    """Resolve the number of boosting rounds into ``params["num_iterations"]``.

    ``num_iterations`` in ``params`` wins, then its aliases (in sorted order), then the
    ``num_boost_round`` keyword; conflicting values in ``params`` log a warning.
    upstream: python-package/lightgbm/engine.py (v4.7.0) ``_choose_num_iterations``.
    """
    provided = {
        alias: params[alias]
        for alias in _ConfigAliases.get_sorted("num_iterations")
        if alias in params
    }
    params = _choose_param_value(
        main_param_name="num_iterations", params=params, default_value=num_boost_round_kwarg
    )
    if len(provided) <= 1 or len(set(provided.values())) <= 1:
        return params
    value_string = ", ".join(f"{alias}={val}" for alias, val in provided.items())
    _log_warning(
        f"Found conflicting values for num_iterations provided via 'params': {value_string}. "
        f"LightGBM will perform up to {params['num_iterations']} boosting rounds. "
        "To be confident in the maximum number of boosting rounds LightGBM will perform and to "
        "suppress this warning, modify 'params' so that only one of those is present."
    )
    return params


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
        raise TypeError(
            f"cv() only accepts Dataset object, train_set has type '{type(train_set).__name__}'."
        )
    # upstream: engine.py (v4.7.0) cv() resolves init_model into a predictor shared by all folds.
    init_booster: Booster | None = None
    if isinstance(init_model, Booster):
        init_booster = init_model
    elif isinstance(init_model, (str, Path)):
        init_booster = Booster(model_file=init_model)
    elif init_model is not None:
        raise TypeError("init_model should be a Booster or a path to a model file")
    params = dict(params)
    # upstream: engine.py::_choose_param_value, objective aliases (v4.7.0).
    objective_aliases = ("objective", "app", "loss", "application", "objective_type")
    objective = next(
        (params[objective_key] for objective_key in objective_aliases if objective_key in params),
        None,
    )
    fobj = objective if callable(objective) else None
    if fobj is not None:
        for objective_key in objective_aliases:
            params.pop(objective_key, None)
        params["objective"] = "none"
    params = _choose_num_iterations(num_boost_round_kwarg=num_boost_round, params=params)
    num_boost_round = int(params["num_iterations"])
    if num_boost_round <= 0:
        raise ValueError(
            f"Number of boosting rounds must be greater than 0. Got {num_boost_round}."
        )
    if metrics is not None:
        for alias in ("metric", "metrics", "metric_types"):
            params.pop(alias, None)
        params["metric"] = metrics
    # upstream: engine.py (v4.7.0) cv(), the Dataset update is reversed when cv returns.
    train_set._check_update_params(params)
    params_back_up = dict(train_set.params)
    train_set.params.update(params)
    try:
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
            fold_params = params
            if fpreproc is not None:
                # upstream: engine.py (v4.7.0) _make_n_folds, fpreproc(train, valid, params.copy())
                # returns the datasets and parameters used for this fold.
                fold_train, fold_valid, fold_params = fpreproc(
                    fold_train, fold_valid, params.copy()
                )
            if init_booster is not None:
                # upstream: the fold's validation init score is the init model's prediction.
                fold_valid.set_init_score(init_booster.predict(fold_valid.data, raw_score=True))
            booster = Booster(params=fold_params, train_set=fold_train)
            if init_booster is not None:
                booster._continue_from(init_booster)
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
            callback
            for callback in callback_list
            if not getattr(callback, "before_iteration", False)
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
    finally:
        train_set.params = params_back_up
    return dict(results)
