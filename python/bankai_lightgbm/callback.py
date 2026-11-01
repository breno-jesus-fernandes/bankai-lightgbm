"""Training callbacks. Mirrors the relevant parts of ``lightgbm.callback``."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

EvaluationResult = tuple[str, str, float, bool] | tuple[str, str, float, bool, float]


@dataclass
class CallbackEnv:
    """State passed to callbacks after each boosting iteration."""

    model: Any
    params: dict[str, object]
    iteration: int
    begin_iteration: int
    end_iteration: int
    evaluation_result_list: Sequence[EvaluationResult] | None


class EarlyStopException(Exception):
    """Internal signal used by the training loop when early stopping is reached."""

    def __init__(self, best_iteration: int, best_score: list[EvaluationResult]) -> None:
        super().__init__("early stopping")
        self.best_iteration = best_iteration
        self.best_score = best_score


def _format_eval_result(value: Sequence[Any], show_stdv: bool) -> str:
    """upstream: callback.py (v4.7.0) ``_format_eval_result``."""
    out = f"{value[0]}'s {value[1]}: {value[2]:g}"
    if show_stdv and len(value) == 5 and value[4] is not None:
        out += f" + {value[4]:g}"
    return out


def _is_using_cv(env: CallbackEnv) -> bool:
    from bankai_lightgbm.engine import CVBooster

    return isinstance(env.model, CVBooster)


def _gt_delta(curr_score: float, best_score: float, delta: float) -> bool:
    return curr_score > best_score + delta


def _lt_delta(curr_score: float, best_score: float, delta: float) -> bool:
    return curr_score < best_score - delta


def _should_enable_early_stopping(stopping_rounds: Any) -> bool:
    """upstream: callback.py (v4.7.0) ``_should_enable_early_stopping``."""
    if not isinstance(stopping_rounds, int):
        raise TypeError(
            f"early_stopping_round should be an integer. Got '{type(stopping_rounds).__name__}'"
        )
    return stopping_rounds > 0


class _EarlyStoppingCallback:
    """Early stopping, ported from upstream's ``callback._EarlyStoppingCallback`` (v4.7.0).

    Metrics are checked in evaluation order and the first one that has not improved for
    ``stopping_rounds`` iterations stops training with *its* best iteration; ``best_score``
    is the full evaluation list of that iteration. On the final iteration the same
    exception reports the best iteration so far. DART and training-set-only evaluation
    disable it with upstream's warnings.
    """

    def __init__(
        self,
        stopping_rounds: int,
        first_metric_only: bool = False,
        verbose: bool = True,
        min_delta: float | list[float] = 0.0,
    ) -> None:
        self.enabled = _should_enable_early_stopping(stopping_rounds)
        self.order = 30
        self.before_iteration = False
        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta
        self._reset_storages()

    def _reset_storages(self) -> None:
        self.best_score: list[float] = []
        self.best_iter: list[int] = []
        self.best_score_list: list[list[EvaluationResult]] = []
        self.cmp_op: list[Callable[..., bool]] = []
        self.first_metric = ""

    def _is_train_set(self, dataset_name: str, env: CallbackEnv) -> bool:
        if _is_using_cv(env) and dataset_name == "train":
            return True
        return dataset_name == getattr(env.model, "_train_data_name", None) and not _is_using_cv(
            env
        )

    def _init(self, env: CallbackEnv) -> None:
        from bankai_lightgbm.basic import _ConfigAliases, _log_info, _log_warning

        results = env.evaluation_result_list
        if not results:
            raise ValueError(
                "For early stopping, at least one dataset and eval metric is required for "
                "evaluation"
            )
        if any(
            env.params.get(alias, "") == "dart" for alias in _ConfigAliases.get_sorted("boosting")
        ):
            self.enabled = False
            _log_warning("Early stopping is not available in dart mode")
            return
        first_dataset_name, first_metric_name = results[0][0], results[0][1]
        if (
            not _is_using_cv(env)
            and len(results) == 1
            and self._is_train_set(first_dataset_name, env)
        ):
            self.enabled = False
            _log_warning("Only training set found, disabling early stopping.")
            return
        if self.verbose:
            _log_info(
                f"Training until validation scores don't improve for {self.stopping_rounds} rounds"
            )
        self._reset_storages()
        n_metrics = len({r[1] for r in results})
        n_datasets = len(results) // n_metrics
        if isinstance(self.min_delta, list):
            if not all(t >= 0 for t in self.min_delta):
                raise ValueError("Values for early stopping min_delta must be non-negative.")
            if len(self.min_delta) == 0:
                if self.verbose:
                    _log_info("Disabling min_delta for early stopping.")
                deltas = [0.0] * n_datasets * n_metrics
            elif len(self.min_delta) == 1:
                if self.verbose:
                    _log_info(f"Using {self.min_delta[0]} as min_delta for all metrics.")
                deltas = self.min_delta * n_datasets * n_metrics
            else:
                if len(self.min_delta) != n_metrics:
                    raise ValueError(
                        "Must provide a single value for min_delta or as many as metrics."
                    )
                if self.first_metric_only and self.verbose:
                    _log_info(f"Using only {self.min_delta[0]} as early stopping min_delta.")
                deltas = self.min_delta * n_datasets
        else:
            if self.min_delta < 0:
                raise ValueError("Early stopping min_delta must be non-negative.")
            if self.min_delta > 0 and n_metrics > 1 and not self.first_metric_only and self.verbose:
                _log_info(f"Using {self.min_delta} as min_delta for all metrics.")
            deltas = [self.min_delta] * n_datasets * n_metrics
        self.first_metric = first_metric_name
        for result, delta in zip(results, deltas, strict=True):
            self.best_iter.append(0)
            if result[3]:
                self.best_score.append(float("-inf"))
                self.cmp_op.append(partial(_gt_delta, delta=delta))
            else:
                self.best_score.append(float("inf"))
                self.cmp_op.append(partial(_lt_delta, delta=delta))

    def _final_iteration_check(self, env: CallbackEnv, metric_name: str, i: int) -> None:
        if env.iteration == env.end_iteration - 1:
            if self.verbose:
                from bankai_lightgbm.basic import _log_info

                best = "\t".join(_format_eval_result(x, True) for x in self.best_score_list[i])
                _log_info(
                    "Did not meet early stopping. Best iteration is:\n"
                    f"[{self.best_iter[i] + 1}]\t{best}"
                )
                if self.first_metric_only:
                    _log_info(f"Evaluated only: {metric_name}")
            raise EarlyStopException(self.best_iter[i], self.best_score_list[i])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        results = env.evaluation_result_list
        if results is None:
            raise RuntimeError("early_stopping() callback enabled but no evaluation results found.")
        first_time = self.best_score_list == []
        for i, result in enumerate(results):
            score = result[2]
            if first_time or self.cmp_op[i](score, self.best_score[i]):
                self.best_score[i] = score
                self.best_iter[i] = env.iteration
                if first_time:
                    self.best_score_list.append(list(results))
                else:
                    self.best_score_list[i] = list(results)
            if self.first_metric_only and self.first_metric != result[1]:
                continue
            if self._is_train_set(result[0], env):
                continue
            if env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    from bankai_lightgbm.basic import _log_info

                    best = "\t".join(_format_eval_result(x, True) for x in self.best_score_list[i])
                    _log_info(
                        f"Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t{best}"
                    )
                    if self.first_metric_only:
                        _log_info(f"Evaluated only: {result[1]}")
                raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
            self._final_iteration_check(env, result[1], i)


def early_stopping(
    stopping_rounds: int,
    first_metric_only: bool = False,
    verbose: bool = True,
    min_delta: float | list[float] = 0.0,
) -> _EarlyStoppingCallback:
    """Stop when a validation metric fails to improve for ``stopping_rounds`` rounds.

    Same semantics as ``lightgbm.early_stopping``; ``stopping_rounds <= 0`` disables it.
    """
    return _EarlyStoppingCallback(
        stopping_rounds=stopping_rounds,
        first_metric_only=first_metric_only,
        verbose=verbose,
        min_delta=min_delta,
    )


class _LogEvaluationCallback:
    order = 10
    before_iteration = False

    def __init__(self, period: int, show_stdv: bool) -> None:
        self.period = period
        self.show_stdv = show_stdv

    def __call__(self, env: CallbackEnv) -> None:
        results = env.evaluation_result_list
        if self.period <= 0 or not results or (env.iteration + 1) % self.period != 0:
            return
        formatted_results = []
        for result in results:
            data_name, metric_name, value, _ = result[:4]
            formatted = f"{data_name}'s {metric_name}: {value:g}"
            if self.show_stdv and len(result) == 5:
                formatted += f" + {result[4]:g}"
            formatted_results.append(formatted)
        formatted = "\t".join(formatted_results)
        logging.getLogger("bankai_lightgbm").info("[%d]\t%s", env.iteration + 1, formatted)


def log_evaluation(period: int = 1, show_stdv: bool = True) -> _LogEvaluationCallback:
    """Log validation metrics every ``period`` iterations (zero disables logging)."""
    return _LogEvaluationCallback(period, show_stdv)


class _RecordEvaluationCallback:
    order = 20
    before_iteration = False

    def __init__(self, eval_result: dict[str, dict[str, list[float]]]) -> None:
        if not isinstance(eval_result, dict):
            raise TypeError("eval_result should be a dictionary")
        self.eval_result = eval_result

    def before_training(self, model: Any) -> Any:
        self.eval_result.clear()
        return model

    def __call__(self, env: CallbackEnv) -> None:
        if not env.evaluation_result_list:
            raise RuntimeError(
                "record_evaluation() callback enabled but no evaluation results were found"
            )
        for result in env.evaluation_result_list:
            data_name, metric_name, value, _ = result[:4]
            metrics = self.eval_result.setdefault(data_name, {})
            if len(result) == 5:
                metrics.setdefault(f"{metric_name}-mean", []).append(value)
                metrics.setdefault(f"{metric_name}-stdv", []).append(result[4])
            else:
                metrics.setdefault(metric_name, []).append(value)


def record_evaluation(
    eval_result: dict[str, dict[str, list[float]]],
) -> _RecordEvaluationCallback:
    """Record each iteration's validation metrics in ``eval_result``."""
    return _RecordEvaluationCallback(eval_result)


class _ResetParameterCallback:
    order = 10
    before_iteration = True

    def __init__(self, parameters: dict[str, list[object] | Callable[[int], object]]) -> None:
        self.parameters = parameters

    def __call__(self, env: CallbackEnv) -> None:
        new_parameters: dict[str, object] = {}
        iteration = env.iteration - env.begin_iteration
        for name, schedule in self.parameters.items():
            if isinstance(schedule, list):
                if len(schedule) != env.end_iteration - env.begin_iteration:
                    raise ValueError(
                        f"Length of list {name!r} has to be equal to 'num_boost_round'."
                    )
                value = schedule[iteration]
            elif callable(schedule):
                value = schedule(iteration)
            else:
                raise ValueError(
                    "Only list and callable values are supported as a mapping "
                    "from boosting round index to new parameter value."
                )
            if value != env.params.get(name):
                new_parameters[name] = value
        if new_parameters:
            env.model.reset_parameter(new_parameters)
            env.params.update(new_parameters)


def reset_parameter(
    **kwargs: list[object] | Callable[[int], object],
) -> _ResetParameterCallback:
    """Reset a supported parameter from a per-iteration list or callback function."""
    return _ResetParameterCallback(kwargs)
