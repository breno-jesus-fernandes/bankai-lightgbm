"""Training callbacks. Mirrors the relevant parts of ``lightgbm.callback``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bankai_lightgbm.basic import Booster

EvaluationResult = tuple[str, str, float, bool]


@dataclass
class CallbackEnv:
    """State passed to callbacks after each boosting iteration."""

    model: Booster
    params: dict[str, object]
    iteration: int
    begin_iteration: int
    end_iteration: int
    evaluation_result_list: list[EvaluationResult] | None


class EarlyStopException(Exception):
    """Internal signal used by the training loop when early stopping is reached."""

    def __init__(self, best_iteration: int, best_score: list[EvaluationResult]) -> None:
        super().__init__("early stopping")
        self.best_iteration = best_iteration
        self.best_score = best_score


class _EarlyStoppingCallback:
    order = 30
    before_iteration = False

    def __init__(
        self,
        stopping_rounds: int,
        first_metric_only: bool,
        verbose: bool,
        min_delta: float,
    ) -> None:
        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta
        self.best: dict[tuple[str, str], tuple[float, int, EvaluationResult]] = {}
        self.first_metric: str | None = None

    def before_training(self, model: Booster) -> Booster:
        self.best.clear()
        self.first_metric = None
        return model

    def __call__(self, env: CallbackEnv) -> None:
        results = env.evaluation_result_list
        if not results:
            return
        if self.first_metric is None:
            self.first_metric = results[0][1]
        if self.first_metric_only:
            results = [result for result in results if result[1] == self.first_metric]
        if not results:
            return

        for result in results:
            data_name, metric_name, score, higher_is_better = result
            key = (data_name, metric_name)
            previous = self.best.get(key)
            improved = previous is None or (
                score > previous[0] + self.min_delta
                if higher_is_better
                else score < previous[0] - self.min_delta
            )
            if improved:
                self.best[key] = (score, env.iteration, result)

        if self.best and all(
            env.iteration - best_iteration >= self.stopping_rounds
            for _, best_iteration, _ in self.best.values()
        ):
            first_key = next(iter(self.best))
            best_iteration = self.best[first_key][1]
            best_score = [entry[2] for entry in self.best.values()]
            if self.verbose:
                print(f"Early stopping, best iteration is: {best_iteration + 1}")
            raise EarlyStopException(best_iteration, best_score)


def early_stopping(
    stopping_rounds: int,
    first_metric_only: bool = False,
    verbose: bool = True,
    min_delta: float = 0.0,
) -> _EarlyStoppingCallback:
    """Stop when validation metrics fail to improve for ``stopping_rounds`` rounds."""
    if stopping_rounds <= 0:
        raise ValueError("stopping_rounds should be greater than zero")
    if min_delta < 0.0:
        raise ValueError("min_delta should be non-negative")
    return _EarlyStoppingCallback(stopping_rounds, first_metric_only, verbose, min_delta)


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
        formatted = "\t".join(
            f"{data_name}'s {metric_name}: {value:g}"
            for data_name, metric_name, value, _ in results
        )
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

    def before_training(self, model: Booster) -> Booster:
        self.eval_result.clear()
        return model

    def __call__(self, env: CallbackEnv) -> None:
        if not env.evaluation_result_list:
            raise RuntimeError(
                "record_evaluation() callback enabled but no evaluation results were found"
            )
        for data_name, metric_name, value, _ in env.evaluation_result_list:
            metrics = self.eval_result.setdefault(data_name, {})
            metrics.setdefault(metric_name, []).append(value)


def record_evaluation(
    eval_result: dict[str, dict[str, list[float]]],
) -> _RecordEvaluationCallback:
    """Record each iteration's validation metrics in ``eval_result``."""
    return _RecordEvaluationCallback(eval_result)
