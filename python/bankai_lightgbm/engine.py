"""Training entry point. Mirrors ``lightgbm.engine``."""

from __future__ import annotations

from typing import Any

from bankai_lightgbm.basic import Booster, Dataset

__all__ = ["train"]


def train(
    params: dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    *,
    valid_sets: list[Dataset] | None = None,
    valid_names: list[str] | None = None,
    feval: Any = None,
    init_model: Any = None,
    feature_name: list[str] | str = "auto",
    categorical_feature: list[int | str] | str = "auto",
    keep_training_booster: bool = False,
    callbacks: list[Any] | None = None,
) -> Booster:
    """Train a booster on ``train_set`` for ``num_boost_round`` iterations.

    Validation sets, custom metrics, callbacks and continued training are roadmap P5 and
    raise ``NotImplementedError`` until they land.
    """
    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    for name, value in (
        ("valid_sets", valid_sets),
        ("feval", feval),
        ("init_model", init_model),
        ("callbacks", callbacks),
    ):
        if value:
            raise NotImplementedError(f"{name} is not implemented yet (roadmap P5)")
    params = dict(params)
    if feature_name != "auto":
        train_set.feature_name = feature_name
    if categorical_feature != "auto":
        train_set.categorical_feature = categorical_feature
    train_set.params.update(params)
    booster = Booster(params=params, train_set=train_set)
    booster._train(num_boost_round)
    return booster
