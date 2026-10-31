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


def test_train_rejects_unimplemented_options() -> None:
    x = np.random.default_rng(0).normal(size=(40, 2))
    ds = lgb.Dataset(x, label=x[:, 0], params={"verbose": -1})
    with pytest.raises(NotImplementedError, match="feval"):
        lgb.train({"verbose": -1}, ds, feval=lambda *_: [])
    referenced = lgb.Dataset(x, label=x[:, 0], reference=ds)
    with pytest.raises(NotImplementedError, match="shared reference binning"):
        referenced.construct()
    with pytest.raises(ValueError, match="greater than zero"):
        lgb.train({"verbose": -1}, ds, num_boost_round=0)
