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


def test_train_rejects_unimplemented_options() -> None:
    x = np.random.default_rng(0).normal(size=(40, 2))
    ds = lgb.Dataset(x, label=x[:, 0], params={"verbose": -1})
    with pytest.raises(NotImplementedError, match="valid_sets"):
        lgb.train({"verbose": -1}, ds, valid_sets=[ds])
    with pytest.raises(ValueError, match="greater than zero"):
        lgb.train({"verbose": -1}, ds, num_boost_round=0)
