"""Generate golden fixtures with upstream LightGBM.

Each case writes, under ``tests/fixtures/<case>/``:
  - ``train.csv`` / ``test.csv``  (label in column 0)
  - ``model.txt`` and ``model.json`` (upstream ``save_model`` / ``dump_model``)
  - ``pred.npy``, ``pred_raw.npy``, ``pred_leaf.npy``, ``pred_contrib.npy``
  - ``params.json``

Usage: ``python scripts/gen_fixtures.py`` (needs ``lightgbm`` at the pinned version).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
UPSTREAM_VERSION = (ROOT / "UPSTREAM_VERSION").read_text().strip()

BASE: dict[str, Any] = {
    "deterministic": True,
    "num_threads": 1,
    "seed": 42,
    "verbose": -1,
}

CASES: dict[str, dict[str, Any]] = {
    "regression_l2": {"objective": "regression", "num_leaves": 15},
    "binary": {"objective": "binary", "num_leaves": 15},
    "multiclass": {"objective": "multiclass", "num_class": 3, "num_leaves": 7},
    "categorical": {"objective": "regression", "num_leaves": 15},
    "missing": {"objective": "regression", "num_leaves": 15},
    "lambdarank": {"objective": "lambdarank", "num_leaves": 15},
}


def make_data(case: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, d = 2000, 10
    x = rng.normal(size=(n, d))
    signal = x[:, 0] * 2 + np.sin(x[:, 1] * 3) + x[:, 2] * x[:, 3]
    if case == "categorical":
        x[:, 4] = rng.integers(0, 20, size=n)
        signal += np.where(x[:, 4] % 3 == 0, 1.5, -0.5)
    if case == "missing":
        x[rng.random(size=(n, d)) < 0.15] = np.nan
        signal = np.nan_to_num(signal)
    if case == "binary":
        y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(np.float64)
    elif case == "multiclass":
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3])).astype(np.float64)
    elif case == "lambdarank":
        y = np.clip(np.digitize(signal, np.quantile(signal, [0.5, 0.8, 0.95])), 0, 3).astype(
            np.float64
        )
    else:
        y = signal + rng.normal(scale=0.1, size=n)
    return x, y


def generate(case: str, extra: dict[str, Any]) -> None:
    rng = np.random.default_rng(0)
    x, y = make_data(case, rng)
    split = int(len(y) * 0.8)
    out = FIXTURES / case
    out.mkdir(parents=True, exist_ok=True)

    params = {**BASE, **extra}
    ds_kwargs: dict[str, Any] = {}
    if case == "categorical":
        ds_kwargs["categorical_feature"] = [4]
    if case == "lambdarank":
        ds_kwargs["group"] = [20] * (split // 20)

    train = lgb.Dataset(x[:split], y[:split], **ds_kwargs)
    booster = lgb.train(params, train, num_boost_round=20)

    x_test = x[split:]
    np.savetxt(out / "train.csv", np.column_stack([y[:split], x[:split]]), delimiter=",")
    np.savetxt(out / "test.csv", np.column_stack([y[split:], x_test]), delimiter=",")
    booster.save_model(str(out / "model.txt"))
    (out / "model.json").write_text(json.dumps(booster.dump_model(), indent=1))
    np.save(out / "pred.npy", booster.predict(x_test))
    np.save(out / "pred_raw.npy", booster.predict(x_test, raw_score=True))
    np.save(out / "pred_leaf.npy", booster.predict(x_test, pred_leaf=True))
    np.save(out / "pred_contrib.npy", booster.predict(x_test, pred_contrib=True))
    (out / "params.json").write_text(
        json.dumps({"params": params, "dataset": ds_kwargs, "num_boost_round": 20}, indent=1)
    )


def main() -> None:
    expected = UPSTREAM_VERSION.lstrip("v")
    if lgb.__version__ != expected:
        raise SystemExit(f"need lightgbm=={expected}, found {lgb.__version__}")
    for case, extra in CASES.items():
        generate(case, extra)
        print(f"wrote {FIXTURES / case}")


if __name__ == "__main__":
    main()
