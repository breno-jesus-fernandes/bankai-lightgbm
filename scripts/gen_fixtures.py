"""Generate golden fixtures with upstream LightGBM.

Each case writes, under ``tests/fixtures/<case>/``:
  - ``train.npy`` / ``test.npy``  (float64, label in column 0)
  - ``model.txt`` and ``model.json`` (upstream ``save_model`` / ``dump_model``)
  - ``pred.npy``, ``pred_raw.npy``, ``pred_leaf.npy``, ``pred_contrib.npy``
  - ``params.json``

Usage: ``python scripts/gen_fixtures.py`` (needs ``lightgbm`` at the pinned version).
"""

from __future__ import annotations

import json
import platform
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

# case name -> (data kind, extra params)
CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "regression_l2": ("regression", {"objective": "regression", "num_leaves": 15}),
    "binary": ("binary", {"objective": "binary", "num_leaves": 15}),
    "multiclass": ("multiclass", {"objective": "multiclass", "num_class": 3, "num_leaves": 7}),
    "multiclassova": (
        "multiclass",
        {"objective": "multiclassova", "num_class": 3, "num_leaves": 7},
    ),
    "categorical": ("categorical", {"objective": "regression", "num_leaves": 15}),
    "missing": ("missing", {"objective": "regression", "num_leaves": 15}),
    "zero_as_missing": (
        "sparse",
        {"objective": "regression", "num_leaves": 15, "zero_as_missing": True},
    ),
    "lambdarank": ("lambdarank", {"objective": "lambdarank", "num_leaves": 15}),
    "rank_xendcg": ("lambdarank", {"objective": "rank_xendcg", "num_leaves": 15}),
    "regression_sqrt": ("regression", {"objective": "regression", "reg_sqrt": True}),
    "huber": ("regression", {"objective": "huber", "alpha": 0.9}),
    "l1": ("regression", {"objective": "l1"}),
    "quantile": ("regression", {"objective": "quantile", "alpha": 0.7}),
    "poisson": ("positive", {"objective": "poisson"}),
    "gamma": ("positive", {"objective": "gamma"}),
    "tweedie": ("positive", {"objective": "tweedie", "tweedie_variance_power": 1.3}),
    "xentropy": ("probability", {"objective": "cross_entropy"}),
    "xentlambda": ("probability", {"objective": "cross_entropy_lambda"}),
    "linear_tree": ("regression", {"objective": "regression", "linear_tree": True}),
    "linear_tree_missing": ("missing", {"objective": "regression", "linear_tree": True}),
    "dart": ("binary", {"objective": "binary", "boosting": "dart", "drop_rate": 0.3}),
    "rf": (
        "binary",
        {"objective": "binary", "boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1},
    ),
    "goss": ("regression", {"objective": "regression", "data_sample_strategy": "goss"}),
    "monotone": (
        "regression",
        {"objective": "regression", "monotone_constraints": [1, 0, -1, 0, 0, 0, 0, 0, 0, 0]},
    ),
}


def make_data(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, d = 2000, 10
    x = rng.normal(size=(n, d))
    signal = x[:, 0] * 2 + np.sin(x[:, 1] * 3) + x[:, 2] * x[:, 3]
    if kind == "categorical":
        x[:, 4] = rng.integers(0, 20, size=n)
        signal += np.where(x[:, 4] % 3 == 0, 1.5, -0.5)
    if kind == "missing":
        x[rng.random(size=(n, d)) < 0.15] = np.nan
        signal = np.nan_to_num(signal)
    if kind == "sparse":
        x[rng.random(size=(n, d)) < 0.6] = 0.0
        signal = x[:, 0] * 2 + np.sin(x[:, 1] * 3) + x[:, 2] * x[:, 3]
    if kind == "binary":
        return x, (signal + rng.normal(scale=0.5, size=n) > 0).astype(np.float64)
    if kind == "multiclass":
        return x, np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3])).astype(np.float64)
    if kind == "lambdarank":
        cuts = np.quantile(signal, [0.5, 0.8, 0.95])
        return x, np.clip(np.digitize(signal, cuts), 0, 3).astype(np.float64)
    if kind == "positive":
        return x, rng.poisson(np.exp(0.3 * signal)).astype(np.float64) + 0.1
    if kind == "probability":
        return x, 1.0 / (1.0 + np.exp(-signal))
    return x, signal + rng.normal(scale=0.1, size=n)


def generate(case: str, kind: str, extra: dict[str, Any]) -> None:
    rng = np.random.default_rng(0)
    x, y = make_data(kind, rng)
    split = int(len(y) * 0.8)
    out = FIXTURES / case
    out.mkdir(parents=True, exist_ok=True)

    params = {**BASE, **extra}
    ds_kwargs: dict[str, Any] = {}
    if kind == "categorical":
        ds_kwargs["categorical_feature"] = [4]
    if kind == "lambdarank":
        ds_kwargs["group"] = [20] * (split // 20)

    train = lgb.Dataset(x[:split], y[:split], **ds_kwargs)
    booster = lgb.train(params, train, num_boost_round=20)

    x_test = x[split:]
    np.save(out / "train.npy", np.column_stack([y[:split], x[:split]]))
    np.save(out / "test.npy", np.column_stack([y[split:], x_test]))
    booster.save_model(str(out / "model.txt"))
    (out / "model.json").write_text(json.dumps(booster.dump_model(), indent=1))
    np.save(out / "pred.npy", booster.predict(x_test))
    np.save(out / "pred_raw.npy", booster.predict(x_test, raw_score=True))
    np.save(out / "pred_leaf.npy", booster.predict(x_test, pred_leaf=True))
    if not extra.get("linear_tree"):
        np.save(out / "pred_contrib.npy", booster.predict(x_test, pred_contrib=True))
    np.save(out / "pred_iter_3_5.npy", booster.predict(x_test, start_iteration=3, num_iteration=5))
    (out / "params.json").write_text(
        json.dumps({"params": params, "dataset": ds_kwargs, "num_boost_round": 20}, indent=1)
    )


def main() -> None:
    expected = UPSTREAM_VERSION.lstrip("v")
    if lgb.__version__ != expected:
        raise SystemExit(f"need lightgbm=={expected}, found {lgb.__version__}")
    # Bit-exact parity only holds on the platform that produced the fixtures: upstream's
    # floating-point rounding depends on its compiler's vectorization and FMA choices.
    plat = {"system": platform.system().lower(), "machine": platform.machine().lower()}
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "platform.json").write_text(json.dumps(plat) + "\n")
    for case, (kind, extra) in CASES.items():
        generate(case, kind, extra)
        print(f"wrote {FIXTURES / case}")


if __name__ == "__main__":
    main()
