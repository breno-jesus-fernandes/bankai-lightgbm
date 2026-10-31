"""Generate upstream *training* fixtures: models trained deterministically on one thread.

Each case writes ``tests/fixtures/train/<case>/``:
  - ``x.npy`` / ``y.npy`` (float64), ``params.json`` (``params`` + ``num_boost_round``)
  - ``model.txt`` (upstream ``model_to_string()`` after ``lgb.train``)

bankai must reproduce ``model.txt`` byte-for-byte (see AGENTS.md, parity tolerances).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "train"
UPSTREAM_VERSION = (ROOT / "UPSTREAM_VERSION").read_text().strip()

# Pin everything that would otherwise depend on timing or thread count.
BASE: dict[str, Any] = {
    "deterministic": True,
    "num_threads": 1,
    "force_col_wise": True,
    "seed": 7,
    "verbose": -1,
}

Data = tuple[np.ndarray, np.ndarray]


def reg(rng: np.random.Generator) -> Data:
    x = rng.normal(size=(2000, 6))
    x[:, 3] = np.round(x[:, 3], 1)
    y = 2 * x[:, 0] + np.sin(3 * x[:, 1]) + x[:, 2] * x[:, 3] + rng.normal(scale=0.2, size=2000)
    return x, y


def reg_nan(rng: np.random.Generator) -> Data:
    x, y = reg(rng)
    x[rng.random(x.shape) < 0.15] = np.nan
    return x, y


def reg_zero(rng: np.random.Generator) -> Data:
    x, y = reg(rng)
    x[rng.random(x.shape) < 0.4] = 0.0
    return x, y


CASES: dict[str, tuple[Callable[[np.random.Generator], Data], dict[str, Any], int]] = {
    "l2_default": (reg, {"objective": "regression"}, 20),
    "l2_small": (
        reg,
        {"objective": "regression", "num_leaves": 7, "min_data_in_leaf": 5, "learning_rate": 0.3},
        15,
    ),
    "l2_nan": (reg_nan, {"objective": "regression", "num_leaves": 15}, 15),
    "l2_zero": (reg_zero, {"objective": "regression", "num_leaves": 15}, 15),
    "l2_zero_as_missing": (
        reg_zero,
        {"objective": "regression", "num_leaves": 15, "zero_as_missing": True},
        15,
    ),
    "l2_regularized": (
        reg,
        {
            "objective": "regression",
            "lambda_l1": 0.5,
            "lambda_l2": 2.0,
            "min_gain_to_split": 0.01,
            "min_sum_hessian_in_leaf": 5.0,
        },
        15,
    ),
    "l2_max_delta_smooth": (
        reg,
        {"objective": "regression", "max_delta_step": 0.5, "path_smooth": 3.0},
        15,
    ),
    "l2_max_depth": (reg, {"objective": "regression", "max_depth": 3}, 15),
    "l2_no_boost_from_average": (reg, {"objective": "regression", "boost_from_average": False}, 10),
}


def main() -> None:
    expected = UPSTREAM_VERSION.lstrip("v")
    if lgb.__version__ != expected:
        raise SystemExit(f"need lightgbm=={expected}, found {lgb.__version__}")
    for case, (make, extra, rounds) in CASES.items():
        rng = np.random.default_rng(0)
        x, y = make(rng)
        params = {**BASE, **extra}
        booster = lgb.train(params, lgb.Dataset(x, y), num_boost_round=rounds)
        out = OUT / case
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "x.npy", x)
        np.save(out / "y.npy", y)
        (out / "params.json").write_text(
            json.dumps({"params": params, "num_boost_round": rounds}, indent=1)
        )
        (out / "model.txt").write_text(booster.model_to_string())
        print(f"wrote {out} ({booster.num_trees()} trees)")


if __name__ == "__main__":
    main()
