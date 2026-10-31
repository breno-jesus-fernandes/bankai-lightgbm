"""Generate upstream binary-dataset fixtures (``save_binary``) for binning/bundling parity.

Each case writes ``tests/fixtures/datasets/<case>/``:
  - ``x.npy`` (float64 features), ``y.npy`` (labels)
  - ``params.json`` (Dataset params + categorical feature indices)
  - ``dataset.bin`` (upstream ``Dataset.save_binary``)

Usage: ``python scripts/gen_dataset_fixtures.py`` (needs ``lightgbm`` at the pinned version).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "datasets"
UPSTREAM_VERSION = (ROOT / "UPSTREAM_VERSION").read_text().strip()

Data = tuple[np.ndarray, np.ndarray]


def dense(rng: np.random.Generator) -> Data:
    x = rng.normal(size=(3000, 8))
    x[:, 1] = np.round(x[:, 1], 1)  # few distinct values
    x[:, 2] = rng.integers(0, 5, size=3000)  # tiny cardinality
    x[:, 3] = rng.exponential(size=3000) * 1e6  # large magnitudes
    x[:, 4] = 7.0  # constant -> trivial
    x[:, 5] = rng.normal(size=3000) * 1e-30  # tiny values around zero threshold
    return x, x[:, 0] + rng.normal(size=3000)


def with_nan(rng: np.random.Generator) -> Data:
    x, y = dense(rng)
    x[rng.random(x.shape) < 0.2] = np.nan
    x[:, 6] = np.nan  # all-NaN column
    return x, y


def sparse(rng: np.random.Generator) -> Data:
    n, d = 4000, 30
    x = np.zeros((n, d))
    # mutually exclusive features -> EFB bundles them
    owner = rng.integers(0, d, size=n)
    x[np.arange(n), owner] = rng.normal(size=n) + 3
    x[:, 0] = rng.normal(size=n)  # one dense column
    x[rng.random((n, d)) < 0.02] = -1.5  # some conflicts
    return x, x[:, 0] + x[:, 1]


def categorical(rng: np.random.Generator) -> Data:
    n = 3000
    x = rng.normal(size=(n, 5))
    x[:, 0] = rng.integers(0, 40, size=n)
    x[:, 1] = rng.choice([0, 1, 2, 3, 1000], size=n, p=[0.5, 0.2, 0.1, 0.1, 0.1])
    x[:, 2] = rng.integers(0, 3, size=n).astype(float)
    x[rng.random(n) < 0.1, 2] = np.nan
    x[rng.random(n) < 0.05, 0] = -1  # negative -> NaN bin
    x[:, 3] = np.where(rng.random(n) < 0.995, 0, rng.integers(1, 300, size=n))  # rare cats
    return x, rng.normal(size=n)


def big(rng: np.random.Generator) -> Data:
    x = rng.normal(size=(30000, 4))
    x[:, 1] = rng.standard_t(2, size=30000)
    return x, x[:, 0]


def skewed(rng: np.random.Generator) -> Data:
    n = 5000
    x = np.column_stack(
        [
            np.where(rng.random(n) < 0.8, 0.0, rng.normal(size=n)),
            np.where(rng.random(n) < 0.5, 1.0, rng.normal(size=n)),  # big-count value
            -np.abs(rng.normal(size=n)),  # all negative
            np.abs(rng.normal(size=n)),  # all positive
            rng.choice([-2.0, 0.0, 2.0], size=n),
        ]
    )
    return x, x.sum(axis=1)


CASES: dict[str, tuple[Callable[[np.random.Generator], Data], dict[str, Any], list[int]]] = {
    "dense_default": (dense, {}, []),
    "dense_maxbin_15": (dense, {"max_bin": 15}, []),
    "dense_min_data_in_bin_50": (dense, {"min_data_in_bin": 50, "min_data_in_leaf": 5}, []),
    "dense_no_prefilter": (dense, {"feature_pre_filter": False}, []),
    "dense_max_bin_by_feature": (
        dense,
        {"max_bin_by_feature": [4, 8, 16, 32, 64, 128, 255, 3]},
        [],
    ),
    "nan_default": (with_nan, {}, []),
    "nan_use_missing_false": (with_nan, {"use_missing": False}, []),
    "nan_zero_as_missing": (with_nan, {"zero_as_missing": True}, []),
    "sparse_bundle": (sparse, {}, []),
    "sparse_no_bundle": (sparse, {"enable_bundle": False}, []),
    "sparse_no_sparse": (sparse, {"is_enable_sparse": False}, []),
    "categorical": (categorical, {}, [0, 1, 2, 3]),
    "categorical_small": (categorical, {"max_cat_to_onehot": 8, "min_data_in_bin": 1}, [0, 1, 2]),
    "big_sampled": (big, {"bin_construct_sample_cnt": 5000, "data_random_seed": 7}, []),
    "skewed": (skewed, {"max_bin": 32}, []),
    "skewed_maxbin_2": (skewed, {"max_bin": 2}, []),
    "skewed_maxbin_3": (skewed, {"max_bin": 3}, []),
}


def main() -> None:
    expected = UPSTREAM_VERSION.lstrip("v")
    if lgb.__version__ != expected:
        raise SystemExit(f"need lightgbm=={expected}, found {lgb.__version__}")
    for case, (make, extra, cats) in CASES.items():
        rng = np.random.default_rng(0)
        x, y = make(rng)
        out = OUT / case
        out.mkdir(parents=True, exist_ok=True)
        params = {"verbose": -1, **extra}
        ds = lgb.Dataset(x, y, params=params, categorical_feature=cats or "auto")
        ds.construct()
        path = out / "dataset.bin"
        path.unlink(missing_ok=True)
        ds.save_binary(str(path))
        np.save(out / "x.npy", x)
        np.save(out / "y.npy", y)
        (out / "params.json").write_text(
            json.dumps({"params": params, "categorical": cats}, indent=1)
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
