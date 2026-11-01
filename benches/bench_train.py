"""Training benchmark against upstream lightgbm (col-wise, incl. a multi-value group).

Usage: ``maturin develop -r && python benches/bench_train.py``.
Exits non-zero if bankai is slower than upstream by more than ``--tolerance``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

import lightgbm
import numpy as np
import scipy.sparse

import bankai_lightgbm as bk


def best_of(f: Callable[[], Any], n: int) -> float:
    f()
    best = float("inf")
    for _ in range(n):
        start = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    dense = rng.normal(size=(50_000, 20))
    y_dense = dense[:, 0] + np.sin(dense[:, 1] * 3) + dense[:, 2] * dense[:, 3]
    sparse = rng.normal(size=(50_000, 100))
    sparse[rng.random(sparse.shape) < 0.9] = 0.0  # EFB builds a multi-value group
    y_sparse = sparse[:, :10].sum(axis=1) + rng.normal(size=50_000) * 0.1
    csr = scipy.sparse.csr_matrix(sparse)

    cases: list[tuple[str, Any, np.ndarray, int]] = []
    for threads in (1, 2):
        cases.append((f"dense 50k x 20, {threads} thr", dense, y_dense, threads))
        cases.append((f"multi-val 50k x 100, {threads} thr", sparse, y_sparse, threads))
        cases.append((f"multi-val csr, {threads} thr", csr, y_sparse, threads))

    failed = False
    for name, data, y, threads in cases:
        params = {"verbose": -1, "num_threads": threads, "force_col_wise": True}

        def upstream(data: Any = data, y: np.ndarray = y, params: dict[str, Any] = params) -> Any:
            return lightgbm.train(params, lightgbm.Dataset(data, y), args.rounds)

        def ours(data: Any = data, y: np.ndarray = y, params: dict[str, Any] = params) -> Any:
            return bk.train(params, bk.Dataset(data, y), args.rounds)

        tu, to = best_of(upstream, args.repeat), best_of(ours, args.repeat)
        ratio = tu / to
        flag = "" if ratio >= 1 - args.tolerance else "  <-- SLOWER"
        failed |= bool(flag)
        times = f"upstream {tu * 1e3:8.1f} ms  bankai {to * 1e3:8.1f} ms"
        print(f"{name:30s} {times}  {ratio:5.2f}x{flag}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
