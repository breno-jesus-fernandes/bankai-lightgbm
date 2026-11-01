"""Dataset construction benchmark against upstream lightgbm (dense, CSR and CSC input).

Usage: ``maturin develop -r && python benches/bench_dataset.py``.
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
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    params = {"verbose": -1, "num_threads": args.threads}
    shapes = [(200_000, 500, 0.01), (50_000, 5_000, 0.002), (100_000, 50, 0.3)]
    cases: list[tuple[str, Any, np.ndarray]] = []
    for nrow, ncol, density in shapes:
        csr = scipy.sparse.random(
            nrow, ncol, density=density, format="csr", random_state=1, data_rvs=rng.standard_normal
        )
        y = rng.normal(size=nrow)
        tag = f"{nrow // 1000}k x {ncol} @{density}"
        cases.append((f"csr {tag}", csr, y))
        cases.append((f"csc {tag}", csr.tocsc(), y))
        if nrow * ncol <= 20_000_000:
            cases.append((f"dense {tag}", csr.toarray(), y))

    failed = False
    for name, data, y in cases:

        def upstream(data: Any = data, y: np.ndarray = y) -> Any:
            return lightgbm.Dataset(data, y, params=params).construct()

        def ours(data: Any = data, y: np.ndarray = y) -> Any:
            return bk.Dataset(data, y, params=params).construct()

        tu, to = best_of(upstream, args.repeat), best_of(ours, args.repeat)
        ratio = tu / to
        flag = "" if ratio >= 1 - args.tolerance else "  <-- SLOWER"
        failed |= bool(flag)
        times = f"upstream {tu * 1e3:8.1f} ms  bankai {to * 1e3:8.1f} ms"
        print(f"{name:28s} {times}  {ratio:5.2f}x{flag}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
