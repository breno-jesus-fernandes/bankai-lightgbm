"""Interleaved training benchmark against upstream lightgbm (AGENTS.md section 12).

bankai and upstream train the same data with the same params in one process, alternating
A, B, A, B, ... over several rounds. The median of each side is reported, so machine drift
within the session cannot make one side look faster.

Usage: ``maturin develop -r && python benches/bench_train_interleaved.py``.
Exits non-zero when bankai's median is slower than upstream's by more than ``--tolerance``
on any case. Build the release wheel first; a debug build is not comparable.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import lightgbm
import numpy as np
import scipy.sparse

import bankai_lightgbm as bk


def _dense(rng: np.random.Generator) -> tuple[Any, np.ndarray]:
    x = rng.normal(size=(50_000, 20))
    y = x[:, 0] * 2 + np.sin(x[:, 1]) + rng.normal(scale=0.1, size=50_000)
    return x, y


def _sparse(rng: np.random.Generator) -> tuple[Any, np.ndarray]:
    x = scipy.sparse.random(50_000, 200, density=0.03, format="csr", random_state=rng)
    y = np.asarray(x[:, 0].todense()).ravel() * 3 + rng.normal(scale=0.1, size=50_000)
    return x, y


CASES: dict[str, Callable[[np.random.Generator], tuple[Any, np.ndarray]]] = {
    "dense": _dense,
    "sparse": _sparse,
}


def _train(mod: Any, x: Any, y: np.ndarray, params: dict[str, Any], rounds: int) -> float:
    start = time.perf_counter()
    mod.train(params, mod.Dataset(x, y, params={"verbose": -1}), num_boost_round=rounds)
    return time.perf_counter() - start


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7, help="interleaved rounds (at least 7)")
    ap.add_argument("--boost", type=int, default=50, help="boosting rounds per training")
    ap.add_argument("--tolerance", type=float, default=0.03, help="allowed slowdown (noise)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--layout", choices=["auto", "col", "row"], default="auto")
    args = ap.parse_args()

    layout = {"auto": {}, "col": {"force_col_wise": True}, "row": {"force_row_wise": True}}
    failed = False
    for name, make in CASES.items():
        x, y = make(np.random.default_rng(0))
        for threads in args.threads:
            params = {
                "objective": "regression",
                "num_leaves": 31,
                "verbose": -1,
                "num_threads": threads,
                **layout[args.layout],
            }
            # One untimed warm-up per side, then alternate.
            _train(lightgbm, x, y, params, 2)
            _train(bk, x, y, params, 2)
            ours: list[float] = []
            theirs: list[float] = []
            for _ in range(args.rounds):
                theirs.append(_train(lightgbm, x, y, params, args.boost))
                ours.append(_train(bk, x, y, params, args.boost))
            ratio = statistics.median(ours) / statistics.median(theirs)
            verdict = "ok" if ratio <= 1 + args.tolerance else "SLOWER"
            failed |= verdict == "SLOWER"
            print(
                f"{name:6s} threads={threads} layout={args.layout:4s} "
                f"bankai={statistics.median(ours):.3f}s upstream={statistics.median(theirs):.3f}s "
                f"ratio={ratio:.3f} {verdict}",
                flush=True,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
