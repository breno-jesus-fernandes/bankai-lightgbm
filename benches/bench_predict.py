"""Prediction / model I/O benchmark against upstream lightgbm.

Usage: ``maturin develop -r && python benches/bench_predict.py``.
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
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 20))
    y = x[:, 0] * 2 + np.sin(x[:, 1] * 3) + x[:, 2] * x[:, 3]
    x[rng.random(x.shape) < 0.1] = np.nan
    params = {"objective": "binary", "num_leaves": 63, "verbose": -1}
    upstream = lightgbm.train(params, lightgbm.Dataset(x, (y > 0).astype(float)), 300)
    model_str = upstream.model_to_string()
    ours = bk.Booster(model_str=model_str)
    xt = rng.normal(size=(args.rows, 20))
    xs = xt.copy()
    xs[rng.random(xs.shape) < 0.8] = 0.0
    csr = scipy.sparse.csr_matrix(xs)
    csc = csr.tocsc()

    cases: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
        ("predict", lambda: upstream.predict(xt), lambda: ours.predict(xt)),
        (
            "predict 1 thread",
            lambda: upstream.predict(xt, num_threads=1),
            lambda: ours.predict(xt, num_threads=1),
        ),
        (
            "pred_leaf",
            lambda: upstream.predict(xt, pred_leaf=True),
            lambda: ours.predict(xt, pred_leaf=True),
        ),
        (
            "pred_contrib 5k",
            lambda: upstream.predict(xt[:5000], pred_contrib=True),
            lambda: ours.predict(xt[:5000], pred_contrib=True),
        ),
        ("predict csr", lambda: upstream.predict(csr), lambda: ours.predict(csr)),
        (
            "predict csr 1 thr",
            lambda: upstream.predict(csr, num_threads=1),
            lambda: ours.predict(csr, num_threads=1),
        ),
        ("predict csc", lambda: upstream.predict(csc), lambda: ours.predict(csc)),
        (
            "contrib csr 5k",
            lambda: upstream.predict(csr[:5000], pred_contrib=True),
            lambda: ours.predict(csr[:5000], pred_contrib=True),
        ),
        (
            "load model",
            lambda: lightgbm.Booster(model_str=model_str),
            lambda: bk.Booster(model_str=model_str),
        ),
        ("save model", upstream.model_to_string, ours.model_to_string),
        ("dump_model", upstream.dump_model, ours.dump_model),
    ]
    failed = False
    for name, theirs_f, ours_f in cases:
        tu, to = best_of(theirs_f, args.repeat), best_of(ours_f, args.repeat)
        ratio = tu / to
        flag = "" if ratio >= 1 - args.tolerance else "  <-- SLOWER"
        failed |= bool(flag)
        times = f"upstream {tu * 1e3:9.1f} ms  bankai {to * 1e3:9.1f} ms"
        print(f"{name:18s} {times}  {ratio:5.2f}x{flag}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
