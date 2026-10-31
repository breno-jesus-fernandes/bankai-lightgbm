"""Generate upstream *metric* fixtures: metric values at exact, arbitrary scores.

Each case writes ``tests/fixtures/metrics/<case>/``:
  - ``label.npy`` / ``weight.npy`` (optional) / ``group.npy`` (optional) / ``score.npy``
    (raw scores, class-major for multiclass)
  - ``params.json`` (``params`` incl. ``objective`` and ``metric``) and ``expected.json``

The expected values come from ``Booster.eval_train()`` *before* the first iteration, so the
internal score equals the ``init_score`` we provide, bit for bit, and no tree noise leaks
into the comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "metrics"
UPSTREAM_VERSION = (ROOT / "UPSTREAM_VERSION").read_text().strip()
N = 600
BASE: dict[str, Any] = {"num_threads": 1, "verbose": -1, "seed": 3}


def make_case(
    name: str,
    params: dict[str, Any],
    label: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray | None = None,
    group: np.ndarray | None = None,
) -> None:
    params = {**BASE, **params}
    rng = np.random.default_rng(0)
    x = rng.normal(size=(len(label), 2))
    # lightgbm flattens (n, k) init scores column-major, i.e. class-major.
    ds = lgb.Dataset(x, label, weight=weight, group=group, init_score=score, params=params)
    booster = lgb.Booster(params, ds)
    results = booster.eval_train()
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "label.npy", label)
    flat = score.T.ravel() if score.ndim == 2 else score
    np.save(out / "score.npy", flat)
    if weight is not None:
        np.save(out / "weight.npy", weight.astype(np.float32))
    if group is not None:
        np.save(out / "group.npy", group.astype(np.int32))
    (out / "params.json").write_text(json.dumps(params, indent=1))
    expected = [[str(r[1]), float(r[2])] for r in results]
    (out / "expected.json").write_text(json.dumps(expected, indent=1))
    print(f"wrote {out}: {[m for m, _ in expected]}")


def main() -> None:
    expected = UPSTREAM_VERSION.lstrip("v")
    if lgb.__version__ != expected:
        raise SystemExit(f"need lightgbm=={expected}, found {lgb.__version__}")
    rng = np.random.default_rng(42)
    w = rng.uniform(0.2, 2.0, size=N)

    # Regression family.
    y = rng.normal(2.0, 3.0, size=N)
    s = y + rng.normal(0, 1.5, size=N)
    reg = ["l2", "rmse", "l1", "huber", "fair", "quantile", "mape", "r2"]
    extra = {"alpha": 0.7, "fair_c": 1.3}
    make_case("reg", {"objective": "regression", "metric": reg, **extra}, y, s)
    make_case("reg_weighted", {"objective": "regression", "metric": reg, **extra}, y, s, w)

    # Positive-label losses (exp link).
    yp = rng.gamma(2.0, 1.5, size=N)
    sp = np.log(yp) + rng.normal(0, 0.4, size=N)
    for obj, metrics in [
        ("poisson", ["poisson"]),
        ("gamma", ["gamma", "gamma_deviance"]),
        ("tweedie", ["tweedie"]),
    ]:
        p = {"objective": obj, "metric": metrics, "tweedie_variance_power": 1.4}
        make_case(obj, p, yp, sp)
        make_case(f"{obj}_weighted", p, yp, sp, w)

    # Binary, with tied scores to exercise the AUC bucket logic.
    yb = (rng.random(N) < 0.35).astype(np.float64)
    sb = np.round(rng.normal(0, 2, size=N) + 1.5 * (yb - 0.5), 1)
    bin_metrics = ["binary_logloss", "binary_error", "auc", "average_precision"]
    make_case("binary", {"objective": "binary", "metric": bin_metrics}, yb, sb)
    make_case("binary_weighted", {"objective": "binary", "metric": bin_metrics}, yb, sb, w)
    make_case(
        "binary_sigmoid",
        {"objective": "binary", "sigmoid": 0.6, "metric": bin_metrics},
        yb,
        sb,
        w,
    )

    # Multiclass.
    k = 4
    ym = rng.integers(0, k, size=N).astype(np.float64)
    sm = rng.normal(0, 1.2, size=(N, k))
    sm[np.arange(N), ym.astype(int)] += 0.8
    mc = ["multi_logloss", "multi_error", "auc_mu"]
    for obj in ["multiclass", "multiclassova"]:
        p = {"objective": obj, "num_class": k, "metric": mc}
        make_case(obj, p, ym, sm)
        make_case(f"{obj}_weighted", p, ym, sm, w)
    make_case(
        "multiclass_top2",
        {"objective": "multiclass", "num_class": k, "multi_error_top_k": 2, "metric": mc},
        ym,
        sm,
    )
    make_case(
        "multiclass_aucmu_weights",
        {
            "objective": "multiclass",
            "num_class": k,
            "auc_mu_weights": [0, 1, 2, 3, 1, 0, 1, 2, 2, 1, 0, 1, 3, 2, 1, 0],
            "metric": ["auc_mu"],
        },
        ym,
        sm,
    )

    # Cross entropy.
    yx = rng.random(N)
    sx = rng.normal(0, 1.5, size=N) + 2 * (yx - 0.5)
    make_case(
        "xentropy",
        {"objective": "cross_entropy", "metric": ["cross_entropy", "kullback_leibler"]},
        yx,
        sx,
    )
    make_case(
        "xentropy_weighted",
        {"objective": "cross_entropy", "metric": ["cross_entropy", "kullback_leibler"]},
        yx,
        sx,
        w,
    )
    make_case(
        "xentlambda",
        {"objective": "cross_entropy_lambda", "metric": ["cross_entropy_lambda"]},
        yx,
        sx,
    )
    make_case(
        "xentlambda_weighted",
        {"objective": "cross_entropy_lambda", "metric": ["cross_entropy_lambda"]},
        yx,
        sx,
        w,
    )

    # Ranking: 60 queries of 10 rows, graded relevance 0..4.
    group = np.full(60, 10)
    yr = rng.integers(0, 5, size=N).astype(np.float64)
    sr = yr + rng.normal(0, 1.5, size=N)
    sr = np.round(sr, 1)
    rank = ["ndcg", "map"]
    rp = {"objective": "lambdarank", "metric": rank, "eval_at": [1, 3, 5, 10]}
    make_case("rank", rp, yr, sr, group=group)
    # Per-row weights become per-query weights (their mean over each query).
    make_case("rank_weighted", rp, yr, sr, w, group)
    make_case(
        "rank_label_gain",
        {**rp, "label_gain": [0, 1, 3, 4, 10], "eval_at": [2, 4]},
        yr,
        sr,
        group=group,
    )


if __name__ == "__main__":
    main()
