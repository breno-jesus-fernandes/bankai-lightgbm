"""The interleaved benchmark gate: parsing Criterion output and comparing rounds."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "bench_interleaved.py"


def load():
    spec = importlib.util.spec_from_file_location("bench_interleaved", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CRITERION_OUTPUT = """\
Benchmarking histogram/all_rows/16
Benchmarking histogram/all_rows/16: Warming up for 1.0000 s
Benchmarking histogram/all_rows/16: Collecting 20 samples in estimated 3.0073 s (14k iterations)
Benchmarking histogram/all_rows/16: Analyzing
histogram/all_rows/16
                        time:   [14.123 µs 14.456 µs 14.789 µs]
Found 2 outliers among 20 measurements (10.00%)
  1 (5.00%) high mild
prediction/dense/4096x16x20
                        time:   [1.2 ms 1.25 ms 1.3 ms]
"""


def test_parse_times_takes_the_median_estimate_in_nanoseconds() -> None:
    module = load()
    times = module.parse_times(CRITERION_OUTPUT)
    assert times == {
        "histogram/all_rows/16": 14456.0,
        "prediction/dense/4096x16x20": 1_250_000.0,
    }


def test_compare_flags_only_benchmarks_slower_than_the_limit() -> None:
    module = load()
    base = [
        {"fast": 100.0, "slow": 100.0},
        {"fast": 102.0, "slow": 98.0},
        {"fast": 98.0, "slow": 101.0},
    ]
    cand = [
        {"fast": 101.0, "slow": 120.0},
        {"fast": 103.0, "slow": 119.0},
        {"fast": 99.0, "slow": 121.0},
    ]
    ratios, failures = module.compare(base, cand, limit=1.05)
    assert round(ratios["fast"], 2) == 1.01
    assert round(ratios["slow"], 3) == 1.2
    assert failures == ["slow"]


def test_compare_ignores_benchmarks_missing_from_the_base() -> None:
    module = load()
    ratios, failures = module.compare([{"a": 1.0}], [{"a": 1.0, "new": 9.0}], limit=1.05)
    assert "new" not in ratios
    assert failures == []
