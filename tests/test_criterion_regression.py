from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_criterion_regression.py"


def write_estimate(path: Path, nanoseconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": {"point_estimate": nanoseconds}}), encoding="utf-8")


def run_gate(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), "candidate", "main"],
        capture_output=True,
        check=False,
        text=True,
    )


def test_criterion_gate_accepts_regression_at_five_percent(tmp_path: Path) -> None:
    bench = tmp_path / "histogram" / "dense"
    write_estimate(bench / "main" / "estimates.json", 100.0)
    write_estimate(bench / "candidate" / "estimates.json", 105.0)

    result = run_gate(tmp_path)

    assert result.returncode == 0


def test_criterion_gate_rejects_regression_over_five_percent(tmp_path: Path) -> None:
    bench = tmp_path / "histogram" / "dense"
    write_estimate(bench / "main" / "estimates.json", 100.0)
    write_estimate(bench / "candidate" / "estimates.json", 105.1)

    result = run_gate(tmp_path)

    assert result.returncode == 1
    assert "1.051x" in result.stderr


def test_criterion_gate_skips_benchmarks_without_a_base_measurement(
    tmp_path: Path,
) -> None:
    write_estimate(tmp_path / "partition" / "new_case" / "candidate" / "estimates.json", 200.0)
    bench = tmp_path / "histogram" / "existing"
    write_estimate(bench / "main" / "estimates.json", 100.0)
    write_estimate(bench / "candidate" / "estimates.json", 100.0)

    result = run_gate(tmp_path)

    assert result.returncode == 0
    assert "no shared" not in result.stderr
