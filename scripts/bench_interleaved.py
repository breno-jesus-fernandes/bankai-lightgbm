#!/usr/bin/env python3
"""Fail when a kernel benchmark is more than 5% slower than the PR's base revision.

Builds the `kernels` Criterion benchmark for the base revision (``BASE_SHA``) and for the
working tree, then runs the two binaries alternately for several rounds in this one job. The
per-benchmark medians are compared, so machine drift between two separate CI jobs cannot
show up as a regression (AGENTS.md section 12: interleaved runs, medians, same session).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

LIMIT = 1.05
ROUNDS = 7
UNIT_NS = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}
NAME = re.compile(r"^[A-Za-z0-9_./-]+$")
# Criterion prints "time:   [low unit mid unit high unit]"; the middle value is the estimate.
TIME = re.compile(r"time:\s+\[\S+ \S+ (\S+) (\S+) \S+ \S+\]")


def parse_times(output: str) -> dict[str, float]:
    """Benchmark name -> median time in nanoseconds, from Criterion's text output."""
    times: dict[str, float] = {}
    name: str | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if NAME.match(stripped) and not line.startswith(" "):
            name = stripped
            continue
        match = TIME.search(line)
        if match and name is not None:
            times[name] = float(match.group(1)) * UNIT_NS[match.group(2)]
    return times


def compare(
    base_runs: list[dict[str, float]], cand_runs: list[dict[str, float]], limit: float
) -> tuple[dict[str, float], list[str]]:
    """Median-over-rounds slowdown per benchmark, and the names above ``limit``."""
    ratios: dict[str, float] = {}
    failures: list[str] = []
    for name in sorted(set(base_runs[0]) & set(cand_runs[0])):
        base = statistics.median(run[name] for run in base_runs if name in run)
        cand = statistics.median(run[name] for run in cand_runs if name in run)
        ratios[name] = cand / base
        if ratios[name] > limit:
            failures.append(name)
    return ratios, failures


def build(source: Path, target: Path) -> Path:
    """Compile the kernels benchmark in ``source`` and return the executable's path."""
    env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
    out = subprocess.run(
        [
            "cargo",
            "bench",
            "--manifest-path",
            str(source / "Cargo.toml"),
            "-p",
            "lgbm-core",
            "--bench",
            "kernels",
            "--no-run",
            "--message-format=json",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.splitlines():
        message = json.loads(line)
        if (
            message.get("reason") == "compiler-artifact"
            and message["target"]["name"] == "kernels"
            and message.get("executable")
        ):
            return Path(message["executable"])
    raise SystemExit("kernels benchmark executable not found")


def run(binary: Path) -> dict[str, float]:
    result = subprocess.run(
        [
            str(binary),
            "--bench",
            "--warm-up-time",
            "1",
            "--measurement-time",
            "3",
            "--sample-size",
            "20",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_times(result.stdout)


def main() -> int:
    base_sha = os.environ["BASE_SHA"]
    if subprocess.run(
        ["git", "cat-file", "-e", f"{base_sha}:Cargo.toml"], capture_output=True
    ).returncode:
        print(f"::warning::base revision {base_sha} has no workspace; 5% benchmark gate skipped")
        return 0
    workdir = Path(tempfile.mkdtemp(prefix="bankai-bench-"))
    base_tree = workdir / "base"
    subprocess.run(["git", "worktree", "add", "--detach", str(base_tree), base_sha], check=True)
    try:
        base_binary = build(base_tree, workdir / "base-target")
        cand_binary = build(Path.cwd(), workdir / "cand-target")
        base_runs: list[dict[str, float]] = []
        cand_runs: list[dict[str, float]] = []
        for _ in range(ROUNDS):
            base_runs.append(run(base_binary))
            cand_runs.append(run(cand_binary))
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(base_tree)], check=False)
        shutil.rmtree(workdir, ignore_errors=True)

    ratios, failures = compare(base_runs, cand_runs, LIMIT)
    for name, ratio in ratios.items():
        rounds = [
            round(cand[name] / base[name], 3)
            for base, cand in zip(base_runs, cand_runs, strict=True)
            if name in base and name in cand
        ]
        print(f"{name}: {ratio:.3f}x vs base (median of {ROUNDS} rounds; per round {rounds})")
    if failures:
        print(f"Benchmark regressions exceed the {int((LIMIT - 1) * 100)}% limit:")
        for name in failures:
            print(f"  {name}: {ratios[name]:.3f}x")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
