#!/usr/bin/env python3
"""Fail when a Criterion benchmark is more than 5% slower than its named baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def point_estimate(path: Path) -> float:
    with path.open(encoding="utf-8") as file:
        return float(json.load(file)["mean"]["point_estimate"])


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("target/criterion")
    candidate = sys.argv[2] if len(sys.argv) > 2 else "candidate"
    baseline = sys.argv[3] if len(sys.argv) > 3 else "main"
    regressions: list[tuple[Path, float]] = []
    compared = 0

    for current_path in root.glob(f"**/{candidate}/estimates.json"):
        baseline_path = current_path.parent.parent / baseline / "estimates.json"
        if not baseline_path.exists():
            continue
        previous = point_estimate(baseline_path)
        current = point_estimate(current_path)
        ratio = current / previous if previous > 0.0 else 1.0
        compared += 1
        print(f"{current_path.parent.parent.relative_to(root)}: {ratio:.3f}x")
        if ratio > 1.05:
            regressions.append((current_path, ratio))

    if compared == 0:
        print(f"no shared Criterion results found for baseline {baseline!r}", file=sys.stderr)
        return 2
    if regressions:
        print("Criterion regressions exceed the 5% limit:", file=sys.stderr)
        for path, ratio in regressions:
            print(f"  {path.parent.parent.relative_to(root)}: {ratio:.3f}x", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
