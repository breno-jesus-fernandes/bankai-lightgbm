"""Shared helpers for parity tests against upstream LightGBM fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def fixture_cases() -> list[str]:
    return sorted(p.name for p in FIXTURES.iterdir() if (p / "model.txt").exists())


@pytest.fixture(
    params=fixture_cases() or [pytest.param(None, marks=pytest.mark.skip("no fixtures"))]
)
def case_dir(request: pytest.FixtureRequest) -> Path:
    return FIXTURES / request.param
