"""Shared helpers for parity tests against upstream LightGBM fixtures."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def fixture_cases() -> list[str]:
    return sorted(p.name for p in FIXTURES.iterdir() if (p / "model.txt").exists())


def on_fixture_platform() -> bool:
    """Bit-exact parity is only required on the platform that generated the fixtures."""
    try:
        plat = json.loads((FIXTURES / "platform.json").read_text())
    except FileNotFoundError:
        return False
    return bool(
        plat["system"] == platform.system().lower()
        and plat["machine"] == platform.machine().lower()
    )


def assert_parity(got: np.ndarray, want: np.ndarray) -> None:
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    if on_fixture_platform():
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)


@pytest.fixture(
    params=fixture_cases() or [pytest.param(None, marks=pytest.mark.skip("no fixtures"))]
)
def case_dir(request: pytest.FixtureRequest) -> Path:
    return FIXTURES / request.param
