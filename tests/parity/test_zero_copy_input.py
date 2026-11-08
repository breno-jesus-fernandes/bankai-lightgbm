"""Dense numpy input is handed to the native layer without a Python-side float64 copy.

upstream: python-package/lightgbm/basic.py (v4.7.0) ``__init_from_np2d`` passes float32 and
float64 matrices through (``np.ascontiguousarray`` only for the memory layout). AGENTS.md
section 7 asks for zero-copy where the layout allows it.
"""

from __future__ import annotations

import tracemalloc

import lightgbm as up
import numpy as np
import pytest

import bankai_lightgbm as bk

ROWS, COLS = 200_000, 10


def _matrix(dtype: type, order: str) -> np.ndarray:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(ROWS, COLS)).astype(dtype)
    return np.asfortranarray(x) if order == "F" else np.ascontiguousarray(x)


def _construct_peak_bytes(matrix: np.ndarray) -> int:
    label = np.arange(ROWS, dtype=np.float64) % 3
    tracemalloc.start()
    try:
        bk.Dataset(matrix, label, params={"verbose": -1}).construct()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_float32_c_contiguous_input_has_no_python_float64_copy(dtype: type) -> None:
    matrix = _matrix(dtype, "C")
    peak = _construct_peak_bytes(matrix)
    one_float64_copy = ROWS * COLS * 8
    # Native code allocates its own buffer, which tracemalloc does not see. Python must not add
    # a float64 copy of the matrix, so the traced peak stays well under one copy.
    assert peak < one_float64_copy // 2, f"peak {peak} bytes is at least half a float64 copy"


def test_column_major_input_has_no_python_copy() -> None:
    matrix = _matrix(np.float64, "F")
    peak = _construct_peak_bytes(matrix)
    one_float64_copy = ROWS * COLS * 8
    assert peak < one_float64_copy // 2, f"peak {peak} bytes is at least half a float64 copy"


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_layout_and_dtype_do_not_change_the_model(dtype: type, order: str) -> None:
    """Same values in any layout or float width train the same model as upstream's float64."""
    matrix = _matrix(dtype, order)
    label = np.sin(matrix[:, 0].astype(np.float64)) + matrix[:, 1].astype(np.float64)
    params = {
        "objective": "regression",
        "num_leaves": 15,
        "verbose": -1,
        "num_threads": 1,
        "deterministic": True,
        "seed": 1,
    }
    reference = matrix.astype(np.float64)
    want = up.train(params, up.Dataset(reference, label), num_boost_round=5)
    got = bk.train(params, bk.Dataset(matrix, label), num_boost_round=5)
    np.testing.assert_array_equal(got.predict(reference), want.predict(reference))
