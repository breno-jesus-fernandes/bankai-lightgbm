"""An upstream binary file with sparse feature groups loads and trains like upstream.

upstream: src/io/sparse_bin.hpp (v4.7.0) ``SparseBin::SaveBinaryToFile`` and ``LoadFromMemory``;
src/io/dataset_loader.cpp (v4.7.0) ``LoadFromBinFile``. A group is stored sparse when most of
its rows are the default bin, which is the case for the matrix below.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm
import numpy as np
import pytest
import scipy.sparse

import bankai_lightgbm as lgb

PARAMS = {
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 11,
}


def _sparse_matrix(rows: int, cols: int, density: float, seed: int) -> scipy.sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    matrix = scipy.sparse.random(rows, cols, density=density, format="csr", random_state=rng)
    return matrix


@pytest.mark.parametrize("density", [0.03, 0.10])
def test_upstream_sparse_binary_file_trains_like_upstream(tmp_path: Path, density: float) -> None:
    x = _sparse_matrix(3000, 40, density, seed=5)
    rng = np.random.default_rng(6)
    y = np.asarray(x[:, 0].todense()).ravel() * 3 + rng.normal(scale=0.1, size=3000)
    path = tmp_path / "sparse.bin"
    lightgbm.Dataset(x, y, params={"verbose": -1}).save_binary(str(path))

    upstream = lightgbm.train(PARAMS, lightgbm.Dataset(str(path)), num_boost_round=10)
    ours = lgb.train(PARAMS, lgb.Dataset(str(path)), num_boost_round=10)

    assert ours.model_to_string() == upstream.model_to_string()


def test_upstream_sparse_binary_file_predicts_like_upstream(tmp_path: Path) -> None:
    x = _sparse_matrix(2000, 30, 0.05, seed=9)
    y = np.asarray(x[:, 1].todense()).ravel() * 2.0
    path = tmp_path / "sparse.bin"
    lightgbm.Dataset(x, y, params={"verbose": -1}).save_binary(str(path))
    upstream = lightgbm.train(PARAMS, lightgbm.Dataset(str(path)), num_boost_round=8)
    ours = lgb.train(PARAMS, lgb.Dataset(str(path)), num_boost_round=8)
    np.testing.assert_array_equal(ours.predict(x), upstream.predict(x))
