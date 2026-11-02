"""``Dataset._dump_text`` matches upstream byte for byte.

upstream: src/io/dataset.cpp (v4.7.0) Dataset::DumpTextFile. The dump lists the bin index of every
feature in every row, so it covers the bin mappers, bundling (``num_groups``) and missing values.
Both libraries construct from the same data with the same parameters and all rows are sampled, so
the bins are deterministic and the files must be identical.
"""

from __future__ import annotations

import lightgbm
import numpy as np
import pytest
import scipy.sparse

import bankai_lightgbm as lgb


def _dense(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 4))
    x[rng.random(size=x.shape) < 0.2] = np.nan
    x[:, 3] = np.where(rng.random(60) < 0.8, 0.0, rng.normal(size=60))
    return x


def _mutually_exclusive(seed: int) -> scipy.sparse.csr_matrix:
    # one-hot blocks: features never fire together, so upstream bundles them (EFB)
    rng = np.random.default_rng(seed)
    rows = np.arange(80)
    cols = rng.integers(0, 6, size=80)
    dense = np.zeros((80, 8))
    dense[rows, cols] = rng.normal(size=80) + 2.0
    dense[:, 6] = (rng.random(80) < 0.1) * rng.normal(size=80)
    return scipy.sparse.csr_matrix(dense)


def _dump(library, data, tmp_path, name: str, **params) -> str:
    base = {"verbosity": -1, "min_data_in_leaf": 1, "min_data_in_bin": 1, "max_bin": 15}
    dataset = library.Dataset(data, label=np.zeros(data.shape[0]), params={**base, **params})
    path = tmp_path / name
    dataset.construct()
    dataset._dump_text(path)
    return path.read_text()


def _with_constant_column(seed: int) -> np.ndarray:
    # a constant feature has one bin, so upstream does not use it (an NA column in the dump)
    x = _dense(seed)
    x[:, 1] = 7.0
    return x


def _wide_sparse(seed: int) -> scipy.sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = np.zeros((120, 40))
    for col in range(40):
        rows = rng.choice(120, size=int(rng.integers(1, 6)), replace=False)
        dense[rows, col] = rng.normal(size=rows.shape[0]) + 3.0
    return scipy.sparse.csr_matrix(dense)


@pytest.mark.parametrize(
    "make_data,params",
    [
        pytest.param(lambda: _dense(1), {}, id="dense-with-nan"),
        pytest.param(lambda: _dense(2), {"max_bin": 255}, id="dense-more-bins"),
        pytest.param(lambda: _mutually_exclusive(3), {}, id="sparse-bundled"),
        pytest.param(lambda: _with_constant_column(4), {}, id="unused-constant-column"),
        pytest.param(lambda: _wide_sparse(5), {"max_bin": 4}, id="wide-sparse-bundled"),
    ],
)
def test_dump_text_matches_upstream_bytes(tmp_path, make_data, params) -> None:
    data = make_data()
    want = _dump(lightgbm, data, tmp_path, "upstream.txt", **params)
    got = _dump(lgb, data, tmp_path, "bankai.txt", **params)
    assert got == want
