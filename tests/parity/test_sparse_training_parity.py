"""P2/P6 acceptance: training from scipy.sparse CSR/CSC matches upstream byte for byte.

Upstream builds CSR/CSC datasets row by row (``LGBM_DatasetCreateFromCSR``/``FromCSC``):
stored pairs are pushed in storage order, then features whose default bin is not their most
frequent bin get an explicit zero (``Dataset::FinishOneRow``). These tests train both
libraries on the same sparse matrices on this machine and compare model text exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pytest
import scipy.sparse

import bankai_lightgbm as lgb

BASE: dict[str, Any] = {
    "deterministic": True,
    "num_threads": 1,
    "force_col_wise": True,
    "seed": 7,
    "verbose": -1,
    "num_leaves": 15,
}

# Exclusive Feature Bundling turns sparse columns into multi-value feature groups, which the
# learner cannot train on yet (ROADMAP P2 `MultiValBin`). Training parity tests therefore
# keep one feature per group; dataset parity (binary files) covers the bundled layouts.
TRAINABLE: dict[str, Any] = {"enable_bundle": False}


def _data(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sparse-ish features, including ones where 0 is *not* the most frequent value."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 10))
    x[rng.random(x.shape) < 0.7] = 0.0
    x[rng.random(x.shape) < 0.03] = np.nan
    x[rng.random(x.shape) < 0.02] = 1e-40
    # Mostly a non-zero constant: default bin != most frequent bin (needs zero pushes).
    x[:, 6] = np.where(rng.random(n) < 0.85, 2.5, rng.normal(size=n))
    x[rng.random(n) < 0.1, 6] = 0.0
    # Integer codes for a categorical feature.
    x[:, 7] = rng.integers(0, 6, n) * (rng.random(n) < 0.6)
    y = (
        np.nan_to_num(x[:, 0]) * 2
        + np.nan_to_num(x[:, 1])
        - (x[:, 6] == 0)
        + rng.normal(size=n) * 0.1
    )
    return x, y


def _one_hot(n: int = 3000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Many mutually-exclusive sparse columns, so EFB bundles them (with a few conflicts)."""
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, 40, n)
    x = np.zeros((n, 40))
    x[np.arange(n), cats] = rng.uniform(0.5, 2.0, n)
    clash = rng.random(n) < 0.02
    x[clash, (cats[clash] + 1) % 40] = 1.0
    y = (cats % 7) + rng.normal(size=n) * 0.1
    return x, y


def _model(
    lib: Any, data: Any, y: np.ndarray, params: dict[str, Any], rounds: int = 10, **kw: Any
) -> str:
    booster = lib.train(
        {**BASE, **TRAINABLE, **params}, lib.Dataset(data, y, **kw), num_boost_round=rounds
    )
    return str(booster.model_to_string())


def _both(data: Any, y: np.ndarray, params: dict[str, Any], **kw: Any) -> tuple[str, str]:
    return _model(lgb, data, y, params, **kw), _model(lightgbm, data, y, params, **kw)


def _same_binary(tmp_path: Path, data: Any, y: np.ndarray, params: dict[str, Any]) -> None:
    # Fresh names every call: upstream's save_binary never overwrites an existing file.
    n = len(list(tmp_path.iterdir()))
    ours, theirs = tmp_path / f"ours{n}.bin", tmp_path / f"theirs{n}.bin"
    lgb.Dataset(data, y, params={**BASE, **params}).save_binary(ours)
    lightgbm.Dataset(data, y, params={**BASE, **params}).save_binary(theirs)
    assert ours.read_bytes() == theirs.read_bytes()


def _non_canonical(x: np.ndarray, duplicates: bool = True) -> Any:
    """``x`` as CSR with shuffled column order per row and, optionally, duplicated entries."""
    csr = scipy.sparse.csr_matrix(x)
    rng = np.random.default_rng(9)
    data, indices, indptr = [], [], [0]
    for r in range(csr.shape[0]):
        lo, hi = csr.indptr[r], csr.indptr[r + 1]
        cols, vals = csr.indices[lo:hi], csr.data[lo:hi]
        order = rng.permutation(len(cols))
        cols, vals = cols[order], vals[order]
        if duplicates and len(cols) and r % 5 == 0:
            cols, vals = np.append(cols, cols[0]), np.append(vals, vals[0] * -3 + 1)
        data.append(vals)
        indices.append(cols)
        indptr.append(indptr[-1] + len(cols))
    raw = scipy.sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.array(indptr)), shape=csr.shape
    )
    assert not raw.has_canonical_format
    return raw


CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "bagging": {"bagging_fraction": 0.7, "bagging_freq": 1},
    "zero_as_missing": {"zero_as_missing": True},
    "no_missing": {"use_missing": False},
    "small_bin_sample": {"bin_construct_sample_cnt": 300},
    "min_data_in_bin": {"min_data_in_bin": 20, "max_bin": 31},
    "categorical": {"categorical_feature": [7]},
}

# Dataset-level parity: the binned data, including multi-value groups and bundles.


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize(
    "params", [{}, {"enable_bundle": False}, {"is_enable_sparse": False}, {"zero_as_missing": True}]
)
def test_sparse_dataset_binary_matches_upstream(
    tmp_path: Path, fmt: str, params: dict[str, Any]
) -> None:
    x, y = _data()
    _same_binary(tmp_path, scipy.sparse.csr_matrix(x).asformat(fmt), y, params)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_bundled_sparse_dataset_binary_matches_upstream(tmp_path: Path, fmt: str) -> None:
    x, y = _one_hot()
    for params in ({}, {"max_conflict_rate": 0.05}):
        _same_binary(tmp_path, scipy.sparse.csr_matrix(x).asformat(fmt), y, params)


@pytest.mark.parametrize("params", [{}, {"enable_bundle": False}])
def test_unsorted_csr_dataset_binary_matches_upstream(
    tmp_path: Path, params: dict[str, Any]
) -> None:
    """Unsorted column indices: pairs are pushed in storage order, as upstream does."""
    x, y = _data()
    _same_binary(tmp_path, _non_canonical(x, duplicates=False), y, params)


def test_duplicate_csr_entries_dataset_binary_matches_upstream(tmp_path: Path) -> None:
    """Duplicated columns in a row: the later push wins in dense bins, as upstream.

    Not covered: a duplicate landing in a *sparse* multi-value column. Upstream sorts those
    pushes with ``std::sort`` by row only (src/io/sparse_bin.hpp (v4.7.0) FinishLoad), which
    is unstable, so which duplicate survives depends on the C++ standard library and thread
    buffering; bankai keeps the first push. See ROADMAP P2.
    """
    x, y = _data()
    _same_binary(tmp_path, _non_canonical(x), y, {"enable_bundle": False})


# Training parity.


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("name", list(CONFIGS))
def test_sparse_training_matches_upstream(fmt: str, name: str) -> None:
    x, y = _data()
    params = dict(CONFIGS[name])
    cats = params.pop("categorical_feature", "auto")
    data = scipy.sparse.csr_matrix(x).asformat(fmt)
    ours, theirs = _both(data, y, params, categorical_feature=cats)
    assert ours == theirs


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_training_with_weights(fmt: str) -> None:
    x, y = _data()
    w = np.random.default_rng(3).uniform(0.2, 2.0, len(y))
    ours, theirs = _both(scipy.sparse.csr_matrix(x).asformat(fmt), y, {}, weight=w)
    assert ours == theirs


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_bundled_features_match_upstream(fmt: str) -> None:
    """One-hot-like columns bundle into dense groups, which train today."""
    x, y = _one_hot()
    data = scipy.sparse.csr_matrix(x).asformat(fmt)
    for params in ({"enable_bundle": True}, {"enable_bundle": True, "max_conflict_rate": 0.05}):
        ours, theirs = _both(data, y, params)
        assert ours == theirs, params


def test_sparse_training_with_multi_value_groups() -> None:
    x, y = _data()
    ours, theirs = _both(scipy.sparse.csr_matrix(x), y, {"enable_bundle": True})
    assert ours == theirs


def test_sparse_multiclass_matches_upstream() -> None:
    x, y = _data()
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(float)
    params = {"objective": "multiclass", "num_class": 3}
    ours, theirs = _both(scipy.sparse.csr_matrix(x), labels, params)
    assert ours == theirs


def test_float32_sparse_matches_upstream() -> None:
    x, y = _data()
    ours, theirs = _both(scipy.sparse.csr_matrix(x.astype(np.float32)), y, {})
    assert ours == theirs


def test_non_canonical_csr_training_matches_upstream() -> None:
    x, y = _data()
    ours, theirs = _both(_non_canonical(x), y, {})
    assert ours == theirs


@pytest.mark.parametrize("train_fmt", ["dense", "csr"])
@pytest.mark.parametrize("valid_fmt", ["csr", "csc"])
def test_sparse_validation_set_with_early_stopping(train_fmt: str, valid_fmt: str) -> None:
    x, y = _data(3000)
    xt, yt, xv, yv = x[:2000], y[:2000], x[2000:], y[2000:]
    results: list[tuple[str, dict[str, Any], int]] = []
    for lib in (lgb, lightgbm):
        train_data = xt if train_fmt == "dense" else scipy.sparse.csr_matrix(xt)
        train = lib.Dataset(train_data, yt)
        valid = lib.Dataset(scipy.sparse.csr_matrix(xv).asformat(valid_fmt), yv, reference=train)
        record: dict[str, Any] = {}
        booster = lib.train(
            {**BASE, **TRAINABLE, "metric": "l2"},
            train,
            num_boost_round=30,
            valid_sets=[valid],
            callbacks=[lib.early_stopping(3, verbose=False), lib.record_evaluation(record)],
        )
        results.append((booster.model_to_string(), record, booster.best_iteration))
    (m1, r1, b1), (m2, r2, b2) = results
    assert m1 == m2
    assert b1 == b2
    np.testing.assert_array_equal(r1["valid_0"]["l2"], r2["valid_0"]["l2"])


def test_sparse_cv_matches_upstream() -> None:
    x, y = _data()
    data = scipy.sparse.csr_matrix(x)
    out = [
        lib.cv(
            {**BASE, **TRAINABLE, "metric": "l2"},
            lib.Dataset(data, y),
            num_boost_round=5,
            nfold=3,
            seed=4,
            stratified=False,
        )
        for lib in (lgb, lightgbm)
    ]
    np.testing.assert_array_equal(out[0]["valid l2-mean"], out[1]["valid l2-mean"])


@pytest.mark.parametrize("fmt", ["coo", "lil", "csr_array"])
def test_other_sparse_types_are_converted_like_upstream(fmt: str) -> None:
    x, y = _data(500)
    data: Any = (
        scipy.sparse.csr_array(x)
        if fmt == "csr_array"
        else scipy.sparse.csr_matrix(x).asformat(fmt)
    )
    ours, theirs = _both(data, y, {})
    assert ours == theirs


def test_integer_sparse_data_is_rejected_like_upstream() -> None:
    x = scipy.sparse.csr_matrix(np.eye(20, 4, dtype=np.int64))
    y = np.arange(20, dtype=float)
    with pytest.raises(TypeError) as want:
        lightgbm.Dataset(x, y).construct()
    with pytest.raises(TypeError) as got:
        lgb.Dataset(x, y).construct()
    assert str(got.value) == str(want.value)


def test_save_binary_never_overwrites_an_existing_file(tmp_path: Path) -> None:
    """upstream: src/io/dataset.cpp (v4.7.0) Dataset::SaveBinaryFile warns and keeps it."""
    x, y = _data(300)
    for name, lib in (("ours", lgb), ("theirs", lightgbm)):
        path = tmp_path / f"{name}.bin"
        lib.Dataset(x, y, params=BASE).save_binary(path)
        first = path.read_bytes()
        lib.Dataset(x[:100], y[:100], params=BASE).save_binary(path)
        assert path.read_bytes() == first, name


def test_sparse_dataset_metadata_and_shape() -> None:
    x, y = _data(300)
    ds = lgb.Dataset(scipy.sparse.csr_matrix(x), y).construct()
    up = lightgbm.Dataset(scipy.sparse.csr_matrix(x), y).construct()
    assert (ds.num_data(), ds.num_feature()) == (up.num_data(), up.num_feature())


def test_custom_objective_on_sparse_training_data() -> None:
    """``fobj`` receives the training score; prediction on the sparse training matrix works."""
    x, y = _data(800)
    data = scipy.sparse.coo_matrix(x)  # converted silently, like upstream

    def objective(preds: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        return preds - dataset.get_label(), np.ones_like(preds)

    params = {**BASE, **TRAINABLE, "objective": objective, "metric": "None"}
    ours = lgb.train(params, lgb.Dataset(data, y), num_boost_round=5)
    theirs = lightgbm.train(params, lightgbm.Dataset(data, y), num_boost_round=5)
    x_csr = scipy.sparse.csr_matrix(x)
    np.testing.assert_allclose(ours.predict(x_csr), theirs.predict(x_csr), rtol=1e-12, atol=1e-15)
