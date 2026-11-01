"""P1 acceptance: scipy.sparse prediction input (and sparse SHAP output) matches upstream.

Every case loads the same upstream model into ``lightgbm==4.7.0`` and ``bankai_lightgbm`` and
predicts the same sparse matrix on this machine, so results must be bit-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pytest
import scipy.sparse
from conftest import FIXTURES, fixture_cases

import bankai_lightgbm as lgb

LINEAR = {
    c for c in fixture_cases() if (FIXTURES / c / "model.txt").read_text().count("is_linear=1")
}


def _sparse_x(case_dir: Path) -> np.ndarray:
    """Test rows with ~70% exact zeros, plus NaNs and sub-``kZeroThreshold`` values."""
    x = np.load(case_dir / "test.npy")[:, 1:].copy()
    rng = np.random.default_rng(42)
    x[rng.random(x.shape) < 0.7] = 0.0
    x[rng.random(x.shape) < 0.03] = np.nan
    x[rng.random(x.shape) < 0.03] = 1e-40
    return x


def _boosters(case_dir: Path) -> tuple[Any, lgb.Booster]:
    model = str(case_dir / "model.txt")
    return lightgbm.Booster(model_file=model), lgb.Booster(model_file=model)


def _assert_same(got: Any, want: Any) -> None:
    if isinstance(want, list):
        assert isinstance(got, list)
        assert len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            _assert_same(g, w)
        return
    if scipy.sparse.issparse(want):
        assert type(got) is type(want)
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        # Upstream stores each row's SHAP entries in std::unordered_map iteration order, which
        # is unspecified; compare the stored pattern and values, not the storage order.
        g, w = got.tocoo(), want.tocoo()
        assert sorted(zip(g.row, g.col, strict=True)) == sorted(zip(w.row, w.col, strict=True))
        np.testing.assert_array_equal(got.toarray(), want.toarray())
        return
    assert isinstance(got, np.ndarray)
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


KINDS: list[dict[str, bool]] = [{}, {"raw_score": True}, {"pred_leaf": True}]


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("kind", KINDS, ids=["normal", "raw", "leaf"])
def test_sparse_predict(case_dir: Path, fmt: str, kind: dict[str, bool]) -> None:
    up, bk = _boosters(case_dir)
    x = scipy.sparse.csr_matrix(_sparse_x(case_dir)).asformat(fmt)
    _assert_same(bk.predict(x, **kind), up.predict(x, **kind))


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_predict_contrib(case_dir: Path, fmt: str) -> None:
    up, bk = _boosters(case_dir)
    x = scipy.sparse.csr_matrix(_sparse_x(case_dir)).asformat(fmt)
    if case_dir.name in LINEAR:
        with pytest.raises(lgb.LightGBMError, match="linear trees"):
            bk.predict(x, pred_contrib=True)
        return
    _assert_same(bk.predict(x, pred_contrib=True), up.predict(x, pred_contrib=True))


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_predict_float32_and_iteration_range(case_dir: Path, fmt: str) -> None:
    up, bk = _boosters(case_dir)
    x = scipy.sparse.csr_matrix(_sparse_x(case_dir).astype(np.float32)).asformat(fmt)
    kw = {"start_iteration": 1, "num_iteration": 2}
    _assert_same(bk.predict(x, **kw), up.predict(x, **kw))
    if case_dir.name not in LINEAR:
        # float32 input gives float32 sparse SHAP output, like upstream.
        _assert_same(bk.predict(x, pred_contrib=True, **kw), up.predict(x, pred_contrib=True, **kw))


def _first_case() -> Path:
    return FIXTURES / next(c for c in fixture_cases() if c not in LINEAR and "multiclass" not in c)


def test_csr_duplicates_and_unsorted_indices_follow_upstream() -> None:
    case = _first_case()
    up, bk = _boosters(case)
    x = _sparse_x(case)
    csr = scipy.sparse.csr_matrix(x)
    rows = []
    for r in range(csr.shape[0]):
        lo, hi = csr.indptr[r], csr.indptr[r + 1]
        cols, vals = csr.indices[lo:hi][::-1], csr.data[lo:hi][::-1]  # unsorted
        if len(cols):  # duplicate the first entry with another value: last one wins upstream
            cols, vals = np.append(cols, cols[0]), np.append(vals, vals[0] * 3 + 1)
        rows.append((cols, vals))
    indptr = np.cumsum([0] + [len(c) for c, _ in rows])
    raw = scipy.sparse.csr_matrix(
        (np.concatenate([v for _, v in rows]), np.concatenate([c for c, _ in rows]), indptr),
        shape=csr.shape,
    )
    assert not raw.has_canonical_format
    for kind in KINDS:
        _assert_same(bk.predict(raw, **kind), up.predict(raw, **kind))


def test_multiclass_sparse_contrib_is_a_list_per_class() -> None:
    cases = [c for c in fixture_cases() if c.startswith("multiclass")]
    if not cases:
        pytest.skip("no multiclass fixture")
    up, bk = _boosters(FIXTURES / cases[0])
    x = scipy.sparse.csr_matrix(_sparse_x(FIXTURES / cases[0]))
    got = bk.predict(x, pred_contrib=True)
    assert isinstance(got, list)
    _assert_same(got, up.predict(x, pred_contrib=True))


@pytest.mark.parametrize("fmt", ["coo", "lil", "dok", "csr_array"])
def test_other_sparse_types_are_converted_with_upstream_warning(fmt: str) -> None:
    case = _first_case()
    up, bk = _boosters(case)
    dense = _sparse_x(case)[:40]
    x: Any = (
        scipy.sparse.csr_array(dense)
        if fmt == "csr_array"
        else scipy.sparse.csr_matrix(dense).asformat(fmt)
    )
    # upstream's default logger emits a UserWarning (basic._DummyLogger.warning).
    with pytest.warns(UserWarning, match="Converting data to scipy sparse matrix."):
        want = up.predict(x)
    with pytest.warns(UserWarning, match="Converting data to scipy sparse matrix."):
        got = bk.predict(x)
    _assert_same(got, want)


def test_sparse_shape_check_and_disable() -> None:
    case = _first_case()
    up, bk = _boosters(case)
    x = scipy.sparse.hstack(
        [scipy.sparse.csr_matrix(_sparse_x(case)[:30]), scipy.sparse.csr_matrix(np.ones((30, 2)))],
        format="csr",
    )
    with pytest.raises(lgb.LightGBMError, match="The number of features in data"):
        bk.predict(x)
    kw = {"predict_disable_shape_check": True}
    _assert_same(bk.predict(x, **kw), up.predict(x, **kw))


def test_integer_sparse_data_is_rejected_like_upstream() -> None:
    case = _first_case()
    up, bk = _boosters(case)
    x = scipy.sparse.csr_matrix(np.eye(5, bk.num_feature(), dtype=np.int64))
    with pytest.raises(TypeError) as want:
        up.predict(x)
    with pytest.raises(TypeError) as got:
        bk.predict(x)
    assert str(got.value) == str(want.value)


def test_empty_sparse_rows() -> None:
    case = _first_case()
    up, bk = _boosters(case)
    for fmt in ("csr", "csc"):
        x = scipy.sparse.csr_matrix((7, bk.num_feature())).asformat(fmt)
        for kind in [*KINDS, {"pred_contrib": True}]:
            _assert_same(bk.predict(x, **kind), up.predict(x, **kind))
