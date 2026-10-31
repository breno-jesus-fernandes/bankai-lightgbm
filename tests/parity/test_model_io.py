"""Model serialization parity: model_to_string / save_model / dump_model / importance."""

import copy
import pickle
from pathlib import Path

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb


def _pair(case_dir: Path) -> tuple[lgb.Booster, lightgbm.Booster]:
    path = case_dir / "model.txt"
    return lgb.Booster(model_file=path), lightgbm.Booster(model_file=str(path))


def test_model_to_string_matches(case_dir: Path) -> None:
    ours, theirs = _pair(case_dir)
    assert ours.model_to_string() == theirs.model_to_string()
    assert ours.model_to_string(importance_type="gain") == theirs.model_to_string(
        importance_type="gain"
    )
    assert ours.model_to_string(num_iteration=4, start_iteration=2) == theirs.model_to_string(
        num_iteration=4, start_iteration=2
    )


def test_save_model_file_matches(case_dir: Path, tmp_path: Path) -> None:
    ours, theirs = _pair(case_dir)
    ours.save_model(tmp_path / "ours.txt")
    theirs.save_model(tmp_path / "theirs.txt")
    assert (tmp_path / "ours.txt").read_bytes() == (tmp_path / "theirs.txt").read_bytes()


def test_dump_model_matches(case_dir: Path) -> None:
    ours, theirs = _pair(case_dir)
    assert ours.dump_model() == theirs.dump_model()
    assert ours.dump_model(num_iteration=3, start_iteration=1, importance_type="gain") == (
        theirs.dump_model(num_iteration=3, start_iteration=1, importance_type="gain")
    )


def test_feature_importance_matches(case_dir: Path) -> None:
    ours, theirs = _pair(case_dir)
    for kind in ("split", "gain"):
        got, want = ours.feature_importance(kind), theirs.feature_importance(kind)
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(
        ours.feature_importance(iteration=3), theirs.feature_importance(iteration=3)
    )


def test_bounds_and_leaf_output(case_dir: Path) -> None:
    ours, theirs = _pair(case_dir)
    assert ours.upper_bound() == theirs.upper_bound()
    assert ours.lower_bound() == theirs.lower_bound()
    assert ours.get_leaf_output(2, 1) == theirs.get_leaf_output(2, 1)


def test_pickle_and_copy_roundtrip(case_dir: Path) -> None:
    ours, _ = _pair(case_dir)
    x = np.load(case_dir / "test.npy")[:20, 1:]
    want = ours.predict(x)
    for clone in (pickle.loads(pickle.dumps(ours)), copy.copy(ours), copy.deepcopy(ours)):
        np.testing.assert_array_equal(clone.predict(x), want)
        assert clone.model_to_string() == ours.model_to_string()


def test_model_from_string(case_dir: Path) -> None:
    ours, theirs = _pair(case_dir)
    other = lgb.Booster(model_file=case_dir.parent / "binary" / "model.txt")
    other.model_from_string(theirs.model_to_string())
    assert other.model_to_string() == ours.model_to_string()


def test_unknown_importance_type_raises(case_dir: Path) -> None:
    ours, _ = _pair(case_dir)
    with pytest.raises(KeyError):
        ours.feature_importance("cover")
