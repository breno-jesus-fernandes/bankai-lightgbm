"""P1 acceptance: upstream models predict bit-exactly through the Python API."""

from pathlib import Path

import numpy as np
import pytest
from conftest import assert_parity

import bankai_lightgbm as lgb


def _x(case_dir: Path) -> np.ndarray:
    return np.load(case_dir / "test.npy")[:, 1:]


def test_predict(case_dir: Path) -> None:
    booster = lgb.Booster(model_file=case_dir / "model.txt")
    assert_parity(booster.predict(_x(case_dir)), np.load(case_dir / "pred.npy"))


def test_predict_raw_score(case_dir: Path) -> None:
    booster = lgb.Booster(model_file=case_dir / "model.txt")
    assert_parity(booster.predict(_x(case_dir), raw_score=True), np.load(case_dir / "pred_raw.npy"))


def test_predict_leaf(case_dir: Path) -> None:
    booster = lgb.Booster(model_file=case_dir / "model.txt")
    assert_parity(
        booster.predict(_x(case_dir), pred_leaf=True), np.load(case_dir / "pred_leaf.npy")
    )


def test_predict_contrib(case_dir: Path) -> None:
    booster = lgb.Booster(model_file=case_dir / "model.txt")
    want = case_dir / "pred_contrib.npy"
    if not want.exists():
        with pytest.raises(lgb.LightGBMError, match="linear trees"):
            booster.predict(_x(case_dir), pred_contrib=True)
        return
    assert_parity(booster.predict(_x(case_dir), pred_contrib=True), np.load(want))


def test_predict_iteration_range(case_dir: Path) -> None:
    booster = lgb.Booster(model_str=(case_dir / "model.txt").read_text())
    got = booster.predict(_x(case_dir), start_iteration=3, num_iteration=5)
    assert_parity(got, np.load(case_dir / "pred_iter_3_5.npy"))
