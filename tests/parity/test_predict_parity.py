"""P1 acceptance test: upstream models must predict bit-exactly in bankai_lightgbm."""

from pathlib import Path

import bankai_lightgbm as lgb
import numpy as np
import pytest


@pytest.mark.xfail(reason="P1: Booster(model_file=...) not implemented", strict=True)
def test_loaded_upstream_model_predicts_bit_exact(case_dir: Path) -> None:
    test = np.loadtxt(case_dir / "test.csv", delimiter=",")
    booster = lgb.Booster(model_file=str(case_dir / "model.txt"))  # type: ignore[attr-defined]
    np.testing.assert_array_equal(booster.predict(test[:, 1:]), np.load(case_dir / "pred.npy"))
