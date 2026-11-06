"""``two_round=true`` reads a text file in two passes; the model must match upstream.

The bin mappers come from a reservoir sample of the file's lines (upstream:
``TextReader::SampleFromFile``), so ``bin_construct_sample_cnt`` is set below the number of rows
to exercise the random draws. The in-memory path samples differently and is not compared here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import lightgbm
import pytest

import bankai_lightgbm as lgb

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "regression" / "regression.train"

PARAMS = {
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
    "two_round": True,
    "bin_construct_sample_cnt": 500,
}


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    path = tmp_path / "regression.train"
    shutil.copy(EXAMPLE, path)
    return path


def test_two_round_training_matches_upstream(data_path: Path) -> None:
    upstream = lightgbm.train(PARAMS, lightgbm.Dataset(str(data_path)), num_boost_round=10)
    ours = lgb.train(PARAMS, lgb.Dataset(str(data_path)), num_boost_round=10)

    assert ours.model_to_string() == upstream.model_to_string()
