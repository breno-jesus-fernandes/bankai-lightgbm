"""A binary dataset file written by either library must train to the same model text.

Each library saves its own binary file with ``save_binary`` and loads it back; the upstream
binary format is shared, so the two models must match exactly.
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
}


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    path = tmp_path / "regression.train"
    shutil.copy(EXAMPLE, path)
    return path


def test_training_from_a_binary_file_matches_upstream(tmp_path: Path, data_path: Path) -> None:
    upstream_bin = tmp_path / "upstream.bin"
    ours_bin = tmp_path / "ours.bin"
    lightgbm.Dataset(str(data_path), params={"verbose": -1}).save_binary(str(upstream_bin))
    lgb.Dataset(str(data_path), params={"verbose": -1}).save_binary(ours_bin)

    upstream = lightgbm.train(PARAMS, lightgbm.Dataset(str(upstream_bin)), num_boost_round=10)
    ours = lgb.train(PARAMS, lgb.Dataset(str(ours_bin)), num_boost_round=10)

    assert ours.model_to_string() == upstream.model_to_string()


def test_upstream_binary_file_loads_and_matches_its_text_source(
    tmp_path: Path, data_path: Path
) -> None:
    upstream_bin = tmp_path / "upstream.bin"
    lightgbm.Dataset(str(data_path), params={"verbose": -1}).save_binary(str(upstream_bin))

    from_text = lgb.train(PARAMS, lgb.Dataset(str(data_path)), num_boost_round=10)
    from_binary = lgb.train(PARAMS, lgb.Dataset(str(upstream_bin)), num_boost_round=10)

    assert from_binary.model_to_string() == from_text.model_to_string()
