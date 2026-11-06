"""Datasets built from a text file must train to upstream's model text.

The example file is copied to a temporary directory without its side files (``.init``,
``.weight``, ``.query``), so the only input is the data file itself.
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
    "force_row_wise": True,
    "seed": 7,
}


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "regression.train"
    shutil.copy(EXAMPLE, path)
    return path


def test_training_from_a_text_file_matches_upstream(data_file: Path) -> None:
    theirs = lightgbm.train(PARAMS, lightgbm.Dataset(str(data_file)), num_boost_round=10)
    ours = lgb.train(PARAMS, lgb.Dataset(str(data_file)), num_boost_round=10)
    assert ours.model_to_string() == theirs.model_to_string()


def test_label_and_shape_match_upstream(data_file: Path) -> None:
    theirs = lightgbm.Dataset(str(data_file)).construct()
    ours = lgb.Dataset(str(data_file)).construct()
    assert ours.num_data() == theirs.num_data()
    assert ours.num_feature() == theirs.num_feature()
    assert list(ours.get_label()) == list(theirs.get_label())


def test_side_files_next_to_the_data_are_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "regression.train"
    shutil.copy(EXAMPLE, path)
    (tmp_path / "regression.train.init").write_text("0.5\n")
    with pytest.raises(NotImplementedError, match="side file"):
        lgb.Dataset(str(path)).construct()
