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


def test_init_and_weight_side_files_match_upstream(tmp_path: Path) -> None:
    """``<data>.init`` and ``<data>.weight`` next to the data are loaded like upstream does."""
    path = tmp_path / "regression.train"
    shutil.copy(EXAMPLE, path)
    num_rows = sum(1 for _ in EXAMPLE.open())
    init_lines = [f"{0.1 * (i % 5):.3f}" for i in range(num_rows)]
    weight_lines = [f"{1.0 + 0.5 * (i % 3):.1f}" for i in range(num_rows)]
    (tmp_path / "regression.train.init").write_text("\n".join(init_lines) + "\n")
    (tmp_path / "regression.train.weight").write_text("\n".join(weight_lines) + "\n")

    upstream = lightgbm.train(PARAMS, lightgbm.Dataset(str(path)), num_boost_round=10)
    ours = lgb.train(PARAMS, lgb.Dataset(str(path)), num_boost_round=10)

    assert ours.model_to_string() == upstream.model_to_string()


def test_explicit_init_score_overrides_the_side_file(tmp_path: Path) -> None:
    """An init_score passed to Dataset wins over ``<data>.init``, as in upstream."""
    import numpy as np

    path = tmp_path / "regression.train"
    shutil.copy(EXAMPLE, path)
    num_rows = sum(1 for _ in EXAMPLE.open())
    (tmp_path / "regression.train.init").write_text("\n".join(["9.0"] * num_rows) + "\n")
    explicit = np.full(num_rows, 0.25)

    upstream = lightgbm.train(
        PARAMS, lightgbm.Dataset(str(path), init_score=explicit), num_boost_round=10
    )
    ours = lgb.train(PARAMS, lgb.Dataset(str(path), init_score=explicit), num_boost_round=10)

    assert ours.model_to_string() == upstream.model_to_string()
