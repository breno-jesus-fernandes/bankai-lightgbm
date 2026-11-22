"""A binary dataset file written by either library must train to the same model text.

Each library saves its own binary file with ``save_binary`` and loads it back; the upstream
binary format is shared, so the two models must match exactly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import lightgbm
import numpy as np
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


@pytest.mark.parametrize("override", [False, True])
def test_binary_file_reference_and_metadata_match_upstream(tmp_path: Path, override: bool) -> None:
    x = np.arange(128, dtype=np.float64).reshape(-1, 1)
    binary = tmp_path / "metadata.bin"
    lightgbm.Dataset(
        x, label=x[:, 0], weight=np.full(128, 2.0), group=[64, 64], params=PARAMS
    ).save_binary(str(binary))
    fields = (
        {
            "label": -x[:, 0],
            "weight": np.linspace(1.0, 3.0, 128),
            "group": [32, 96],
            "init_score": np.column_stack((x[:, 0], -x[:, 0])),
        }
        if override
        else {}
    )
    expected = []
    actual = []
    for lib, results in ((lightgbm, expected), (lgb, actual)):
        # Binary files keep their own feature layout, regardless of the reference.
        reference = lib.Dataset(np.column_stack((x, x)), label=x[:, 0], params=PARAMS)
        loaded = lib.Dataset(str(binary), reference=reference, params=PARAMS, **fields).construct()
        assert loaded.num_feature() == 1
        results.extend(
            (loaded.get_label(), loaded.get_weight(), loaded.get_group(), loaded.get_init_score())
        )
    for ours, theirs in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_training_uses_binary_file_overrides(tmp_path: Path) -> None:
    x = np.arange(128, dtype=np.float64).reshape(-1, 1)
    binary = tmp_path / "training.bin"
    lightgbm.Dataset(x, label=x[:, 0], params=PARAMS).save_binary(str(binary))
    label = np.sin(x[:, 0] / 10.0)
    weight = np.linspace(1.0, 3.0, 128)
    init_score = x[:, 0] / 128
    scores = []
    for lib in (lightgbm, lgb):
        data = lib.Dataset(
            str(binary), label=label, weight=weight, init_score=init_score, params=PARAMS
        )
        model = lib.train(PARAMS, data, num_boost_round=3)
        scores.append(model.predict(x, raw_score=True, num_threads=1))
    np.testing.assert_allclose(scores[1], scores[0], rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label", [1.0], "Length of labels differs"),
        ("weight", [2.0], "Length of weights differs"),
        ("group", [50], r"Sum of query counts \(128\) differs.*\(50\)"),
        ("init_score", [1.0], "Initial score size doesn't match"),
    ],
)
def test_binary_file_rejects_incompatible_metadata(
    tmp_path: Path, field: str, value: list[float], message: str
) -> None:
    x = np.arange(128, dtype=np.float64).reshape(-1, 1)
    binary = tmp_path / "invalid-metadata.bin"
    lightgbm.Dataset(x, label=x[:, 0], params=PARAMS).save_binary(str(binary))
    for lib in (lightgbm, lgb):
        with pytest.raises(lib.basic.LightGBMError, match=message):
            lib.Dataset(str(binary), params=PARAMS, **{field: value}).construct()
