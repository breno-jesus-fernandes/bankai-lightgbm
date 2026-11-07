"""Position files (``<data>.position``) must load like upstream.

upstream: src/io/metadata.cpp (Metadata::LoadPositions, SetPosition) and
python-package/lightgbm/basic.py (Dataset.get_position). Position ids are densified in
first-seen order.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb


def _write_ranking_files(directory: Path, num_rows: int) -> Path:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(num_rows, 3))
    y = rng.integers(0, 3, size=num_rows)
    data = directory / "rank.train"
    with data.open("w") as f:
        for label, row in zip(y, x, strict=True):
            f.write("\t".join([str(label), *(f"{v:.6f}" for v in row)]) + "\n")
    (directory / "rank.train.query").write_text("\n".join(["4"] * (num_rows // 4)) + "\n")
    return data


def _write_positions(directory: Path, num_rows: int) -> None:
    # Raw ids in a non-sorted order, so first-seen densification differs from sorting.
    raw = ["pos_3" if i % 3 == 0 else ("pos_1" if i % 3 == 1 else "pos_2") for i in range(num_rows)]
    (directory / "rank.train.position").write_text("\n".join(raw) + "\n")


def test_position_file_is_read_and_densified_like_upstream(tmp_path: Path) -> None:
    num_rows = 40
    data = _write_ranking_files(tmp_path, num_rows)
    _write_positions(tmp_path, num_rows)
    up = lightgbm.Dataset(str(data)).construct()
    ours = lgb.Dataset(str(data)).construct()
    np.testing.assert_array_equal(np.asarray(ours.get_position()), np.asarray(up.get_position()))


def test_position_file_with_extra_row_is_rejected(tmp_path: Path) -> None:
    num_rows = 40
    data = _write_ranking_files(tmp_path, num_rows)
    _write_positions(tmp_path, num_rows)
    with (tmp_path / "rank.train.position").open("a") as f:
        f.write("pos_1000\n")
    with pytest.raises(
        lgb.basic.LightGBMError, match=r"Positions size \(41\) doesn't match data size"
    ):
        lgb.Dataset(str(data)).construct()
