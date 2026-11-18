"""`forcedbins_filename` shows up in the dataset text dump (ROADMAP T5.2).

upstream: src/io/dataset.cpp (v4.7.0) Dataset::DumpTextFile prints a `forced_bins` section with
the per-feature upper bounds that `DatasetLoader::GetForcedBins` read from `forcedbins_filename`.
bankai applied the forced bounds when binning but did not store them on the dataset, so the dump's
`forced_bins` section was empty. Both libraries construct from the same data and the same file, so
the dumps (bin columns and the forced-bins section) must be identical.
"""

from __future__ import annotations

import json

import lightgbm
import numpy as np

import bankai_lightgbm as lgb


def _dense(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(80, 4))
    x[rng.random(size=x.shape) < 0.1] = np.nan
    return x


def _forced_bins_file(tmp_path) -> str:
    spec = [
        {"feature": 0, "bin_upper_bound": [-0.5, 0.0, 0.5]},
        {"feature": 2, "bin_upper_bound": [0.25, 1.0]},
    ]
    path = tmp_path / "forced_bins.json"
    path.write_text(json.dumps(spec))
    return str(path)


def _dump(library, data, forced_file: str, tmp_path, name: str) -> str:
    params = {
        "verbosity": -1,
        "min_data_in_leaf": 1,
        "min_data_in_bin": 1,
        "max_bin": 15,
        "forcedbins_filename": forced_file,
    }
    dataset = library.Dataset(data, label=np.zeros(data.shape[0]), params=params)
    path = tmp_path / name
    dataset.construct()
    dataset._dump_text(path)
    return path.read_text()


def test_forced_bins_dump_matches_upstream_bytes(tmp_path) -> None:
    data = _dense()
    forced = _forced_bins_file(tmp_path)
    want = _dump(lightgbm, data, forced, tmp_path, "upstream.txt")
    got = _dump(lgb, data, forced, tmp_path, "bankai.txt")
    assert got == want
    # The forced bounds must actually appear (guards against an empty section passing vacuously).
    assert "forced_bins" in got
    assert "0.500000" in got
