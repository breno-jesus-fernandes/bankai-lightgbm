"""An unknown parameter is reported as often as upstream reports it.

upstream: src/io/config.cpp (v4.7.0) and include/LightGBM/config.h (v4.7.0) ``Config::Set``
log ``Unknown parameter: <key>`` once per config parse; the Booster path parses the config
when the Dataset is constructed and again when the C++ booster is created.
"""

from __future__ import annotations

from typing import Any

import lightgbm as up
import numpy as np
import pytest

import bankai_lightgbm as bk

KEY = "my_unknown_kw"


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    return x, x[:, 0] * 2


def _booster_path(mod: Any) -> None:
    x, y = _data()
    params = {"objective": "regression", KEY: 7, "verbose": 1, "num_threads": 1}
    mod.train(params, mod.Dataset(x, y), num_boost_round=1)


def _sklearn_path(mod: Any) -> None:
    x, y = _data()
    mod.LGBMRegressor(n_estimators=1, verbose=1, **{KEY: 7}).fit(x, y)


@pytest.mark.parametrize("path", [_booster_path, _sklearn_path], ids=["booster", "sklearn"])
def test_unknown_parameter_warning_count_matches_upstream(
    path: Any, capfd: pytest.CaptureFixture[str]
) -> None:
    capfd.readouterr()
    path(up)
    want = capfd.readouterr().out.count(f"Unknown parameter: {KEY}")
    assert want > 0, "upstream printed no warning: the case is stale"
    path(bk)
    got = capfd.readouterr().out.count(f"Unknown parameter: {KEY}")
    assert got == want
