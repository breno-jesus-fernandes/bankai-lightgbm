"""Dataset field get/set on a constructed Dataset: exception types and messages match upstream.

upstream: python-package/lightgbm/basic.py (v4.7.0) ``Dataset.set_field`` / ``get_field`` and
``Dataset.set_label``, ``set_weight``, ``set_group``; src/io/metadata.cpp (v4.7.0).
"""

from __future__ import annotations

from typing import Any

import lightgbm as up
import numpy as np
import pytest

import bankai_lightgbm as bk


def _constructed(mod: Any) -> Any:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    return mod.Dataset(x, y, free_raw_data=False, params={"verbose": -1}).construct()


# case name -> action on a constructed Dataset of `mod`
ERROR_CASES = {
    "get_unknown_field": lambda ds: ds.get_field("nope"),
    "set_unknown_field": lambda ds: ds.set_field("nope", [1]),
    "set_label_none": lambda ds: ds.set_field("label", None),
    "set_label_2d": lambda ds: ds.set_field("label", np.ones((20, 2))),
    "set_label_wrong_length": lambda ds: ds.set_field("label", np.ones(19)),
    "set_weight_wrong_length": lambda ds: ds.set_field("weight", [1.0, 2.0]),
    "set_group_wrong_sum": lambda ds: ds.set_field("group", [5, 5]),
}


def _outcome(mod: Any, case: str) -> tuple[str, str] | None:
    try:
        ERROR_CASES[case](_constructed(mod))
    except Exception as exc:
        return type(exc).__name__, str(exc)
    return None


@pytest.mark.parametrize("case", list(ERROR_CASES))
def test_field_error_type_and_message_match_upstream(case: str) -> None:
    want = _outcome(up, case)
    assert want is not None, f"upstream did not raise for {case}: the case is stale"
    assert _outcome(bk, case) == want


def test_field_values_match_upstream() -> None:
    up_ds, bk_ds = _constructed(up), _constructed(bk)
    for name in ["label", "weight", "init_score", "group", "position"]:
        want = up_ds.get_field(name)
        got = bk_ds.get_field(name)
        assert (want is None) == (got is None), name
        if want is not None:
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want), err_msg=name)
    for mod_ds in (up_ds, bk_ds):
        mod_ds.set_field("group", [10, 10])
        mod_ds.set_field("group", None)
    np.testing.assert_array_equal(
        np.asarray(bk_ds.set_field("label", list(range(20))).get_field("label")),
        np.asarray(up_ds.set_field("label", list(range(20))).get_field("label")),
    )
