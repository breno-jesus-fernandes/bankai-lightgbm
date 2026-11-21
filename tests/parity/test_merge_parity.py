"""``Booster.merge`` must match upstream ``GBDT::MergeFrom`` / ``LGBM_BoosterMerge`` semantics.

Upstream LightGBM 4.7.0 exposes no Python ``Booster.merge``, so the defining property is checked
against upstream numbers: merging ``B`` into ``A`` appends ``B``'s trees, so the merged model's raw
score equals the sum of ``A``'s and ``B``'s raw scores. Both component models are trained in
``lightgbm`` and loaded into ``bankai_lightgbm`` for the merge.
"""

from __future__ import annotations

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb

PARAMS = {
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "seed": 7,
}


def _data(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 2.0 * X[:, 0] + X[:, 1] ** 2 + rng.normal(scale=0.1, size=n)
    return X, y


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_merge_sums_component_raw_scores(objective: str) -> None:
    Xa, ya = _data(1, 400)
    Xb, yb = _data(2, 350)
    Xt, _ = _data(3, 120)
    if objective == "binary":
        ya = (ya > 0).astype(float)
        yb = (yb > 0).astype(float)
    params = {**PARAMS, "objective": objective}

    up_a = lightgbm.train(params, lightgbm.Dataset(Xa, ya), num_boost_round=8)
    up_b = lightgbm.train(params, lightgbm.Dataset(Xb, yb), num_boost_round=6)
    want = up_a.predict(Xt, raw_score=True) + up_b.predict(Xt, raw_score=True)

    a = lgb.Booster(model_str=up_a.model_to_string())
    b = lgb.Booster(model_str=up_b.model_to_string())
    merged = a.merge(b)

    assert merged is a  # merge mutates in place and returns self
    assert a.num_trees() == up_a.num_trees() + up_b.num_trees()

    got = a.predict(Xt, raw_score=True)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)


def test_merge_rejects_incompatible_models() -> None:
    Xr, yr = _data(1, 300)
    Xm, _ = _data(2, 300)
    ym = (np.arange(300) % 3).astype(float)

    reg = lightgbm.train({**PARAMS, "objective": "regression"}, lightgbm.Dataset(Xr, yr), num_boost_round=5)
    multi = lightgbm.train(
        {**PARAMS, "objective": "multiclass", "num_class": 3},
        lightgbm.Dataset(Xm, ym),
        num_boost_round=5,
    )
    a = lgb.Booster(model_str=reg.model_to_string())
    b = lgb.Booster(model_str=multi.model_to_string())

    with pytest.raises(lgb.basic.LightGBMError):
        a.merge(b)
