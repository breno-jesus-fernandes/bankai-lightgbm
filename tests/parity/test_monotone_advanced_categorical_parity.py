"""Advanced categorical searches use the initial reverse cumulative child bounds.

upstream: src/treelearner/feature_histogram.cpp (v4.7.0)
FindBestThresholdCategoricalInner / FindBestThresholdCategoricalIntInner.
"""

import lightgbm
import numpy as np
import pytest

import bankai_lightgbm as lgb


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("onehot", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_advanced_categorical_tree_parity(quantized: bool, onehot: bool, missing: bool) -> None:
    rng = np.random.default_rng(41)
    x = rng.normal(size=(1500, 4))
    y = 2 * x[:, 0] - 1.5 * x[:, 1] + np.sin(3 * x[:, 3]) + rng.normal(scale=0.3, size=1500)
    x[:, 2] = rng.integers(0, 5, size=1500)
    y += 0.5 * (x[:, 2] == 1)
    if missing:
        x[rng.random(1500) < 0.1, 3] = np.nan
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "min_data_in_bin": 1,
        "monotone_constraints": [1, -1, 0, 1],
        "monotone_constraints_method": "advanced",
        "max_cat_to_onehot": 10 if onehot else 4,
        "use_quantized_grad": quantized,
        "num_grad_quant_bins": 16,
        "stochastic_rounding": False,
        "quant_train_renew_leaf": False,
        "verbosity": -1,
        "num_threads": 1,
        "deterministic": True,
        "seed": 5,
    }
    theirs = lightgbm.train(
        params, lightgbm.Dataset(x, y, categorical_feature=[2]), num_boost_round=20
    )
    ours = lgb.train(params, lgb.Dataset(x, y, categorical_feature=[2]), num_boost_round=20)
    # Text fields cover features, thresholds, child indices, counts and unrounded leaf values.
    # Parameter blocks echo implementation-specific defaults, so compare just the tree section.
    assert (
        ours.model_to_string().split("\nparameters:")[0]
        == theirs.model_to_string().split("\nparameters:")[0]
    )
