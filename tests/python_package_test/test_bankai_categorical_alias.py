"""Categorical columns given through a parameter alias (upstream: Config alias categorical_column)."""

import numpy as np
import pytest

import bankai_lightgbm as lgb


@pytest.mark.parametrize("alias", ["categorical_column", "categorical_feature", "cat_column"])
def test_categorical_alias_in_params_builds_categorical_split(alias):
    x = np.arange(8).reshape(8, 1).astype(float)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    params = {
        "objective": "regression",
        "verbose": -1,
        "boost_from_average": False,
        "min_data": 1,
        "num_leaves": 2,
        "learning_rate": 1,
        "min_data_in_bin": 1,
        "min_data_per_group": 1,
        "cat_smooth": 1,
        "cat_l2": 0,
        "max_cat_to_onehot": 1,
        "zero_as_missing": True,
        alias: 0,
    }
    bst = lgb.train(params, lgb.Dataset(x, y), num_boost_round=1)
    np.testing.assert_allclose(bst.predict(x), [0, 1, 0, 1, 0, 1, 0, 1])
    tree = bst.dump_model()["tree_info"][0]["tree_structure"]
    assert tree["decision_type"] == "=="
