"""A validation Dataset takes its reference's parameters (upstream ``Dataset.construct``)."""

import warnings

import numpy as np

import bankai_lightgbm as lgb

_MSG = "Overriding the parameters from Reference Dataset."


def _data():
    rng = np.random.default_rng(0)
    return rng.uniform(size=(100, 2)), rng.uniform(size=100)


def test_reference_params_are_copied_with_a_warning():
    X, y = _data()
    train = lgb.Dataset(X, y, params={"max_bin": 100}, free_raw_data=False).construct()
    val = lgb.Dataset(X, y, reference=train, params={"max_bin": 50}, free_raw_data=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val.construct()
    assert val.get_params()["max_bin"] == 100
    assert any(_MSG in str(w.message) for w in caught)


def test_only_categorical_difference_does_not_warn():
    X, y = _data()
    train_params = {"max_bin": 100, "categorical_feature": [0]}
    train = lgb.Dataset(X, y, params=train_params, free_raw_data=False).construct()
    val = lgb.Dataset(X, y, reference=train, params={"max_bin": 100}, free_raw_data=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val.construct()
    assert val.get_params()["categorical_feature"] == [0]
    assert not any(_MSG in str(w.message) for w in caught)


def test_train_restores_validation_params_without_warning():
    X, y = _data()
    train = lgb.Dataset(X, y)
    val = lgb.Dataset(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lgb.train(
            {"objective": "binary", "verbose": -1, "num_threads": 1}, train, 2, valid_sets=[val]
        )
    assert not any(_MSG in str(w.message) for w in caught)
    assert "objective" not in val.get_params()
