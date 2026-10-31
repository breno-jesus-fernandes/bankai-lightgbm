"""Booster API behaviour compared live against upstream lightgbm."""

from pathlib import Path

import bankai_lightgbm as lgb
import lightgbm
import numpy as np
import pytest

FIX = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def pair() -> tuple[lgb.Booster, lightgbm.Booster, np.ndarray]:
    path = FIX / "multiclass" / "model.txt"
    x = np.load(FIX / "multiclass" / "test.npy")[:, 1:]
    return lgb.Booster(model_file=path), lightgbm.Booster(model_file=str(path)), x


def test_introspection_matches(pair: tuple[lgb.Booster, lightgbm.Booster, np.ndarray]) -> None:
    ours, theirs, _ = pair
    assert ours.num_trees() == theirs.num_trees()
    assert ours.num_model_per_iteration() == theirs.num_model_per_iteration()
    assert ours.current_iteration() == theirs.current_iteration()
    assert ours.num_feature() == theirs.num_feature()
    assert ours.feature_name() == theirs.feature_name()


def test_float32_and_list_inputs(pair: tuple[lgb.Booster, lightgbm.Booster, np.ndarray]) -> None:
    ours, theirs, x = pair
    x32 = x.astype(np.float32)
    np.testing.assert_array_equal(ours.predict(x32), theirs.predict(x32))
    np.testing.assert_array_equal(ours.predict(x[:5].tolist()), theirs.predict(x[:5].tolist()))


def test_fortran_order_input(pair: tuple[lgb.Booster, lightgbm.Booster, np.ndarray]) -> None:
    ours, theirs, x = pair
    xf = np.asfortranarray(x)
    np.testing.assert_array_equal(ours.predict(xf), theirs.predict(xf))


def test_shape_check(pair: tuple[lgb.Booster, lightgbm.Booster, np.ndarray]) -> None:
    ours, theirs, x = pair
    with pytest.raises(lgb.LightGBMError, match="number of features"):
        ours.predict(x[:, :5])
    with pytest.raises(lightgbm.basic.LightGBMError, match="number of features"):
        theirs.predict(x[:, :5])
    wide = np.hstack([x, x[:, :2]])
    np.testing.assert_array_equal(
        ours.predict(wide, predict_disable_shape_check=True),
        theirs.predict(wide, predict_disable_shape_check=True),
    )


def test_prediction_early_stopping(pair: tuple[lgb.Booster, lightgbm.Booster, np.ndarray]) -> None:
    ours, theirs, x = pair
    kw = {"pred_early_stop": True, "pred_early_stop_freq": 2, "pred_early_stop_margin": 0.5}
    np.testing.assert_array_equal(ours.predict(x, **kw), theirs.predict(x, **kw))
    np.testing.assert_array_equal(
        ours.predict(x, raw_score=True, **kw), theirs.predict(x, raw_score=True, **kw)
    )


def test_one_dimensional_input_is_rejected(
    pair: tuple[lgb.Booster, lightgbm.Booster, np.ndarray],
) -> None:
    ours, _, x = pair
    with pytest.raises(ValueError, match="2 dimensional"):
        ours.predict(x[0])


def test_needs_a_model() -> None:
    with pytest.raises(TypeError):
        lgb.Booster()


def test_bad_model_raises_lightgbm_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("tree\nnum_class=1\n")
    with pytest.raises(lgb.LightGBMError):
        lgb.Booster(model_file=bad)


def test_pandas_categorical_roundtrip() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "c": pd.Categorical(rng.choice(["x", "y", "z", "w"], size=n)),
        }
    )
    y = df["a"] + (df["c"] == "y") * 2.0
    params = {"verbose": -1, "num_leaves": 7, "deterministic": True, "num_threads": 1}
    theirs = lightgbm.train(params, lightgbm.Dataset(df, y), num_boost_round=5)
    ours = lgb.Booster(model_str=theirs.model_to_string())
    assert ours.pandas_categorical == theirs.pandas_categorical
    test = df.head(50).copy()
    test["c"] = pd.Categorical(test["c"].astype(str), categories=["w", "z", "y", "x", "new"])
    np.testing.assert_array_equal(ours.predict(test), theirs.predict(test))
