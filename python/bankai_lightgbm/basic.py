"""Core objects: ``Booster`` (and, later, ``Dataset``). Mirrors ``lightgbm.basic``."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bankai_lightgbm._lib import LightGBMError, NativeBooster, NativeDataset

__all__ = ["Booster", "Dataset", "LightGBMError"]


def _param_value_str(value: Any) -> str:
    """Parameter values as the C API takes them (``lightgbm`` formats them the same way)."""
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, list | tuple | np.ndarray):
        return ",".join(_param_value_str(v) for v in value)
    return str(value)


def _params_to_strings(params: dict[str, Any]) -> dict[str, str]:
    return {str(k): _param_value_str(v) for k, v in params.items() if v is not None}


def _categorical_indices(
    categorical_feature: Any, feature_names: list[str] | None, pandas_cat_cols: list[int]
) -> list[int]:
    if categorical_feature == "auto" or categorical_feature is None:
        return pandas_cat_cols
    out: list[int] = []
    for item in categorical_feature:
        if isinstance(item, str):
            if feature_names is None or item not in feature_names:
                raise ValueError(f"categorical_feature {item!r} is not a feature name")
            out.append(feature_names.index(item))
        else:
            out.append(int(item))
    return out


# upstream: C_API_PREDICT_* constants
_C_API_PREDICT_NORMAL = 0
_C_API_PREDICT_RAW_SCORE = 1
_C_API_PREDICT_LEAF_INDEX = 2
_C_API_PREDICT_CONTRIB = 3

_FEATURE_IMPORTANCE_TYPE_MAPPER = {"split": 0, "gain": 1}

# Prediction parameters and their upstream aliases (include/LightGBM/config.h).
_PREDICT_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "num_threads": ("num_threads", "num_thread", "nthread", "nthreads", "n_jobs"),
    "pred_early_stop": ("pred_early_stop",),
    "pred_early_stop_freq": ("pred_early_stop_freq",),
    "pred_early_stop_margin": ("pred_early_stop_margin",),
    "predict_disable_shape_check": ("predict_disable_shape_check",),
}


def _resolve_predict_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, aliases in _PREDICT_PARAM_ALIASES.items():
        for alias in aliases:
            if alias in params:
                out[name] = params[alias]
                break
    return out


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "+")
    return bool(value)


def _load_pandas_categorical(raw: str | None) -> list[list[Any]] | None:
    if raw is None:
        return None
    loaded = json.loads(raw)
    return None if loaded is None else list(loaded)


def _json_default_with_numpy(obj: Any) -> Any:
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _dump_pandas_categorical(pandas_categorical: list[list[Any]] | None) -> str:
    categorical_json = json.dumps(pandas_categorical, default=_json_default_with_numpy)
    return f"\npandas_categorical:{categorical_json}\n"


def _data_from_pandas(data: Any, pandas_categorical: list[list[Any]] | None) -> npt.NDArray[Any]:
    """Map categorical columns to the training categories' codes (NaN for unseen)."""
    import pandas as pd

    data = data.copy(deep=False)
    cat_cols = [
        col
        for col, dtype in zip(data.columns, data.dtypes, strict=True)
        if isinstance(dtype, pd.CategoricalDtype)
    ]
    if cat_cols:
        if pandas_categorical is not None:
            if len(cat_cols) != len(pandas_categorical):
                raise ValueError("train and valid dataset categorical_feature do not match.")
            for col, category in zip(cat_cols, pandas_categorical, strict=True):
                if list(data[col].cat.categories) != list(category):
                    data[col] = data[col].cat.set_categories(category)
        for col in cat_cols:
            codes = data[col].cat.codes.astype(np.float64)
            data[col] = codes.where(codes != -1, np.nan)
    return np.asarray(data.to_numpy(dtype=np.float64, na_value=np.nan))


def _is_pandas_df(data: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(data, pd.DataFrame)


def _is_scipy_sparse(data: Any) -> bool:
    try:
        import scipy.sparse
    except ImportError:  # pragma: no cover
        return False
    return bool(scipy.sparse.issparse(data))


class Dataset:
    """Training data. API-compatible with ``lightgbm.Dataset`` for dense inputs.

    Binning happens on :meth:`construct` (also called by :func:`bankai_lightgbm.train`).
    """

    def __init__(
        self,
        data: Any,
        label: Any = None,
        reference: Dataset | None = None,
        weight: Any = None,
        group: Any = None,
        init_score: Any = None,
        feature_name: list[str] | str = "auto",
        categorical_feature: list[int | str] | str = "auto",
        params: dict[str, Any] | None = None,
        free_raw_data: bool = True,
    ) -> None:
        self.data = data
        self.label = label
        self.reference = reference
        self.weight = weight
        self.group = group
        self.init_score = init_score
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.params: dict[str, Any] = deepcopy(params) if params else {}
        self.free_raw_data = free_raw_data
        self.pandas_categorical: list[list[Any]] | None = None
        self._categorical: list[int] = []
        self._handle: NativeDataset | None = None

    def construct(self) -> Dataset:
        """Bin the data (no-op if already constructed)."""
        if self._handle is None:
            self._handle = self._build()
        return self

    def num_data(self) -> int:
        return self.construct()._handle_or_raise().num_data()

    def num_feature(self) -> int:
        return self.construct()._handle_or_raise().num_feature()

    def _handle_or_raise(self) -> NativeDataset:
        if self._handle is None:  # pragma: no cover - construct() always sets it
            raise LightGBMError("Dataset is not constructed")
        return self._handle

    def _build(self) -> NativeDataset:
        if self.reference is not None:
            raise NotImplementedError("validation datasets (reference=) are not implemented yet")
        if self.init_score is not None:
            raise NotImplementedError("init_score is not implemented yet (roadmap P5)")
        if _is_scipy_sparse(self.data):
            raise NotImplementedError("sparse training input is not implemented yet (roadmap P2)")
        if self.data is None:
            raise ValueError("Cannot construct a Dataset without data")
        feature_names: list[str] | None = None
        pandas_cat_cols: list[int] = []
        if _is_pandas_df(self.data):
            import pandas as pd

            feature_names = [str(c) for c in self.data.columns]
            pandas_cat_cols = [
                i
                for i, dtype in enumerate(self.data.dtypes)
                if isinstance(dtype, pd.CategoricalDtype)
            ]
            self.pandas_categorical = [
                list(self.data[self.data.columns[i]].cat.categories) for i in pandas_cat_cols
            ] or None
            matrix = _data_from_pandas(self.data, None)
        else:
            matrix = np.asarray(self.data, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        n = matrix.shape[0]
        if self.label is None:
            label = np.zeros(n, dtype=np.float64)
        else:
            label = np.asarray(self.label, dtype=np.float64).reshape(-1)
            if label.shape[0] != n:
                raise ValueError(f"Length of label ({label.shape[0]}) is not same with #data ({n})")
        categorical = _categorical_indices(self.categorical_feature, feature_names, pandas_cat_cols)
        self._categorical = categorical
        params = _params_to_strings(self.params)
        if categorical:
            params["categorical_feature"] = ",".join(str(c) for c in categorical)
        weight = None
        if self.weight is not None:
            weight = [float(w) for w in np.asarray(self.weight, dtype=np.float32).reshape(-1)]
        group = None
        if self.group is not None:
            group = [int(g) for g in np.asarray(self.group).reshape(-1)]
        return NativeDataset.from_dense(matrix, label.tolist(), params, categorical, weight, group)


class Booster:
    """Booster in LightGBM. API-compatible with ``lightgbm.Booster``."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        train_set: Any = None,
        model_file: str | Path | None = None,
        model_str: str | None = None,
    ) -> None:
        self.params: dict[str, Any] = deepcopy(params) if params else {}
        self.best_iteration = -1
        self.best_score: dict[str, dict[str, float]] = {}
        self._train_set = train_set
        self._handle: NativeBooster | None = None
        if train_set is not None:
            if model_file is not None or model_str is not None:
                raise TypeError("Booster cannot take both a training set and a model")
            self.pandas_categorical = None
            return
        if model_file is not None:
            self._handle = NativeBooster.from_file(str(model_file))
        elif model_str is not None:
            self._handle = NativeBooster.from_string(model_str)
        else:
            raise TypeError(
                "Need at least one training dataset or model file or model string "
                "to create Booster instance"
            )
        self.pandas_categorical = _load_pandas_categorical(self._native().pandas_categorical())

    def _native(self) -> NativeBooster:
        if self._handle is None:
            raise LightGBMError("Booster has not been trained yet; use lgb.train()")
        return self._handle

    def _train(self, num_boost_round: int) -> None:
        """Train ``num_boost_round`` iterations on the training set (roadmap P5: incremental)."""
        if self._train_set is None:
            raise LightGBMError("Booster has no training set")
        dataset = self._train_set.construct()
        merged = {**self._train_set.params, **self.params}
        if dataset._categorical:
            merged["categorical_feature"] = ",".join(str(c) for c in dataset._categorical)
        merged["num_iterations"] = num_boost_round
        self._handle = NativeBooster.train(_params_to_strings(merged), dataset._handle_or_raise())
        self.pandas_categorical = dataset.pandas_categorical

    def update(self, train_set: Dataset | None = None, fobj: Any = None) -> bool:
        """Train one additional iteration and return whether training has finished."""
        if fobj is not None:
            raise NotImplementedError("custom objectives are not implemented yet (roadmap P4)")
        if train_set is not None:
            if self._train_set is not None and train_set is not self._train_set:
                raise NotImplementedError("changing the training Dataset is not implemented yet")
            self._train_set = train_set
        if self._handle is None:
            if self._train_set is None:
                raise LightGBMError("Booster has no training Dataset")
            dataset = self._train_set.construct()
            merged = {**dataset.params, **self.params}
            if dataset._categorical:
                merged["categorical_feature"] = ",".join(str(c) for c in dataset._categorical)
            self._handle = NativeBooster.create_training(
                _params_to_strings(merged), dataset._handle_or_raise()
            )
            self.pandas_categorical = dataset.pandas_categorical
        return self._native().update()

    def __copy__(self) -> Booster:
        return self.__deepcopy__(None)

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> Booster:
        return Booster(model_str=self.model_to_string(num_iteration=-1))

    def __getstate__(self) -> dict[str, Any]:
        this = self.__dict__.copy()
        this.pop("train_set", None)
        this.pop("valid_sets", None)
        this["_handle"] = self.model_to_string(num_iteration=-1)
        return this

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_str = state.get("_handle", state.get("handle"))
        if isinstance(model_str, str):
            state["_handle"] = NativeBooster.from_string(model_str)
        self.__dict__.update(state)

    def model_from_string(self, model_str: str) -> Booster:
        """Load Booster from a string."""
        self._handle = NativeBooster.from_string(model_str)
        self.pandas_categorical = _load_pandas_categorical(self._native().pandas_categorical())
        return self

    def model_to_string(
        self,
        num_iteration: int | None = None,
        start_iteration: int = 0,
        importance_type: str = "split",
    ) -> str:
        """Save Booster to string (same bytes as ``lightgbm.Booster.model_to_string``)."""
        if num_iteration is None:
            num_iteration = self.best_iteration
        ret = self._native().save_model_to_string(
            int(start_iteration),
            int(num_iteration),
            _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type],
        )
        return ret + _dump_pandas_categorical(self.pandas_categorical)

    def save_model(
        self,
        filename: str | Path,
        num_iteration: int | None = None,
        start_iteration: int = 0,
        importance_type: str = "split",
    ) -> Booster:
        """Save Booster to file."""
        text = self.model_to_string(num_iteration, start_iteration, importance_type)
        Path(filename).write_text(text, encoding="utf-8", newline="")
        return self

    def dump_model(
        self,
        num_iteration: int | None = None,
        start_iteration: int = 0,
        importance_type: str = "split",
        object_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Dump Booster to JSON format."""
        if num_iteration is None:
            num_iteration = self.best_iteration
        raw = self._native().dump_model(
            int(start_iteration),
            int(num_iteration),
            _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type],
        )
        ret: dict[str, Any] = json.loads(raw, object_hook=object_hook)
        ret["pandas_categorical"] = json.loads(
            json.dumps(self.pandas_categorical, default=_json_default_with_numpy)
        )
        return ret

    def feature_importance(
        self, importance_type: str = "split", iteration: int | None = None
    ) -> npt.NDArray[Any]:
        """Get feature importances (int32 for ``split``, float64 for ``gain``)."""
        if iteration is None:
            iteration = self.best_iteration
        type_int = _FEATURE_IMPORTANCE_TYPE_MAPPER[importance_type]
        result = np.array(
            self._native().feature_importance(int(iteration), type_int), dtype=np.float64
        )
        return result.astype(np.int32) if type_int == 0 else result

    def upper_bound(self) -> float:
        """Get upper bound value of a model."""
        return self._native().upper_bound()

    def lower_bound(self) -> float:
        """Get lower bound value of a model."""
        return self._native().lower_bound()

    def get_leaf_output(self, tree_id: int, leaf_id: int) -> float:
        """Get the output of a leaf."""
        return self._native().leaf_value(int(tree_id), int(leaf_id))

    def num_trees(self) -> int:
        return self._native().num_trees()

    def num_model_per_iteration(self) -> int:
        return self._native().num_tree_per_iteration()

    def current_iteration(self) -> int:
        return self._native().current_iteration()

    def num_feature(self) -> int:
        return self._native().num_feature()

    def feature_name(self) -> list[str]:
        return self._native().feature_names()

    def predict(
        self,
        data: Any,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> npt.NDArray[Any]:
        """Make a prediction. Same semantics and output shapes as ``lightgbm.Booster.predict``."""
        if num_iteration is None:
            num_iteration = self.best_iteration if start_iteration <= 0 else -1
        if isinstance(data, (str, Path)):
            raise NotImplementedError("Predicting from a file is not implemented yet (roadmap P2)")
        if _is_pandas_df(data):
            if validate_features:
                names = [str(c) for c in data.columns]
                if names != self.feature_name():
                    raise LightGBMError(
                        f"Expected '{self.feature_name()}' but got '{names}' as feature names"
                    )
            mat = _data_from_pandas(data, self.pandas_categorical)
        else:
            mat = self._to_dense(data)
        if mat.ndim != 2:
            raise ValueError("Input numpy.ndarray or list must be 2 dimensional")
        mat = np.ascontiguousarray(mat, dtype=np.float64)

        predict_type = _C_API_PREDICT_NORMAL
        if raw_score:
            predict_type = _C_API_PREDICT_RAW_SCORE
        if pred_leaf:
            predict_type = _C_API_PREDICT_LEAF_INDEX
        if pred_contrib:
            predict_type = _C_API_PREDICT_CONTRIB

        p = _resolve_predict_params({**self.params, **kwargs})
        preds = self._native().predict_dense(
            mat,
            predict_type,
            int(start_iteration),
            int(num_iteration),
            pred_early_stop=_to_bool(p.get("pred_early_stop", False)),
            pred_early_stop_freq=int(p.get("pred_early_stop_freq", 10)),
            pred_early_stop_margin=float(p.get("pred_early_stop_margin", 10.0)),
            num_threads=int(p.get("num_threads", 0)),
            disable_shape_check=_to_bool(p.get("predict_disable_shape_check", False)),
        )
        nrow = mat.shape[0]
        if pred_leaf:
            return preds.astype(np.int32)
        if preds.size != nrow or pred_contrib:
            return preds
        return preds.reshape(nrow)

    @staticmethod
    def _to_dense(data: Any) -> npt.NDArray[Any]:
        if isinstance(data, np.ndarray):
            return data
        if _is_scipy_sparse(data):
            raise NotImplementedError("Sparse input is not implemented yet (roadmap P6)")
        if isinstance(data, list):
            try:
                return np.array(data)
            except BaseException as err:
                raise ValueError("Cannot convert data list to numpy array.") from err
        raise TypeError(f"Cannot predict data for type {type(data).__name__}")
