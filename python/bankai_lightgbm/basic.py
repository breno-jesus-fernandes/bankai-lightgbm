"""Core objects: ``Booster`` (and, later, ``Dataset``). Mirrors ``lightgbm.basic``."""

from __future__ import annotations

import abc
import ctypes
import json
import warnings
from collections.abc import Callable
from collections.abc import Sequence as _ABCSequence
from copy import deepcopy
from enum import Enum
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, NamedTuple, Union

import narwhals as nw
import narwhals.dependencies as nwd
import narwhals.typing as nwt
import numpy as np
import numpy.typing as npt
import scipy.sparse

from bankai_lightgbm._lib import LightGBMError, NativeBooster, NativeDataset, validate_params
from bankai_lightgbm.compat import PANDAS_INSTALLED
from bankai_lightgbm.compat import pd_DataFrame as pd_DataFrame
from bankai_lightgbm.compat import pd_Series as pd_Series

__all__ = ["Booster", "Dataset", "LightGBMError"]


def _is_numeric(obj: Any) -> bool:
    """Check whether object is a number or not, include numpy number, etc.

    upstream: python-package/lightgbm/basic.py (v4.7.0) ``_is_numeric``
    """
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        # TypeError: obj is not a string or a number
        # ValueError: obj is a string that cannot be parsed as a float
        return False


def _is_numpy_1d_array(data: Any) -> bool:
    """Check whether data is a numpy 1-D array."""
    return isinstance(data, np.ndarray) and len(data.shape) == 1


def _is_numpy_column_array(data: Any) -> bool:
    """Check whether data is a column numpy array."""
    if not isinstance(data, np.ndarray):
        return False
    shape = data.shape
    return len(shape) == 2 and shape[1] == 1


def _cast_numpy_array_to_dtype(array: np.ndarray, dtype: npt.DTypeLike) -> np.ndarray:
    """Cast numpy array to given dtype."""
    if array.dtype == dtype:
        return array
    return array.astype(dtype=dtype, copy=False)


def _is_1d_list(data: Any) -> bool:
    """Check whether data is a 1-D list."""
    return isinstance(data, list) and (not data or _is_numeric(data[0]))


def _is_allowed_numpy_dtype(dtype: type) -> bool:
    float128 = getattr(np, "float128", type(None))
    return issubclass(dtype, (np.integer, np.floating, np.bool_)) and not issubclass(
        dtype, (np.timedelta64, float128)
    )


def _check_for_bad_pandas_dtypes(pandas_dtypes_series: pd_Series) -> None:
    """Raise if any pandas column dtype cannot be converted to a numeric array.

    upstream: python-package/lightgbm/basic.py (v4.7.0) ``_check_for_bad_pandas_dtypes``
    """
    bad_pandas_dtypes = [
        f"{column_name}: {pandas_dtype}"
        for column_name, pandas_dtype in pandas_dtypes_series.items()
        if not _is_allowed_numpy_dtype(pandas_dtype.type)
    ]
    if bad_pandas_dtypes:
        raise ValueError(
            f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}"
        )


def _list_to_1d_numpy(*, data: Any, dtype: npt.DTypeLike, name: str) -> npt.NDArray[Any]:
    """Convert data to numpy 1-D array.

    upstream: python-package/lightgbm/basic.py (v4.7.0) ``_list_to_1d_numpy``
    """
    if _is_numpy_1d_array(data):
        return _cast_numpy_array_to_dtype(data, dtype)
    elif _is_numpy_column_array(data):
        _log_warning("Converting column-vector to 1d array")
        array = data.ravel()
        return _cast_numpy_array_to_dtype(array, dtype)
    elif _is_1d_list(data):
        return np.asarray(data, dtype=dtype)
    elif isinstance(data, pd_Series):
        _check_for_bad_pandas_dtypes(data.to_frame().dtypes)
        return np.asarray(data, dtype=dtype)  # SparseArray should be supported as well
    elif nwd.is_into_series(data) or nwd.is_into_dataframe(data):
        # pyarrow and polars inputs (and one-column frames). upstream streams these to C in
        # ``set_field``; here they are flattened to the same 1-D array.
        frame = nw.from_native(data, allow_series=True)
        if isinstance(frame, nw.DataFrame):
            if frame.shape[1] != 1:
                raise TypeError(
                    f"Wrong type({type(data).__name__}) for {name}.\n"
                    "It should be list, numpy 1-D array or pandas Series"
                )
            frame = frame.get_column(frame.columns[0])
        return _cast_numpy_array_to_dtype(frame.to_numpy().reshape(-1), dtype)
    else:
        raise TypeError(
            f"Wrong type({type(data).__name__}) for {name}.\nIt should be list, numpy 1-D array or pandas Series"
        )


def _custom_derivative(data: Any, name: str) -> npt.NDArray[np.float32]:
    """Match upstream basic.py::_list_to_1d_numpy for custom derivatives (v4.7.0)."""
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            return np.asarray(data, dtype=np.float32)
        if data.ndim == 2 and data.shape[1] == 1:
            warnings.warn("Converting column-vector to 1d array", UserWarning, stacklevel=3)
            return np.asarray(data.ravel(), dtype=np.float32)
    elif (isinstance(data, list) and all(np.isscalar(value) for value in data)) or type(
        data
    ).__name__ == "Series":
        return np.asarray(data, dtype=np.float32)
    raise TypeError(
        f"Wrong type({type(data).__name__}) for {name}.\n"
        "It should be list, numpy 1-D array or pandas Series"
    )


def _param_value_str(value: Any) -> str:
    """Parameter values as the C API takes them (``lightgbm`` formats them the same way).

    Like upstream's ``_param_dict_to_str``/``_to_string`` (basic.py v4.7.0), a list inside a
    list is written as ``[a,b]``, so ``interaction_constraints=[[0, 1], [2]]`` becomes
    ``[0,1],[2]`` rather than one flattened group.
    """
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, list | tuple | set | frozenset | np.ndarray):
        return ",".join(_param_list_item_str(v) for v in value)
    return str(value)


def _param_list_item_str(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    return _param_value_str(value)


def _params_to_strings(params: dict[str, Any]) -> dict[str, str]:
    return {str(k): _param_value_str(v) for k, v in params.items() if v is not None}


@cache
def _known_parameter_names() -> frozenset[str]:
    """Every parameter name and alias the native library accepts."""
    from bankai_lightgbm._lib import param_aliases

    aliases = param_aliases()
    return frozenset(aliases) | frozenset(a for values in aliases.values() for a in values)


def _log_unknown_params(params: dict[str, Any]) -> None:
    """Warn about parameter names the library does not know.

    upstream: src/io/config.cpp (v4.7.0) ``Config::Set`` logs ``Unknown parameter: <key>``; the
    native log goes through ``_LOGGER.info`` and is printed only when ``verbosity >= 0``.
    """
    if int(params.get("verbosity", params.get("verbose", 1))) < 0:
        return
    known = _known_parameter_names()
    for key in params:
        if str(key) not in known:
            _log_info(f"[LightGBM] [Warning] Unknown parameter: {key}")


def _validate_feature_names(booster_names: list[str], data_names: list[str]) -> None:
    """Check prediction data's column names against the model's (upstream ``validate_features``).

    upstream: src/c_api.cpp (v4.7.0) ``LGBM_BoosterValidateFeatureNames``.
    """
    if len(booster_names) != len(data_names):
        raise LightGBMError(
            f"Model was trained on {len(booster_names)} features, "
            f"but got {len(data_names)} input features to predict."
        )
    for i, (expected, found) in enumerate(zip(booster_names, data_names, strict=True)):
        if expected != found:
            raise LightGBMError(f"Expected '{expected}' at position {i} but found '{found}'")


def _typed_param_value(text: str) -> Any:
    """Type a value from the model text's ``parameters:`` block (bool, int, float, list or str)."""
    if text in ("true", "false"):
        return text == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if "," in text and "[" not in text:
        return [_typed_param_value(v) for v in text.split(",")]
    return text


def _table_column_names(data: Any) -> list[str] | None:
    """Column names of a pyarrow Table or a polars DataFrame, else ``None``."""
    module = type(data).__module__
    if module.startswith("pyarrow") and hasattr(data, "column_names"):
        return [str(name) for name in data.column_names]
    if module.startswith("polars") and hasattr(data, "columns") and hasattr(data, "schema"):
        return [str(name) for name in data.columns]
    return None


# upstream: include/LightGBM/config.h (v4.7.0) categorical_feature aliases
_CATEGORICAL_FEATURE_ALIASES = (
    "categorical_feature",
    "categorical_column",
    "cat_feature",
    "cat_column",
    "categorical_features",
)


def _categorical_from_params(params: dict[str, Any], feature_names: list[str] | None) -> list[int]:
    """Categorical columns given through a parameter (or one of its aliases) instead of the argument."""
    for key in _CATEGORICAL_FEATURE_ALIASES:
        value = params.get(key)
        if value is None or value == "" or value == "auto":
            continue
        items = list(value) if isinstance(value, (list, tuple)) else str(value).split(",")
        normalized: list[int | str] = [
            int(item) if isinstance(item, str) and item.strip().isdigit() else item
            for item in items
        ]
        return _categorical_indices(normalized, feature_names, [])
    return []


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


# Rows per native call when predicting contributions from a sparse matrix (bounds the dense
# per-block output; the stacked result stays sparse).
_SPARSE_PREDICT_BLOCK_ROWS = 16384

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


def _remap_positions(position: Any, num_data: int) -> npt.NDArray[np.int32]:
    """Remap raw position ids to dense int32 indices, numbered in first-seen order.

    upstream: src/io/metadata.cpp (v4.7.0) Metadata::SetPosition
    """
    values = np.asarray(position).reshape(-1)
    if values.shape[0] != num_data:
        raise LightGBMError(
            f"Positions size ({values.shape[0]}) doesn't match data size ({num_data})"
        )
    _, first_index, inverse = np.unique(values, return_index=True, return_inverse=True)
    rank = np.empty(first_index.shape[0], dtype=np.int32)
    rank[np.argsort(first_index, kind="stable")] = np.arange(first_index.shape[0], dtype=np.int32)
    return rank[inverse.reshape(-1)]


def _drop_all_ones_weight(weight: Any) -> Any:
    """Return ``None`` for a weight whose values are all one, as upstream does.

    upstream: python-package/lightgbm/basic.py ``Dataset.set_weight``
    (``np.all(weight == 1)`` and ``(series == 1).all()`` set the weight to ``None``).
    """
    if weight is None or isinstance(weight, (list, tuple)):
        # upstream compares only arrays and Series; a list's ``== 1`` is a scalar False.
        return weight
    if np.all(np.asarray(weight) == 1):
        return None
    return weight


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


def _pandas_to_matrix(data: Any, pandas_categorical: list[list[Any]] | None) -> npt.NDArray[Any]:
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


def _data_from_pandas(
    *,
    data: Any,
    feature_name: Any,
    categorical_feature: Any,
    pandas_categorical: list[list[Any]] | None,
) -> tuple[npt.NDArray[Any], Any, Any, list[list[Any]] | None]:
    """Convert a DataFrame; returns ``(matrix, feature_name, categorical_feature, pandas_categorical)``.

    upstream: python-package/lightgbm/basic.py (v4.7.0) ``_data_from_pandas``. ``"auto"`` names
    the columns, and ``"auto"`` categorical features are the unordered category columns.
    """
    import pandas as pd

    cat_cols = [
        col
        for col, dtype in zip(data.columns, data.dtypes, strict=True)
        if isinstance(dtype, pd.CategoricalDtype)
    ]
    cat_cols_not_ordered = [col for col in cat_cols if not data[col].cat.ordered]
    if pandas_categorical is None:
        pandas_categorical = [list(data[col].cat.categories) for col in cat_cols]
    matrix = _pandas_to_matrix(data, pandas_categorical)
    if feature_name == "auto":
        feature_name = [str(col) for col in data.columns]
    if categorical_feature == "auto":
        categorical_feature = [list(data.columns).index(col) for col in cat_cols_not_ordered]
    return matrix, feature_name, categorical_feature, pandas_categorical


def _init_score_values(init_score: Any, num_data: int) -> list[float]:
    """Flatten init scores class-major, as upstream's Python ``set_field`` does.

    Two-dimensional ``(num_data, num_class)`` input is raveled in Fortran order.
    """
    values = np.asarray(init_score, dtype=np.float64)
    flat = values.ravel(order="F")
    # upstream: src/io/metadata.cpp (v4.7.0) SetInitScoresFromIterator.
    if flat.size % num_data != 0:
        raise LightGBMError("Initial score size doesn't match data size")
    return flat.tolist()


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


# numpy array, or a scipy.sparse matrix / list of them for pred_contrib on sparse input
# (upstream: ``_LGBM_PredictReturnType``; scipy is untyped here, hence ``Any``).
_PredictReturnType = npt.NDArray[Any] | Any


def _dataset_sparse(data: Any) -> Any:
    """The CSR/CSC matrix a Dataset is built from, or ``None`` for dense inputs.

    Mirrors upstream ``Dataset._lazy_init``: ``csr_matrix``/``csc_matrix`` are used as-is
    after upstream's dtype and length checks (``__init_from_csr``/``__init_from_csc``);
    other scipy.sparse inputs are converted with ``csr_matrix`` (no warning, unlike
    prediction), failures raising ``TypeError("Cannot initialize Dataset from ...")``.
    """
    if not _is_scipy_sparse(data):
        return None
    import scipy.sparse

    if not isinstance(data, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        try:
            converted = scipy.sparse.csr_matrix(data)
            _check_sparse_arrays(converted)
        except BaseException as err:
            raise TypeError(f"Cannot initialize Dataset from {type(data).__name__}") from err
        return converted
    _check_sparse_arrays(data)
    return data


def _check_sparse_arrays(cs: Any) -> None:
    if len(cs.indices) != len(cs.data):
        raise ValueError(f"Length mismatch: {len(cs.indices)} vs {len(cs.data)}")
    _c_int_dtype_check(cs.indptr)
    _c_float_dtype_check(cs.data)


def _internal_rows(data: Any) -> Any:
    """Rows to re-predict where upstream reads its internal scores (custom objectives and
    metrics): other scipy.sparse formats become ``csr_matrix`` silently, since upstream
    never converts (or warns) there."""
    if _is_scipy_sparse(data):
        import scipy.sparse

        if not isinstance(data, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            return scipy.sparse.csr_matrix(data)
    return data


class _ConfigAliases:
    """Parameter aliases from the table generated from upstream's ``config.h``.

    upstream: python-package/lightgbm/basic.py (v4.7.0) ``_ConfigAliases``; aliases are
    ordered like ``Config::DumpAliases`` (``SortAlias``: shorter first, then lexicographic),
    after the main name.
    """

    aliases: dict[str, list[str]] | None = None
    raw_aliases: dict[str, list[str]] | None = None

    @classmethod
    def get_sorted(cls, name: str) -> list[str]:
        if cls.aliases is None:
            from bankai_lightgbm._lib import param_aliases

            cls.aliases = {
                key: [key, *sorted(values, key=lambda a: (len(a), a))]
                for key, values in param_aliases().items()
            }
        return cls.aliases.get(name, [name])

    @classmethod
    def get_by_alias(cls, *args: str) -> set[str]:
        """The given names plus every parameter they are an alias of (upstream ``get_by_alias``)."""
        if cls.raw_aliases is None:
            from bankai_lightgbm._lib import param_aliases

            cls.raw_aliases = param_aliases()
        ret = set(args)
        for arg in args:
            for aliases in cls.raw_aliases.values():
                if arg in aliases:
                    ret.update(aliases)
                    break
        return ret

    @classmethod
    def get(cls, *args: str) -> set[str]:
        """All names and aliases of the given parameters (upstream ``_ConfigAliases.get``)."""
        ret: set[str] = set()
        for arg in args:
            ret.update(cls.get_sorted(arg))
        return ret


def _choose_param_value(
    main_param_name: str, params: dict[str, Any], default_value: Any
) -> dict[str, Any]:
    """One value for ``main_param_name`` with its aliases removed; the main name wins, then the
    first alias found in sorted order, then ``default_value``.

    upstream: python-package/lightgbm/basic.py (v4.7.0) ``_choose_param_value``.
    """
    params = deepcopy(params)
    aliases = [a for a in _ConfigAliases.get_sorted(main_param_name) if a != main_param_name]
    if main_param_name in params:
        for alias in aliases:
            params.pop(alias, None)
        return params
    for alias in aliases:
        if alias in params:
            params[main_param_name] = params[alias]
            break
    if main_param_name in params:
        for alias in aliases:
            params.pop(alias, None)
        return params
    params[main_param_name] = default_value
    return params


class _TempFile:
    """Proxy class to workaround errors on Windows (upstream ``_TempFile``)."""

    def __enter__(self) -> _TempFile:
        with NamedTemporaryFile(prefix="lightgbm_tmp_", delete=True) as f:
            self.name = f.name
            self.path = Path(self.name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.path.is_file():
            self.path.unlink()


class _DummyLogger:
    """upstream's default logger: ``info`` prints, ``warning`` emits a ``UserWarning``."""

    def info(self, msg: str) -> None:
        print(msg)

    def warning(self, msg: str) -> None:
        warnings.warn(msg, stacklevel=3)


_LOGGER: Any = _DummyLogger()
_INFO_METHOD_NAME = "info"
_WARNING_METHOD_NAME = "warning"


# upstream: python-package/lightgbm/basic.py (v4.7.0) register_logger
def _has_method(logger: Any, method_name: str) -> bool:
    return callable(getattr(logger, method_name, None))


def register_logger(
    logger: Any,
    info_method_name: str = "info",
    warning_method_name: str = "warning",
) -> None:
    """Register custom logger.

    Parameters
    ----------
    logger : Any
        Custom logger.
    info_method_name : str, optional (default="info")
        Method used to log info messages.
    warning_method_name : str, optional (default="warning")
        Method used to log warning messages.
    """
    if not _has_method(logger, info_method_name) or not _has_method(logger, warning_method_name):
        raise TypeError(
            f"Logger must provide '{info_method_name}' and '{warning_method_name}' method"
        )

    global _LOGGER, _INFO_METHOD_NAME, _WARNING_METHOD_NAME
    _LOGGER = logger
    _INFO_METHOD_NAME = info_method_name
    _WARNING_METHOD_NAME = warning_method_name


def _native_log(level: str, msg: str) -> None:
    """Route a warning or info record from the native library, as upstream's log callback does."""
    label = "Warning" if level == "warning" else "Info"
    _log_info(f"[LightGBM] [{label}] {msg}")


def _log_info(msg: str) -> None:
    getattr(_LOGGER, _INFO_METHOD_NAME)(msg)


def _log_warning(msg: str) -> None:
    getattr(_LOGGER, _WARNING_METHOD_NAME)(msg)


def _c_float_dtype_check(data: npt.NDArray[Any]) -> None:
    """Accept float32/float64 values like upstream's ``_c_float_array``."""
    if data.dtype not in (np.float32, np.float64):
        raise TypeError(f"Expected np.float32 or np.float64, met type({data.dtype})")


def _c_int_dtype_check(data: npt.NDArray[Any]) -> None:
    """Accept int32/int64 offsets like upstream's ``_c_int_array``."""
    if data.dtype not in (np.int32, np.int64):
        raise TypeError(f"Expected np.int32 or np.int64, met type({data.dtype})")


# Type aliases and constants, as upstream python-package/lightgbm/basic.py (v4.7.0, MIT).
_ctypes_float_ptr = Union[
    "ctypes._Pointer[ctypes.c_float]",
    "ctypes._Pointer[ctypes.c_double]",
]
_ctypes_float_array = Union[
    "ctypes.Array[ctypes._Pointer[ctypes.c_float]]",
    "ctypes.Array[ctypes._Pointer[ctypes.c_double]]",
]
_LGBM_EvalFunctionResultType = tuple[str, float, bool]
_LGBM_BoosterBestScoreType = dict[str, dict[str, float]]
_LGBM_CategoricalFeatureConfiguration = Union[list[str], list[int], "Literal['auto']"]
_LGBM_FeatureNameConfiguration = Union[list[str], "Literal['auto']"]
_LGBM_GroupType = Union[
    list[float],
    list[int],
    np.ndarray,
    pd_Series,
    nwt.IntoSeries,
]
# 'position' intentionally does not support 'list' inputs
# ref: https://github.com/lightgbm-org/LightGBM/pull/5929#discussion_r1262646998
_LGBM_PositionType = Union[
    np.ndarray,
    pd_Series,
    nwt.IntoSeries,
]
_LGBM_InitScoreType = Union[
    list[float],
    list[list[float]],
    np.ndarray,
    pd_Series,
    pd_DataFrame,
    nwt.IntoSeries,
    nwt.IntoDataFrame,
]
_LGBM_TrainDataType = Union[
    str,
    Path,
    np.ndarray,
    pd_DataFrame,
    scipy.sparse.spmatrix,
    "_ABCSequence[Any]",
    list["_ABCSequence[Any]"],
    list[np.ndarray],
    nwt.IntoDataFrame,
]
_LGBM_LabelType = Union[
    list[float],
    list[int],
    np.ndarray,
    pd_Series,
    pd_DataFrame,
    nwt.IntoSeries,
    nwt.IntoDataFrame,
]
_LGBM_PredictDataType = Union[
    str,
    Path,
    np.ndarray,
    pd_DataFrame,
    scipy.sparse.spmatrix,
    nwt.IntoDataFrame,
]
_LGBM_PredictReturnType = Union[
    np.ndarray,
    scipy.sparse.spmatrix,
    list[scipy.sparse.spmatrix],
]
_LGBM_PredictSparseReturnType = Union[
    scipy.sparse.spmatrix,
    list[scipy.sparse.spmatrix],
]
_LGBM_WeightType = Union[
    list[float],
    list[int],
    np.ndarray,
    pd_Series,
    nwt.IntoSeries,
]
_LGBM_SetFieldType = Union[
    list[list[float]],
    list[list[int]],
    list[float],
    list[int],
    np.ndarray,
    pd_Series,
    pd_DataFrame,
    nwt.IntoDataFrame,
]

ZERO_THRESHOLD = 1e-35

_MULTICLASS_OBJECTIVES = {"multiclass", "multiclassova", "multiclass_ova", "ova", "ovr", "softmax"}


class LGBMDeprecationWarning(FutureWarning):
    """Custom deprecation warning."""

    pass


def _is_zero(x: float) -> bool:
    return -ZERO_THRESHOLD <= x <= ZERO_THRESHOLD


class _MissingType(Enum):
    NONE = "None"
    NAN = "NaN"
    ZERO = "Zero"


# upstream: python-package/lightgbm/basic.py (v4.7.0) Sequence
class Sequence(abc.ABC):
    """
    Generic data access interface.

    Object should support the following operations:

    .. code-block::

        # Get total row number.
        >>> len(seq)
        # Random access by row index. Used for data sampling.
        >>> seq[10]
        # Range data access. Used to read data in batch when constructing Dataset.
        >>> seq[0:100]
        # Optionally specify batch_size to control range data read size.
        >>> seq.batch_size

    - With random access, **data sampling does not need to go through all data**.
    - With range data access, there's **no need to read all data into memory thus reduce memory usage**.

    .. versionadded:: 3.3.0

    Attributes
    ----------
    batch_size : int
        Default size of a batch.
    """

    batch_size = 4096  # Defaults to read 4K rows in each batch.

    @abc.abstractmethod
    def __getitem__(self, idx: int | slice | list[int]) -> np.ndarray:
        """Return data for given row index.

        A basic implementation should look like this:

        .. code-block:: python

            if isinstance(idx, numbers.Integral):
                return self._get_one_line(idx)
            elif isinstance(idx, slice):
                return np.stack([self._get_one_line(i) for i in range(idx.start, idx.stop)])
            elif isinstance(idx, list):
                # Only required if using ``Dataset.subset()``.
                return np.array([self._get_one_line(i) for i in idx])
            else:
                raise TypeError(f"Sequence index must be integer, slice or list, got {type(idx).__name__}")

        Parameters
        ----------
        idx : int, slice[int], list[int]
            Item index.

        Returns
        -------
        result : numpy 1-D array or numpy 2-D array
            1-D array if idx is int, 2-D array if idx is slice or list.
        """
        raise NotImplementedError("Sub-classes of lightgbm.Sequence must implement __getitem__()")

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return row count of this sequence."""
        raise NotImplementedError("Sub-classes of lightgbm.Sequence must implement __len__()")


# upstream: python-package/lightgbm/basic.py (v4.7.0) EvalResult
class EvalResult(NamedTuple):
    """
    Result from computing an evaluation metric on a dataset.

    In ``lightgbm<4.7.0``, evaluation results were stored in tuples like this:

      * train(): ``(dataset_name, metric_name, metric_value, maximize)``
      * cv(): ``(dataset_name, metric_name, mean(metric_value), maximize, std_dev(metric_value))``

    Parameters
    ----------
    dataset_name : str
        Unique identifier for the dataset this result was computed on.
    metric_name : str
        Unique identifier for the metric (e.g. "rmse").
    metric_value : float
        Value of the evaluation metric.
    maximize : bool
        Are higher values better? e.g. ``True`` for AUC and ``False`` for binary error.
    metric_std_dev : float or None
        If not ``None``, the standard deviation of metric values computed over a range of results.
        For example, used when aggregating over cross-validation folds in ``cv()``.
    """

    dataset_name: str
    metric_name: str
    metric_value: float
    maximize: bool
    metric_std_dev: float | None = None

    def __len__(self) -> int:
        if not self.is_cv_result():
            return 4
        else:
            return 5

    def __iter__(self) -> Any:
        i = 0
        while i < len(self):
            yield getattr(self, self._fields[i])
            i += 1

    def is_cv_result(self) -> bool:
        """
        Whether the result was created by ``cv()``.

        If ``True``:

          * ``metric_value`` = mean of ``metric_name`` over CV folds
          * ``metric_std_dev`` = standard deviation of ``metric_name`` over CV folds
        """
        return self.metric_std_dev is not None


def _is_sequence_data(data: Any) -> bool:
    """Whether data is a ``Sequence`` or a list holding one (upstream ``_is_list_of_sequences``)."""
    return isinstance(data, Sequence) or (
        isinstance(data, list) and any(isinstance(item, Sequence) for item in data)
    )


def _rows_from_sequences(data: Any) -> npt.NDArray[np.float64]:
    """Read all rows of a ``Sequence`` (or of a list of Sequences and arrays) in batches.

    upstream reads the same batches but streams them into its bin finder; here the rows are
    gathered into one matrix, so the memory saving of ``Sequence`` is not kept.
    """
    parts: list[npt.NDArray[np.float64]] = []
    for source in data if isinstance(data, list) else [data]:
        if isinstance(source, np.ndarray):
            parts.append(np.asarray(source, dtype=np.float64).reshape(len(source), -1))
            continue
        n_rows = len(source)
        batch = getattr(source, "batch_size", None) or Sequence.batch_size
        for start in range(0, n_rows, batch):
            stop = min(start + batch, n_rows)
            parts.append(np.asarray(source[start:stop], dtype=np.float64).reshape(stop - start, -1))
    if not parts:
        raise ValueError("Cannot construct a Dataset from empty data")
    return np.vstack(parts)


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
        categorical_feature: _ABCSequence[int | str] | str = "auto",
        params: dict[str, Any] | None = None,
        free_raw_data: bool = True,
        position: Any = None,
    ) -> None:
        self.data = data
        self.position = position
        self._position: npt.NDArray[np.int32] | None = None
        # a subset keeps its parent's dense ids, as upstream's Metadata::Init does
        self._inherited_position: npt.NDArray[np.int32] | None = None
        self.label = label
        self.reference = reference
        self.weight = _drop_all_ones_weight(weight)
        self.group = group
        self.init_score = init_score
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.params: dict[str, Any] = deepcopy(params) if params else {}
        self.free_raw_data = free_raw_data
        self.pandas_categorical: list[list[Any]] | None = None
        self._categorical: list[int] = []
        # upstream: True when feature names came from the data or `feature_name` (sklearn reads it).
        self._has_non_default_feature_names: bool = False
        self._handle: NativeDataset | None = None
        self._evaluation_handle: NativeDataset | None = None

    def construct(self) -> Dataset:
        """Bin the data (no-op if already constructed)."""
        if self._handle is None:
            self._handle = self._build()
            # upstream: Dataset._lazy_init ends with ``self.feature_name = self.get_feature_name()``
            self.feature_name = self.get_feature_name()
        return self

    def num_data(self) -> int:
        return self.construct()._handle_or_raise().num_data()

    def num_feature(self) -> int:
        return self.construct()._handle_or_raise().num_feature()

    def feature_num_bin(self, feature: int | str) -> int:
        """Number of bins of a feature given by index or name (0 for features that are not used).

        upstream: python-package/lightgbm/basic.py (v4.7.0) ``Dataset.feature_num_bin``
        """
        self.construct()
        if isinstance(feature, str):
            if feature not in self.feature_name:
                raise LightGBMError(f"Could not find feature name {feature} in the dataset")
            feature = self.feature_name.index(feature)
        return self._handle_or_raise().feature_num_bin(int(feature))

    def create_valid(
        self,
        data: Any,
        label: Any = None,
        weight: Any = None,
        group: Any = None,
        init_score: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Dataset:
        """Create a validation Dataset that uses this Dataset's bin boundaries (upstream ``create_valid``)."""
        return Dataset(
            data,
            label=label,
            reference=self,
            weight=weight,
            group=group,
            init_score=init_score,
            params=params,
        )

    def get_params(self) -> dict[str, Any]:
        """Return a copy of the parameters given to this Dataset (upstream ``get_params``)."""
        return deepcopy(self.params)

    def get_data(self) -> Any:
        """Return the raw data this Dataset was created from (upstream ``get_data``)."""
        return self.data

    def set_reference(self, reference: Dataset) -> Dataset:
        """Use ``reference``'s bin mappers, categorical features and feature names (upstream ``set_reference``).

        A Dataset that is already binned is rebinned from its raw data, which must still be kept.
        upstream: python-package/lightgbm/basic.py (v4.7.0) ``Dataset.set_reference``
        """
        if self.reference is reference:
            return self
        if self._handle is not None or self._evaluation_handle is not None:
            if self.data is None:
                raise LightGBMError(
                    "Cannot set reference after freed raw data, "
                    "set free_raw_data=False when construct Dataset to avoid this."
                )
            self._handle = None
            self._evaluation_handle = None
        self.categorical_feature = reference.categorical_feature
        if isinstance(reference.feature_name, list):
            self.feature_name = list(reference.feature_name)
            self._has_non_default_feature_names = True
        self.reference = reference
        return self

    def set_feature_name(self, feature_name: Any) -> Dataset:
        """Set the feature names (upstream ``set_feature_name``); ``"auto"`` leaves them unset."""
        if isinstance(feature_name, str) and feature_name == "auto":
            return self
        names = [str(name) for name in feature_name]
        if self._handle is not None and len(names) != self.num_feature():
            raise ValueError(
                f"Length of feature_name({len(names)}) and num_feature({self.num_feature()}) don't match"
            )
        self.feature_name = names
        self._has_non_default_feature_names = True
        return self

    def get_feature_name(self) -> list[str]:
        """Return the feature names (upstream ``get_feature_name``)."""
        if isinstance(self.feature_name, list):
            return [str(name) for name in self.feature_name]
        if _is_pandas_df(self.data):
            return [str(c) for c in self.data.columns]
        return [f"Column_{i}" for i in range(self.num_feature())]

    def set_field(self, field_name: str, data: Any) -> Dataset:
        """Set a field by name: ``label``, ``weight``, ``init_score`` or ``group`` (upstream ``set_field``)."""
        setters = {
            "label": self.set_label,
            "weight": self.set_weight,
            "init_score": self.set_init_score,
            "group": self.set_group,
            "position": self.set_position,
        }
        if field_name not in setters:
            raise NotImplementedError(f"field '{field_name}' is not supported yet (roadmap P2)")
        setters[field_name](data)
        return self

    def get_field(self, field_name: str) -> npt.NDArray[Any] | None:
        """Get a field by name; ``group`` returns the query boundaries (upstream ``get_field``)."""
        if field_name == "label":
            return self.get_label()
        if field_name == "weight":
            return self.get_weight()
        if field_name == "init_score":
            return self.get_init_score()
        if field_name == "position":
            if self.position is None:
                return None
            if self._position is None:
                self.construct()
            return self._position
        if field_name == "group":
            group = self.get_group()
            if group is None:
                return None
            return np.concatenate(([0], np.cumsum(group))).astype(np.int32)
        raise NotImplementedError(f"field '{field_name}' is not supported yet (roadmap P2)")

    def get_label(self) -> npt.NDArray[np.float32] | None:
        """Return the dataset labels as a one-dimensional float32 array.

        Once constructed, the labels are read back from the native Dataset, as upstream's
        ``set_label`` does, so non-finite values come back sanitized (NaN -> 0, +-1e38).
        """
        if self._handle is not None:
            return np.asarray(self._handle.label(), dtype=np.float32)
        if self.label is None:
            return None
        return np.asarray(self.label, dtype=np.float32).reshape(-1)

    def subset(self, used_indices: Any, params: dict[str, Any] | None = None) -> Dataset:
        """Select rows while reusing this Dataset's bin mappers."""
        self.construct()
        indices = np.sort(np.asarray(used_indices, dtype=np.int64).reshape(-1))
        # upstream: src/c_api.cpp (v4.7.0) LGBM_DatasetGetSubset.
        if indices.size == 0:
            raise LightGBMError("Check failed: (num_used_row_indices) > (0)")
        num_rows = self.num_data()
        invalid = np.flatnonzero((indices < 0) | (indices >= num_rows))
        if invalid.size:
            position = int(invalid[0])
            raise LightGBMError(
                "[Used indices of subset]: does not tolerate element "
                f"[#{position} = {indices[position]}] outside [0, {num_rows - 1}]"
            )
        native = self._handle_or_raise().subset(indices.tolist())
        if _is_pandas_df(self.data):
            subset_data = self.data.iloc[indices]
        elif _is_scipy_sparse(self.data):
            subset_data = self.data.tocsr()[indices]
        elif type(self.data).__module__.startswith("pyarrow") and hasattr(self.data, "take"):
            # upstream: Dataset.get_data slices a pyarrow Table or polars DataFrame, keeping its type.
            subset_data = self.data.take(indices)
        elif _table_column_names(self.data) is not None:
            subset_data = self.data[indices]
        else:
            subset_data = np.asarray(self.data)[indices]

        def take(values: Any) -> Any:
            if values is None:
                return None
            return np.asarray(values)[indices]

        subset_group = None
        if self.group is not None:
            group_sizes = [int(size) for size in np.asarray(self.group).reshape(-1)]
            group_ends = np.cumsum(group_sizes)
            subset_group = []
            current_group = -1
            current_size = 0
            for row in indices:
                group_index = int(np.searchsorted(group_ends, row, side="right"))
                if group_index != current_group:
                    if current_size:
                        subset_group.append(current_size)
                    current_group = group_index
                    current_size = 0
                current_size += 1
            if current_size:
                subset_group.append(current_size)

        subset = Dataset(
            subset_data,
            label=take(self.label),
            weight=take(self.weight),
            init_score=take(self.init_score),
            group=subset_group,
            position=take(self.position),
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            params={**self.params, **(params or {})},
            free_raw_data=self.free_raw_data,
        )
        if self._position is not None:
            subset._inherited_position = self._position[indices]
        subset.pandas_categorical = self.pandas_categorical
        subset._categorical = self._categorical.copy()
        subset._handle = native
        return subset

    def save_binary(self, filename: str | Path) -> Dataset:
        """Save the constructed Dataset to a binary file (upstream ``LGBM_DatasetSaveBinary``).

        The file is byte-identical to the one upstream writes for the same data and params.
        """
        self.construct()._handle_or_raise().save_binary(str(filename))
        return self

    def _handle_or_raise(self) -> NativeDataset:
        if self._handle is None:  # pragma: no cover - construct() always sets it
            raise LightGBMError("Dataset is not constructed")
        return self._handle

    def get_weight(self) -> npt.NDArray[np.float32] | None:
        """Return the sample weights as a one-dimensional float32 array (read back from the
        native Dataset once constructed, sanitized like upstream)."""
        if self._handle is not None:
            native = self._handle.weight()
            return None if native is None else np.asarray(native, dtype=np.float32)
        if self.weight is None:
            return None
        return np.asarray(self.weight, dtype=np.float32).reshape(-1)

    def get_group(self) -> npt.NDArray[np.int32] | None:
        """Return the query group sizes as a one-dimensional int32 array."""
        if self.group is None:
            return None
        return np.asarray(self.group, dtype=np.int32).reshape(-1)

    def get_init_score(self) -> npt.NDArray[np.float64] | None:
        """Return init scores; multiclass scores are ``(num_data, num_class)``."""
        if self.init_score is None:
            return None
        values = np.asarray(self.init_score, dtype=np.float64).ravel(order="F")
        num_data = self.num_data()
        num_classes = values.size // num_data if num_data else 0
        if num_classes > 1:
            return values.reshape((num_data, num_classes), order="F")
        return values

    def _update_native_fields(
        self,
        *,
        label: list[float] | None = None,
        weight: list[float] | None = None,
        group: list[int] | None = None,
    ) -> None:
        """Replace fields on the constructed handles; ``None`` leaves a field unchanged."""
        for attribute in ("_handle", "_evaluation_handle"):
            handle = getattr(self, attribute)
            if handle is not None:
                setattr(
                    self,
                    attribute,
                    handle.with_fields(label=label, weight=weight, group=group),
                )

    def set_label(self, label: Any) -> Dataset:
        """Set the label; a constructed Dataset is updated in place."""
        self.label = label
        if self._handle is not None or self._evaluation_handle is not None:
            values = np.asarray(label, dtype=np.float64).reshape(-1).tolist()
            self._update_native_fields(label=values)
        return self

    def set_weight(self, weight: Any) -> Dataset:
        """Set the sample weights; a constructed Dataset is updated in place."""
        weight = _drop_all_ones_weight(weight)
        self.weight = weight
        if self._handle is not None or self._evaluation_handle is not None:
            values = (
                []
                if weight is None
                else [float(w) for w in np.asarray(weight, np.float32).reshape(-1)]
            )
            self._update_native_fields(weight=values)
        return self

    def set_position(self, position: Any) -> Dataset:
        """Set the position of each row (ranking position bias); remapped to dense int32 ids."""
        self.position = position
        self._position = None
        if position is not None and (
            self._handle is not None or self._evaluation_handle is not None
        ):
            self._position = _remap_positions(position, self.num_data())
            self.position = self._position
        return self

    def _dump_text(self, filename: str | Path) -> None:
        """Write the bins of every feature in every row (upstream ``Dataset._dump_text``)."""
        self.construct()
        self._handle_or_raise().dump_text(str(filename))

    def get_position(self) -> npt.NDArray[np.int32] | None:
        """Return the position of each row as dense int32 ids (upstream ``get_position``)."""
        return self.get_field("position")

    def set_group(self, group: Any) -> Dataset:
        """Set query group sizes; a constructed Dataset is updated in place."""
        self.group = group
        if self._handle is not None or self._evaluation_handle is not None:
            sizes = [int(g) for g in np.asarray(group).reshape(-1)] if group is not None else []
            num_data = self.num_data()
            if sizes and sum(sizes) != num_data:
                raise LightGBMError(
                    f"Sum of query counts ({num_data}) differs from the length of #data "
                    f"({sum(sizes)})"
                )
            self._update_native_fields(group=sizes)
        return self

    def set_init_score(self, init_score: Any) -> Dataset:
        """Set the init score; constructed handles are updated in place."""
        self.init_score = init_score
        flat = _init_score_values(init_score, self.num_data()) if init_score is not None else []
        if self._handle is not None:
            self._handle = self._handle.with_init_score(flat)
        if self._evaluation_handle is not None:
            self._evaluation_handle = self._evaluation_handle.with_init_score(flat)
        return self

    def _construct_for_eval(self) -> NativeDataset:
        if self._evaluation_handle is None:
            if self._handle is not None:
                return self._handle
            self._evaluation_handle = self._build(evaluation=True)
        return self._evaluation_handle

    def _build(self, *, evaluation: bool = False) -> NativeDataset:
        if self.reference is not None:
            self.reference.construct()
        if self.data is None:
            raise ValueError("Cannot construct a Dataset without data")
        feature_names: list[str] | None = None
        pandas_cat_cols: list[int] = []
        # upstream: _data_from_pandas (v4.7.0) takes only the unordered columns for 'auto'.
        pandas_auto_cat_cols: list[int] = []
        sparse = _dataset_sparse(self.data)
        matrix: npt.NDArray[np.float64] | None = None
        if sparse is not None:
            n = int(sparse.shape[0])
        elif _is_pandas_df(self.data):
            import pandas as pd

            feature_names = [str(c) for c in self.data.columns]
            pandas_cat_cols = [
                i
                for i, dtype in enumerate(self.data.dtypes)
                if isinstance(dtype, pd.CategoricalDtype)
            ]
            pandas_auto_cat_cols = [
                i for i in pandas_cat_cols if not self.data.dtypes.iloc[i].ordered
            ]
            observed_categorical = [
                list(self.data[self.data.columns[i]].cat.categories) for i in pandas_cat_cols
            ] or None
            self.pandas_categorical = (
                self.reference.pandas_categorical
                if self.reference is not None
                else observed_categorical
            )
            matrix = _pandas_to_matrix(self.data, self.pandas_categorical)
        elif _is_sequence_data(self.data):
            matrix = _rows_from_sequences(self.data)
        else:
            # upstream: _data_from_pyarrow_table and _data_from_polars_frame take the column names.
            table_columns = _table_column_names(self.data)
            if table_columns is not None:
                feature_names = table_columns
            matrix = np.asarray(self.data, dtype=np.float64)
        if matrix is not None:
            if matrix.ndim != 2:
                raise ValueError("Input data must be 2-dimensional")
            matrix = np.ascontiguousarray(matrix, dtype=np.float64)
            n = matrix.shape[0]
        if self.position is None:
            self._position = None
        elif self._inherited_position is not None:
            self._position = self._inherited_position
        else:
            self._position = _remap_positions(self.position, n)
        if self._position is not None:
            self.position = self._position
        if self.label is None:
            label = np.zeros(n, dtype=np.float64)
        else:
            label = _list_to_1d_numpy(data=self.label, dtype=np.float64, name="label")
            if label.shape[0] != n:
                raise ValueError(f"Length of label ({label.shape[0]}) is not same with #data ({n})")
        if isinstance(self.feature_name, list):
            # An explicit ``feature_name`` names the columns, so ``categorical_feature`` may use it.
            feature_names = [str(name) for name in self.feature_name]
        if (
            isinstance(self.categorical_feature, str) and self.categorical_feature == "auto"
        ) and not (pandas_auto_cat_cols):
            categorical = _categorical_from_params(self.params, feature_names)
        else:
            categorical = _categorical_indices(
                self.categorical_feature, feature_names, pandas_auto_cat_cols
            )
        self._has_non_default_feature_names = (
            feature_names is not None or self.feature_name != "auto"
        )
        if self.reference is not None:
            categorical = self.reference._categorical.copy()
        self._categorical = categorical
        params = _params_to_strings(self.params)
        if categorical:
            params["categorical_feature"] = ",".join(str(c) for c in categorical)
        weight = None
        if self.weight is not None:
            weight = [
                float(w)
                for w in _list_to_1d_numpy(data=self.weight, dtype=np.float32, name="weight")
            ]
            if weight and len(weight) != n:
                raise LightGBMError("Length of weights differs from the length of #data")
        group = None
        if self.group is not None:
            group = [int(g) for g in np.asarray(self.group).reshape(-1)]
            # upstream: src/io/metadata.cpp (v4.7.0) Metadata::SetQuery.
            if group and sum(group) != n:
                raise LightGBMError(
                    f"Sum of query counts ({n}) differs from the length of #data ({sum(group)})"
                )
        init_score = None
        if self.init_score is not None:
            init_score = _init_score_values(self.init_score, n)
        reference = self.reference._handle_or_raise() if self.reference is not None else None
        # upstream: an explicit `feature_name` list wins; else DataFrame columns name the features.
        names: list[str] | None = None
        if isinstance(self.feature_name, list):
            names = [str(name) for name in self.feature_name]
        elif feature_names is not None:
            names = feature_names
        if sparse is not None:
            import scipy.sparse

            return NativeDataset.from_sparse(
                np.ascontiguousarray(sparse.indptr, dtype=np.int64),
                np.ascontiguousarray(sparse.indices, dtype=np.int32),
                np.ascontiguousarray(sparse.data, dtype=np.float64),
                int(sparse.shape[0]),
                int(sparse.shape[1]),
                isinstance(sparse, scipy.sparse.csc_matrix),
                label.tolist(),
                params,
                categorical,
                weight,
                group,
                reference,
                init_score,
                feature_names=names,
            )
        assert matrix is not None
        return NativeDataset.from_dense(
            matrix,
            label.tolist(),
            params,
            categorical,
            weight,
            group,
            reference,
            init_score,
            feature_names=names,
        )


def _custom_metrics(feval: Any) -> list[Any]:
    """Normalize ``feval`` (one callable or a list of them) to a list, as upstream accepts both."""
    if feval is None:
        return []
    if callable(feval):
        return [feval]
    if isinstance(feval, list):
        return feval
    raise TypeError("feval should be a callable or a list of callables")


def _custom_results(
    name: str, predictions: Any, dataset: Dataset, custom_metrics: list[Any]
) -> list[tuple[str, str, float, bool]]:
    """Run custom metrics on ``predictions`` and return ``(name, metric, value, higher_is_better)``."""
    results: list[tuple[str, str, float, bool]] = []
    for custom_metric in custom_metrics:
        if custom_metric is None:
            continue
        metric_results = custom_metric(predictions, dataset)
        if not isinstance(metric_results, list):
            metric_results = [metric_results]
        for metric_result in metric_results:
            if len(metric_result) != 3:
                raise ValueError("custom evaluation must return (name, value, higher_is_better)")
            metric_name, value, higher_is_better = metric_result
            results.append((name, str(metric_name), float(value), bool(higher_is_better)))
    return results


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
        # Name under which the training data is evaluated when passed in ``valid_sets``.
        self._train_data_name = "training"
        self.best_score: dict[str, dict[str, float]] = {}
        self._train_set = train_set
        self._handle: NativeBooster | None = None
        # Trees of the model that continued training starts from (``init_model``).
        self._init_model: NativeBooster | None = None
        if train_set is not None:
            # Validation Datasets are tracked only for a booster that trains (see free_dataset).
            self.valid_sets: list[tuple[Dataset, str]] = []
            if model_file is not None or model_str is not None:
                raise TypeError("Booster cannot take both a training set and a model")
            merged_params = {**getattr(train_set, "params", {}), **self.params}
            _log_unknown_params(merged_params)
            # upstream: Booster.__init__ creates the C++ booster (LGBM_BoosterCreate), which
            # parses the config and emits its warnings and errors at this point.
            validate_params(_params_to_strings(merged_params))
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
        # The constructor already reported this config's warnings, so do not repeat them.
        self._handle = NativeBooster.train(
            _params_to_strings(merged),
            dataset._handle_or_raise(),
            self._init_model,
            warn_config=False,
        )
        self.pandas_categorical = dataset.pandas_categorical

    def _continue_from(self, init_model: Any) -> None:
        """Start training after ``init_model``'s trees (upstream ``LGBM_BoosterMerge``).

        The init model's raw predictions on the training data become the training init score,
        so the new trees fit what the initial trees leave over.
        """
        if self._train_set is None:
            raise LightGBMError("Booster has no training Dataset")
        if isinstance(init_model, Booster):
            init = init_model
        elif isinstance(init_model, (str, Path)):
            init = Booster(model_file=init_model)
        else:
            # Upstream silently ignores other objects; bankai refuses them instead.
            raise TypeError("init_model should be a Booster or a path to a model file")
        dataset = self._train_set
        if dataset.init_score is not None:
            warnings.warn(
                "The init_score will be overridden by the prediction of init_model.",
                UserWarning,
                stacklevel=3,
            )
        dataset.set_init_score(init.predict(dataset.data, raw_score=True))
        self._init_model = init._native()
        self.pandas_categorical = dataset.pandas_categorical

    def shuffle_models(self, start_iteration: int = 0, end_iteration: int = -1) -> Booster:
        """Shuffle the iterations in ``[start_iteration, end_iteration)`` (upstream order)."""
        self._native().shuffle_models(int(start_iteration), int(end_iteration))
        return self

    def set_leaf_output(self, tree_id: int, leaf_id: int, value: float) -> Booster:
        """Set one leaf output of a tree (model tree index, as in ``get_leaf_output``)."""
        self._native().set_leaf_output(int(tree_id), int(leaf_id), float(value))
        return self

    def rollback_one_iter(self) -> Booster:
        """Remove the last iteration's trees and their effect on the training scores."""
        self._native().rollback_one_iter()
        return self

    # upstream: python-package/lightgbm/basic.py (v4.7.0) Booster.get_split_value_histogram
    def get_split_value_histogram(
        self,
        feature: int | str,
        bins: int | str | None = None,
        xgboost_style: bool = False,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]] | npt.NDArray[Any] | pd_DataFrame:
        """Get split value histogram for the specified feature.

        Parameters
        ----------
        feature : int or str
            The feature name or index the histogram is calculated for.
            If int, interpreted as index.
            If str, interpreted as name.

            .. warning::

                Categorical features are not supported.

        bins : int, str or None, optional (default=None)
            The maximum number of bins.
            If None, or int and > number of unique split values and ``xgboost_style=True``,
            the number of bins equals number of unique split values.
            If str, it should be one from the list of the supported values by ``numpy.histogram()`` function.
        xgboost_style : bool, optional (default=False)
            Whether the returned result should be in the same form as it is in XGBoost.
            If False, the returned value is tuple of 2 numpy arrays as it is in ``numpy.histogram()`` function.
            If True, the returned value is matrix, in which the first column is the right edges of non-empty bins
            and the second one is the histogram values.

        Returns
        -------
        result_tuple : tuple of 2 numpy arrays
            If ``xgboost_style=False``, the values of the histogram of used splitting values for the specified feature
            and the bin edges.
        result_array_like : numpy array or pandas DataFrame (if pandas is installed)
            If ``xgboost_style=True``, the histogram of used splitting values for the specified feature.
        """

        model = self.dump_model()
        feature_names: list[str] | None = model.get("feature_names")
        tree_infos = model["tree_info"]
        values: list[float] = []

        def add(root: dict[str, Any]) -> None:
            """Recursively add thresholds."""
            if "split_index" in root:  # non-leaf
                if feature_names is not None and isinstance(feature, str):
                    split_feature = feature_names[root["split_feature"]]
                else:
                    split_feature = root["split_feature"]
                if split_feature == feature:
                    if isinstance(root["threshold"], str):
                        raise LightGBMError(
                            "Cannot compute split value histogram for the categorical feature"
                        )
                    values.append(root["threshold"])
                add(root["left_child"])
                add(root["right_child"])

        for tree_info in tree_infos:
            add(tree_info["tree_structure"])

        if bins is None or (isinstance(bins, int) and xgboost_style):
            n_unique = len(np.unique(values))
            bins = max(min(n_unique, bins) if bins is not None else n_unique, 1)
        hist, bin_edges = np.histogram(values, bins=bins)
        if xgboost_style:
            ret = np.column_stack((bin_edges[1:], hist))
            ret = ret[ret[:, 1] > 0]
            if PANDAS_INSTALLED:
                return pd_DataFrame(ret, columns=["SplitValue", "Count"])
            else:
                return ret
        else:
            return hist, bin_edges

    def _get_loaded_param(self) -> dict[str, Any]:
        """Parameters of the ``parameters:`` block of the model text (upstream ``GetLoadedParam``).

        upstream: src/boosting/gbdt.h (v4.7.0) ``GetLoadedParam`` returns the same block as a
        JSON object typed by ``Config::ParameterTypes``; here the types are read from the text.
        """
        text = self.model_to_string()
        header = "\nparameters:\n"
        start = text.find(header)
        if start < 0:
            return {}
        block = text[start + len(header) : text.find("\nend of parameters", start)]
        out: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("[") and line.endswith("]") and ": " in line:
                key, _, value = line[1:-1].partition(": ")
                out[key] = _typed_param_value(value)
        return out

    def free_dataset(self) -> Booster:
        """Drop the references to the training and validation Datasets (upstream ``free_dataset``).

        Returns
        -------
        self : Booster
            Booster without Datasets.
        """
        self._train_set = None
        self.__dict__.pop("valid_sets", None)
        return self

    def update(self, train_set: Dataset | None = None, fobj: Any = None) -> bool:
        """Train one additional iteration and return whether training has finished."""
        if fobj is not None:
            if self._handle is not None and self._native().objective() is not None:
                self._native().reset_custom_objective()
            self.params["objective"] = "none"
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
                _params_to_strings(merged), dataset._handle_or_raise(), self._init_model
            )
            self.pandas_categorical = dataset.pandas_categorical
        if fobj is not None:
            # upstream: fobj(self.__inner_predict(data_idx=0), ...) -- the internal raw
            # training score (init score included; DART drops this iteration's trees first),
            # class-major and reshaped to (num_data, num_class) in Fortran order.
            predictions = self._native().training_score()
            if self.num_model_per_iteration() > 1:
                predictions = predictions.reshape((-1, self.num_model_per_iteration()), order="F")
            gradients, hessians = fobj(predictions, self._train_set)
            num_class = self.num_model_per_iteration()
            if num_class > 1:
                gradients = gradients.ravel(order="F")
                hessians = hessians.ravel(order="F")
            grad = _custom_derivative(gradients, "gradient")
            hess = _custom_derivative(hessians, "hessian")
            if len(grad) != len(hess):
                raise ValueError(
                    f"Lengths of gradient ({len(grad)}) and Hessian ({len(hess)}) don't match"
                )
            num_rows = self._train_set.num_data()
            if len(grad) != num_rows * num_class:
                raise ValueError(
                    f"Lengths of gradient ({len(grad)}) and Hessian ({len(hess)}) "
                    f"don't match training data length ({num_rows}) * "
                    f"number of models per one iteration ({num_class})"
                )
            return self._native().update_custom(grad.tolist(), hess.tolist())
        return self._native().update()

    def add_valid(self, data: Dataset, name: str) -> None:
        """Register a validation dataset for :meth:`eval_valid`."""
        self.__dict__.setdefault("valid_sets", []).append((data, name))

    def eval(self, data: Dataset, name: str, feval: Any = None) -> list[EvalResult]:
        """Evaluate the configured metrics on ``data`` (upstream ``Booster.eval``).

        ``feval`` adds custom metrics, which are computed from the converted predictions.
        """
        if not isinstance(data, Dataset):
            raise TypeError("Can only eval for Dataset instance")
        params = {**data.params, **self.params}
        native = self._native().eval(data._construct_for_eval(), _params_to_strings(params), name)
        results = [EvalResult(*item) for item in native]
        custom_metrics = _custom_metrics(feval)
        if custom_metrics:
            predictions = self.predict(_internal_rows(data.data))
            results.extend(
                EvalResult(*item)
                for item in _custom_results(name, predictions, data, custom_metrics)
            )
        return results

    def eval_valid(self, feval: Any = None) -> list[tuple[str, str, float, bool]]:
        """Evaluate configured and custom metrics on registered validation data."""
        custom_metrics = _custom_metrics(feval)
        results: list[tuple[str, str, float, bool]] = []
        for dataset, name in self.__dict__.get("valid_sets", []):
            native_dataset = dataset._construct_for_eval()
            params = {**dataset.params, **self.params}
            results.extend(self._native().eval(native_dataset, _params_to_strings(params), name))
            if custom_metrics:
                predictions = self.predict(_internal_rows(dataset.data))
                results.extend(_custom_results(name, predictions, dataset, custom_metrics))
        return results

    def reset_parameter(self, params: dict[str, Any]) -> None:
        """Reset supported parameters while continuing training."""
        aliases = {
            "learning_rate": "learning_rate",
            "eta": "learning_rate",
            "shrinkage_rate": "learning_rate",
        }
        normalized: dict[str, float] = {}
        for name, value in params.items():
            canonical = aliases.get(name)
            if canonical is None:
                raise NotImplementedError(f"reset_parameter for {name!r} is not implemented yet")
            rate = float(value)
            if not np.isfinite(rate) or rate <= 0.0:
                raise ValueError("learning_rate must be finite and greater than zero")
            normalized[canonical] = rate
        if self._handle is not None and "learning_rate" in normalized:
            self._handle.reset_learning_rate(normalized["learning_rate"])
        self.params.update(normalized)

    def __copy__(self) -> Booster:
        return self.__deepcopy__(None)

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> Booster:
        return Booster(model_str=self.model_to_string(num_iteration=-1))

    def __getstate__(self) -> dict[str, Any]:
        this = self.__dict__.copy()
        this.pop("train_set", None)
        this.pop("valid_sets", None)
        this.pop("_init_model", None)
        this["_handle"] = self.model_to_string(num_iteration=-1)
        return this

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_str = state.get("_handle", state.get("handle"))
        if isinstance(model_str, str):
            state["_handle"] = NativeBooster.from_string(model_str)
        state.setdefault("_train_set", None)
        state.setdefault("valid_sets", [])
        state.setdefault("_init_model", None)
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
    ) -> _PredictReturnType:
        """Make a prediction. Same semantics and output shapes as ``lightgbm.Booster.predict``.

        ``pred_contrib`` on a ``scipy.sparse`` CSR/CSC input returns a sparse matrix of the
        same format (a list of them for multiclass), like upstream.
        """
        if num_iteration is None:
            num_iteration = self.best_iteration if start_iteration <= 0 else -1
        if isinstance(data, (str, Path)):
            raise NotImplementedError("Predicting from a file is not implemented yet (roadmap P2)")
        p = _resolve_predict_params({**self.params, **kwargs})
        predict_type = _C_API_PREDICT_NORMAL
        if raw_score:
            predict_type = _C_API_PREDICT_RAW_SCORE
        if pred_leaf:
            predict_type = _C_API_PREDICT_LEAF_INDEX
        if pred_contrib:
            predict_type = _C_API_PREDICT_CONTRIB
        native_kwargs: dict[str, Any] = {
            "pred_early_stop": _to_bool(p.get("pred_early_stop", False)),
            "pred_early_stop_freq": int(p.get("pred_early_stop_freq", 10)),
            "pred_early_stop_margin": float(p.get("pred_early_stop_margin", 10.0)),
            "num_threads": int(p.get("num_threads", 0)),
            "disable_shape_check": _to_bool(p.get("predict_disable_shape_check", False)),
        }
        sparse = self._as_sparse(data)
        if sparse is not None:
            return self._predict_sparse(
                sparse, predict_type, int(start_iteration), int(num_iteration), native_kwargs
            )
        if _is_pandas_df(data):
            if validate_features:
                _validate_feature_names(self.feature_name(), [str(c) for c in data.columns])
            mat = _pandas_to_matrix(data, self.pandas_categorical)
        elif _table_column_names(data) is not None:
            mat = np.asarray(data, dtype=np.float64)
        else:
            mat = self._to_dense(data)
        if mat.ndim != 2:
            raise ValueError("Input numpy.ndarray or list must be 2 dimensional")
        mat = np.ascontiguousarray(mat, dtype=np.float64)
        preds = self._native().predict_dense(
            mat, predict_type, int(start_iteration), int(num_iteration), **native_kwargs
        )
        nrow = mat.shape[0]
        if pred_leaf:
            return preds.astype(np.int32)
        if preds.size != nrow or pred_contrib:
            return preds
        return preds.reshape(nrow)

    @staticmethod
    def _as_sparse(data: Any) -> Any:
        """The CSR/CSC matrix to predict, or ``None`` for dense inputs.

        Mirrors upstream's ``_InnerPredictor.predict`` dispatch: ``csr_matrix`` and
        ``csc_matrix`` are used as-is; any other input that is not a numpy array, pandas
        DataFrame or list (other scipy formats, ``sparray`` types, ...) is converted with
        ``scipy.sparse.csr_matrix`` after logging a warning.
        """
        if isinstance(data, (np.ndarray, list)) or _is_pandas_df(data):
            return None
        if _table_column_names(data) is not None:
            # upstream: dataframe-like inputs (pyarrow Table, polars DataFrame) predict densely.
            return None
        try:
            import scipy.sparse
        except ImportError:  # pragma: no cover
            return None
        if isinstance(data, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            return data
        try:
            _log_warning("Converting data to scipy sparse matrix.")
            return scipy.sparse.csr_matrix(data)
        except BaseException as err:
            raise TypeError(f"Cannot predict data for type {type(data).__name__}") from err

    def _predict_sparse(
        self,
        cs: Any,
        predict_type: int,
        start_iteration: int,
        num_iteration: int,
        native_kwargs: dict[str, Any],
    ) -> Any:
        """Predict a ``csr_matrix``/``csc_matrix`` natively, row by row.

        ``pred_contrib`` returns a sparse matrix of the input's format with one extra
        (bias) column, or a list of them for multiclass, with the input's float dtype,
        like upstream's ``LGBM_BoosterPredictSparseOutput``.
        """
        import scipy.sparse

        is_csc = isinstance(cs, scipy.sparse.csc_matrix)
        nrow_total = int(cs.shape[0])
        if predict_type == _C_API_PREDICT_CONTRIB and nrow_total > _SPARSE_PREDICT_BLOCK_ROWS:
            # The native output is built for the rows it is given, so a large input is predicted
            # in row blocks; the sparse blocks are stacked, as upstream's sparse output is.
            csr = cs.tocsr()
            blocks = [
                self._predict_sparse(
                    csr[start : start + _SPARSE_PREDICT_BLOCK_ROWS],
                    predict_type,
                    start_iteration,
                    num_iteration,
                    native_kwargs,
                )
                for start in range(0, nrow_total, _SPARSE_PREDICT_BLOCK_ROWS)
            ]
            if isinstance(blocks[0], list):
                stacked: Any = [
                    scipy.sparse.vstack([block[k] for block in blocks], format="csr")
                    for k in range(len(blocks[0]))
                ]
            else:
                stacked = scipy.sparse.vstack(blocks, format="csr")
            if not is_csc:
                return stacked
            if isinstance(stacked, list):
                return [m.tocsc() for m in stacked]
            return stacked.tocsc()
        _c_int_dtype_check(cs.indptr)
        _c_float_dtype_check(cs.data)
        nrow, ncol = cs.shape
        out = self._native().predict_sparse(
            np.ascontiguousarray(cs.indptr, dtype=np.int64),
            np.ascontiguousarray(cs.indices, dtype=np.int32),
            np.ascontiguousarray(cs.data, dtype=np.float64),
            int(nrow),
            int(ncol),
            is_csc,
            predict_type,
            start_iteration,
            num_iteration,
            **native_kwargs,
        )
        if predict_type == _C_API_PREDICT_LEAF_INDEX:
            return out.astype(np.int32)
        if predict_type != _C_API_PREDICT_CONTRIB:
            return out if out.shape[1] != 1 else out.reshape(nrow)
        make = scipy.sparse.csc_matrix if is_csc else scipy.sparse.csr_matrix
        mats = [
            make(
                (
                    data.astype(cs.data.dtype, copy=False),
                    indices,
                    indptr.astype(cs.indptr.dtype, copy=False),
                ),
                shape=(nrow, ncol + 1),
            )
            for indptr, indices, data in out
        ]
        return mats[0] if len(mats) == 1 else mats

    @staticmethod
    def _to_dense(data: Any) -> npt.NDArray[Any]:
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, list):
            try:
                return np.array(data)
            except BaseException as err:
                raise ValueError("Cannot convert data list to numpy array.") from err
        raise TypeError(f"Cannot predict data for type {type(data).__name__}")
