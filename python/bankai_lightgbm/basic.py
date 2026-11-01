"""Core objects: ``Booster`` (and, later, ``Dataset``). Mirrors ``lightgbm.basic``."""

from __future__ import annotations

import ctypes
import json
import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any, Literal, Union

import narwhals.typing as nwt
import numpy as np
import numpy.typing as npt
import scipy.sparse

from bankai_lightgbm._lib import LightGBMError, NativeBooster, NativeDataset
from bankai_lightgbm.compat import PANDAS_INSTALLED
from bankai_lightgbm.compat import pd_DataFrame as pd_DataFrame
from bankai_lightgbm.compat import pd_Series as pd_Series

__all__ = ["Booster", "Dataset", "LightGBMError"]


def _list_to_1d_numpy(data: Any, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """Flatten a label- or weight-like field to 1-D (upstream ``_list_to_1d_numpy``, v4.7.0).

    A column vector is flattened with the warning upstream logs; lists and pandas Series are
    accepted as they are.
    """
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 1:
        _log_warning("Converting column-vector to 1d array")
    return np.asarray(data, dtype=dtype).reshape(-1)


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
            _LOGGER.info(f"[LightGBM] [Warning] Unknown parameter: {key}")


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


class _DummyLogger:
    """upstream's default logger: ``info`` prints, ``warning`` emits a ``UserWarning``."""

    def info(self, msg: str) -> None:
        print(msg)

    def warning(self, msg: str) -> None:
        warnings.warn(msg, stacklevel=3)


_LOGGER: Any = _DummyLogger()


def _log_info(msg: str) -> None:
    _LOGGER.info(msg)


def _log_warning(msg: str) -> None:
    _LOGGER.warning(msg)


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
    "Sequence[Any]",
    list["Sequence[Any]"],
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
        categorical_feature: Sequence[int | str] | str = "auto",
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
        # upstream: True when feature names came from the data or `feature_name` (sklearn reads it).
        self._has_non_default_feature_names: bool = False
        self._handle: NativeDataset | None = None
        self._evaluation_handle: NativeDataset | None = None

    def construct(self) -> Dataset:
        """Bin the data (no-op if already constructed)."""
        if self._handle is None:
            self._handle = self._build()
        return self

    def num_data(self) -> int:
        return self.construct()._handle_or_raise().num_data()

    def num_feature(self) -> int:
        return self.construct()._handle_or_raise().num_feature()

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
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            params={**self.params, **(params or {})},
            free_raw_data=self.free_raw_data,
        )
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
        self.weight = weight
        if self._handle is not None or self._evaluation_handle is not None:
            values = (
                []
                if weight is None
                else [float(w) for w in np.asarray(weight, np.float32).reshape(-1)]
            )
            self._update_native_fields(weight=values)
        return self

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
            matrix = _data_from_pandas(self.data, self.pandas_categorical)
        else:
            matrix = np.asarray(self.data, dtype=np.float64)
        if matrix is not None:
            if matrix.ndim != 2:
                raise ValueError("Input data must be 2-dimensional")
            matrix = np.ascontiguousarray(matrix, dtype=np.float64)
            n = matrix.shape[0]
        if self.label is None:
            label = np.zeros(n, dtype=np.float64)
        else:
            label = _list_to_1d_numpy(self.label, np.float64)
            if label.shape[0] != n:
                raise ValueError(f"Length of label ({label.shape[0]}) is not same with #data ({n})")
        if isinstance(self.feature_name, list):
            # An explicit ``feature_name`` names the columns, so ``categorical_feature`` may use it.
            feature_names = [str(name) for name in self.feature_name]
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
            weight = [float(w) for w in _list_to_1d_numpy(self.weight, np.float32)]
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
            _log_unknown_params({**getattr(train_set, "params", {}), **self.params})
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
        self._handle = NativeBooster.train(
            _params_to_strings(merged), dataset._handle_or_raise(), self._init_model
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

    def eval_valid(self, feval: Any = None) -> list[tuple[str, str, float, bool]]:
        """Evaluate configured and custom metrics on registered validation data."""
        if feval is None:
            custom_metrics: list[Any] = []
        elif callable(feval):
            custom_metrics = [feval]
        elif isinstance(feval, list):
            custom_metrics = feval
        else:
            raise TypeError("feval should be a callable or a list of callables")
        results: list[tuple[str, str, float, bool]] = []
        for dataset, name in self.__dict__.get("valid_sets", []):
            native_dataset = dataset._construct_for_eval()
            params = {**dataset.params, **self.params}
            results.extend(self._native().eval(native_dataset, _params_to_strings(params), name))
            if custom_metrics:
                predictions = self.predict(_internal_rows(dataset.data))
                for custom_metric in custom_metrics:
                    if custom_metric is None:
                        continue
                    metric_results = custom_metric(predictions, dataset)
                    if not isinstance(metric_results, list):
                        metric_results = [metric_results]
                    for metric_result in metric_results:
                        if len(metric_result) != 3:
                            raise ValueError(
                                "custom evaluation must return (name, value, higher_is_better)"
                            )
                        metric_name, value, higher_is_better = metric_result
                        results.append(
                            (name, str(metric_name), float(value), bool(higher_is_better))
                        )
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
            mat = _data_from_pandas(data, self.pandas_categorical)
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
