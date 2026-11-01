"""bankai-lightgbm: LightGBM rewritten in Rust, with a drop-in Python API.

Use ``import bankai_lightgbm as lgb`` wherever ``import lightgbm as lgb`` works.
"""

from bankai_lightgbm import dask as dask
from bankai_lightgbm._lib import UPSTREAM_VERSION, __version__
from bankai_lightgbm.basic import (
    Booster,
    Dataset,
    EvalResult,
    LightGBMError,
    Sequence,
    register_logger,
)
from bankai_lightgbm.callback import (
    EarlyStopException,
    early_stopping,
    log_evaluation,
    record_evaluation,
    reset_parameter,
)
from bankai_lightgbm.engine import CVBooster, cv, train
from bankai_lightgbm.plotting import (
    create_tree_digraph,
    plot_importance,
    plot_metric,
    plot_split_value_histogram,
    plot_tree,
)
from bankai_lightgbm.sklearn import LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor

__all__ = [
    "UPSTREAM_VERSION",
    "Booster",
    "CVBooster",
    "Dataset",
    "EarlyStopException",
    "EvalResult",
    "LGBMClassifier",
    "LGBMModel",
    "LGBMRanker",
    "LGBMRegressor",
    "LightGBMError",
    "Sequence",
    "__version__",
    "create_tree_digraph",
    "cv",
    "early_stopping",
    "log_evaluation",
    "plot_importance",
    "plot_metric",
    "plot_split_value_histogram",
    "plot_tree",
    "record_evaluation",
    "register_logger",
    "reset_parameter",
    "train",
]
