"""bankai-lightgbm: LightGBM rewritten in Rust, with a drop-in Python API.

Use ``import bankai_lightgbm as lgb`` wherever ``import lightgbm as lgb`` works.
"""

from bankai_lightgbm._lib import UPSTREAM_VERSION, __version__
from bankai_lightgbm.basic import Booster, Dataset, LightGBMError
from bankai_lightgbm.callback import (
    early_stopping,
    log_evaluation,
    record_evaluation,
    reset_parameter,
)
from bankai_lightgbm.engine import CVBooster, cv, train
from bankai_lightgbm.sklearn import LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor

__all__ = [
    "UPSTREAM_VERSION",
    "Booster",
    "CVBooster",
    "Dataset",
    "LGBMClassifier",
    "LGBMModel",
    "LGBMRanker",
    "LGBMRegressor",
    "LightGBMError",
    "__version__",
    "cv",
    "early_stopping",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
    "train",
]
