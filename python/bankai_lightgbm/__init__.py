"""bankai-lightgbm: LightGBM rewritten in Rust, with a drop-in Python API.

Use ``import bankai_lightgbm as lgb`` wherever ``import lightgbm as lgb`` works.
"""

from bankai_lightgbm._lib import UPSTREAM_VERSION, __version__
from bankai_lightgbm.basic import Booster, LightGBMError

__all__ = ["UPSTREAM_VERSION", "Booster", "LightGBMError", "__version__"]
