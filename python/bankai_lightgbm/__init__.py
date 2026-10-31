"""bankai-lightgbm: LightGBM rewritten in Rust, with a drop-in Python API.

Use ``import bankai_lightgbm as lgb`` wherever ``import lightgbm as lgb`` works.
"""

from bankai_lightgbm._lib import UPSTREAM_VERSION, __version__

__all__ = ["UPSTREAM_VERSION", "__version__"]
