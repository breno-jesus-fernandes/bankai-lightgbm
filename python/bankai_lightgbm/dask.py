"""Dask estimators are not implemented yet (roadmap P7: distributed training).

Mirrors the import surface of ``lightgbm.dask``: ``_DaskImportErrorTypes`` lists the errors
that mean "dask is not usable", as upstream does. There are no Dask estimator classes here, so
``lgb.DaskLGBM*`` is an ``AttributeError`` rather than a silently different model.
"""

# upstream: python-package/lightgbm/dask.py (v4.7.0) _DaskImportErrorTypes
_DaskImportErrorTypes = (ImportError, ValueError)
