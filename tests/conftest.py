"""Repository-wide pytest hooks."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Run upstream's ported suite with non-strict ``xfail`` marks.

    The ported files keep upstream's ``xfail`` marks verbatim. Upstream runs them without
    ``xfail_strict``, so a check that passes here is reported as XPASS rather than failing the
    run. Every other test in the repository keeps the strict setting from ``pyproject.toml``.
    """
    for item in items:
        if "python_package_test" not in item.path.parts:
            continue
        for index, mark in enumerate(item.own_markers):
            if mark.name == "xfail" and mark.kwargs.get("strict", True):
                kwargs = {**mark.kwargs, "strict": False}
                item.own_markers[index] = pytest.mark.xfail(*mark.args, **kwargs).mark
