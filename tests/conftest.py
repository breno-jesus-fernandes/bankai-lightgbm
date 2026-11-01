"""Repository-wide pytest hooks."""

from __future__ import annotations

import pathlib

import pytest

_GAPS_FILE = pathlib.Path(__file__).with_name("known_upstream_gaps.tsv")


def _known_gaps() -> dict[str, str]:
    """Read ``known_upstream_gaps.tsv``: node id to the reason the upstream test still fails."""
    gaps: dict[str, str] = {}
    for line in _GAPS_FILE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            node_id, reason = line.split("\t", 1)
            gaps[node_id] = reason
    return gaps


_KNOWN_GAPS = _known_gaps()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Run upstream's ported suite with non-strict ``xfail`` marks, and mark known gaps.

    The ported files keep upstream's ``xfail`` marks verbatim. Upstream runs them without
    ``xfail_strict``, so a check that passes here is reported as XPASS rather than failing the
    run. Every other test in the repository keeps the strict setting from ``pyproject.toml``.

    Node ids listed in ``known_upstream_gaps.tsv`` get strict ``xfail`` marks: upstream tests that
    still fail, each with the ROADMAP item that owns the gap.
    """
    for item in items:
        if "python_package_test" not in item.path.parts:
            continue
        for index, mark in enumerate(item.own_markers):
            if mark.name == "xfail" and mark.kwargs.get("strict", True):
                kwargs = {**mark.kwargs, "strict": False}
                item.own_markers[index] = pytest.mark.xfail(*mark.args, **kwargs).mark
        reason = _KNOWN_GAPS.get(item.nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason))
