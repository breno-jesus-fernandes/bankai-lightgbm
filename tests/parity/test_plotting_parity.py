"""Plotting helpers: the plotted data and graph source match upstream `lightgbm.plotting`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import lightgbm as up
import matplotlib.pyplot as plt
import numpy as np
import pytest

import bankai_lightgbm as bk

PARAMS = dict(
    objective="regression",
    num_leaves=7,
    min_data_in_leaf=5,
    learning_rate=0.1,
    num_threads=1,
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
    seed=5,
)


def _data(n: int = 500, f: int = 4, seed: int = 9) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.1 * rng.normal(size=n)
    return x, y


def _trained(lib, x: np.ndarray, y: np.ndarray, record: dict):
    train_set = lib.Dataset(x[:400], y[:400], feature_name=["a", "b", "c", "d"])
    valid = lib.Dataset(x[400:], y[400:], reference=train_set)
    return lib.train(
        PARAMS,
        train_set,
        num_boost_round=15,
        valid_sets=[valid],
        valid_names=["valid"],
        callbacks=[lib.record_evaluation(record)],
    )


def _bar_widths(ax) -> list[float]:
    return [round(p.get_width(), 12) for p in ax.patches]


def test_plot_importance_matches_upstream() -> None:
    x, y = _data()
    want = _trained(up, x, y, {})
    got = _trained(bk, x, y, {})
    ax_want = up.plot_importance(want)
    ax_got = bk.plot_importance(got)
    assert [t.get_text() for t in ax_got.get_yticklabels()] == [
        t.get_text() for t in ax_want.get_yticklabels()
    ]
    assert _bar_widths(ax_got) == _bar_widths(ax_want)
    plt.close("all")


def test_plot_metric_matches_upstream() -> None:
    x, y = _data()
    rec_want: dict = {}
    rec_got: dict = {}
    _trained(up, x, y, rec_want)
    _trained(bk, x, y, rec_got)
    ax_want = up.plot_metric(rec_want)
    ax_got = bk.plot_metric(rec_got)
    np.testing.assert_allclose(
        ax_got.lines[0].get_ydata(), ax_want.lines[0].get_ydata(), rtol=1e-12
    )
    plt.close("all")


def test_split_value_histogram_matches_upstream() -> None:
    x, y = _data()
    want = _trained(up, x, y, {})
    got = _trained(bk, x, y, {})
    for feature in ["a", 0]:
        hist_want, edges_want = want.get_split_value_histogram(feature)
        hist_got, edges_got = got.get_split_value_histogram(feature)
        np.testing.assert_array_equal(hist_got, hist_want)
        np.testing.assert_allclose(edges_got, edges_want, rtol=1e-12)
    ax_got = bk.plot_split_value_histogram(got, "a")
    ax_want = up.plot_split_value_histogram(want, "a")
    assert _bar_widths(ax_got) == _bar_widths(ax_want)
    plt.close("all")


def test_tree_digraph_source_matches_upstream() -> None:
    # The graphviz package (not the `dot` binary) is needed to build the digraph source.
    pytest.importorskip("graphviz")
    x, y = _data()
    want = _trained(up, x, y, {})
    got = _trained(bk, x, y, {})
    assert (
        bk.create_tree_digraph(got, tree_index=0).source
        == up.create_tree_digraph(want, tree_index=0).source
    )


def test_plot_tree_axes_match_upstream() -> None:
    if __import__("shutil").which("dot") is None:
        pytest.skip("graphviz 'dot' binary is needed to render plot_tree")
    x, y = _data()
    want = _trained(up, x, y, {})
    got = _trained(bk, x, y, {})
    ax_got = bk.plot_tree(got, tree_index=0)
    ax_want = up.plot_tree(want, tree_index=0)
    assert ax_got.get_images()[0].get_array().shape == ax_want.get_images()[0].get_array().shape
    plt.close("all")
