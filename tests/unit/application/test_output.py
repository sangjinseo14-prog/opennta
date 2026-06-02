"""Smoke tests for ``opennta.application.tab_analysis.output``.

These tests verify that plot factories return ``None`` for empty inputs
(the guard relied on by the Analysis tab to skip drawing) and produce a
matplotlib ``Figure`` for valid inputs, without asserting visual details.
``MPLBACKEND=Agg`` is set in conftest so plotting works headlessly.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from opennta.application.tab_analysis.output import TabAnalysisOutput


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_create_size_plot_returns_none_for_empty_input():
    creator = TabAnalysisOutput()
    assert creator.create_size_plot({}) is None
    assert creator.create_size_plot({"diameters": [], "bin_edges": None, "counts": None}) is None


def test_create_brownian_plot_returns_figure_for_valid_tracks():
    tracks = {
        1: {
            "original": {"x": np.linspace(0.0, 1.0, 20), "y": np.linspace(0.0, 0.5, 20)},
            "corrected": {"x": np.linspace(0.0, 0.5, 20), "y": np.linspace(0.0, 0.25, 20)},
        },
        2: {
            "original": {"x": np.linspace(0.0, -1.0, 20), "y": np.linspace(0.0, 0.7, 20)},
            "corrected": {"x": np.linspace(0.0, -0.4, 20), "y": np.linspace(0.0, 0.3, 20)},
        },
    }

    fig = TabAnalysisOutput().create_brownian_plot(
        {"tracks": tracks, "sample_ids": list(tracks.keys())}
    )

    assert fig is not None
    # Left/right (original / corrected) panels — the gap column is a
    # gridspec spacer, not an Axes, so the figure carries exactly two Axes.
    assert len(fig.axes) == 2


def test_create_brownian_plot_returns_none_for_empty_tracks():
    fig = TabAnalysisOutput().create_brownian_plot({"tracks": {}})
    assert fig is None
