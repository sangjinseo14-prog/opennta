"""Smoke tests for ``opennta.application.common.statistics``.

Covers the two entry points used by both the Analysis tab and the Batch
output: ``_calculate_distribution_stats`` (the bottom of every panel) and
``_build_drift_vector_summary`` (drift panels, pre/post correction).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from opennta.application.common.statistics import StatisticsUtils


def test_distribution_stats_diameter_matches_numpy_moments():
    diameters = np.array([90.0, 95.0, 100.0, 105.0, 110.0])

    stats = StatisticsUtils._calculate_distribution_stats(diameters, kind="diameter")

    assert stats["unit"] == "nm"
    assert stats["name"] == "Particle Size"
    assert stats["mean"] == pytest.approx(float(np.mean(diameters)))
    assert stats["median"] == pytest.approx(float(np.median(diameters)))
    assert stats["std"] == pytest.approx(float(np.std(diameters)))
    # Diameter histograms also surface peaks for the UI; with five evenly
    # spaced points there is no prominent peak, but the field must exist.
    assert isinstance(stats["peaks"], list)


def test_distribution_stats_empty_returns_nan_envelope():
    stats = StatisticsUtils._calculate_distribution_stats(
        np.array([], dtype=float), kind="diameter"
    )

    assert math.isnan(stats["mean"])
    assert math.isnan(stats["median"])
    assert math.isnan(stats["std"])
    assert stats["peaks"] == []
    assert stats["unit"] == "nm"


def test_drift_vector_summary_empty_returns_nan_tuple():
    summary = StatisticsUtils._build_drift_vector_summary(
        drift_by_id=None, passed_ids=set(), name="Drift",
    )

    assert summary["n_tracks"] == 0
    assert all(math.isnan(x) for x in summary["mean_xy"])
    assert all(math.isnan(x) for x in summary["median_xy"])
    assert summary["unit"] == "um/s"


def test_drift_vector_summary_filters_by_passed_ids():
    drift_by_id = {1: (1.0, 2.0), 2: (3.0, 4.0), 3: (5.0, 6.0)}
    passed = {1, 3}

    summary = StatisticsUtils._build_drift_vector_summary(
        drift_by_id, passed_ids=passed, name="Drift",
    )

    assert summary["n_tracks"] == 2
    assert summary["mean_xy"][0] == pytest.approx(3.0)
    assert summary["mean_xy"][1] == pytest.approx(4.0)
