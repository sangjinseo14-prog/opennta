"""Unit test for ``component_stats`` — NaN-aware min/max/mean/sd summary.

``component_stats`` backs the numerical dialog's post-compute u, v readout and
is not exercised by any pipeline test (the readout is pure UI), so its NaN
handling and empty-field contract are pinned directly here.
"""
from __future__ import annotations

import numpy as np
import pytest

from opennta.analysis.numerical_field.field_stats import (
    ComponentStats,
    component_stats,
)


def test_component_stats_ignores_nan_cells():
    # A 2x2 field with one empty (NaN) cell: the summary uses only finite cells.
    field = np.array([[1.0, np.nan], [3.0, 5.0]])

    s = component_stats(field)

    assert isinstance(s, ComponentStats)
    assert s.min == pytest.approx(1.0)
    assert s.max == pytest.approx(5.0)
    assert s.mean == pytest.approx(3.0)  # (1 + 3 + 5) / 3
    assert s.std == pytest.approx(np.std([1.0, 3.0, 5.0]))  # population sd


def test_component_stats_all_nan_returns_none():
    assert component_stats(np.full((3, 3), np.nan)) is None


def test_component_stats_empty_returns_none():
    assert component_stats(np.array([])) is None
