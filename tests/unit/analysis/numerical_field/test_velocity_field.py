"""Targeted unit test for ``compute_velocity_field`` — MAD outlier rejection.

Per ``tests/README``, ``numerical_field`` is a *plugin module* verified
end-to-end through ``test_numerical_corrector`` rather than with per-helper
unit tests. This single test is the documented exception ("a targeted unit
test earns its place" when the integration cannot localise — or here, cannot
even *detect* — a sub-step regression):

The integration feeds a *uniform* drift field, which contains no outliers, and
asserts only that drift is removed within an end-to-end tolerance. That
assertion is insensitive to whether the MAD outlier-rejection branch
(``velocity_field.py`` lines 23-35) works at all — a regression there would not
flip the integration test. The window mean is the observable that the rejection
protects, so we pin it directly here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opennta.analysis.numerical_field.velocity_field import compute_velocity_field


def _single_window_df(good_dx, good_dy, *, outlier_dx, outlier_dy, n_outlier):
    """All points land in the one window of a 1x1 grid over [0, 100)^2."""
    dx = list(good_dx) + [outlier_dx] * n_outlier
    dy = list(good_dy) + [outlier_dy] * n_outlier
    n = len(dx)
    return pd.DataFrame({
        "ID": range(n),
        "X": [50.0] * n,
        "Y": [50.0] * n,
        "FRAME": [0] * n,
        "dX_um": dx,
        "dY_um": dy,
    })


def test_mad_outlier_rejection_protects_window_mean():
    # Good tracks spread symmetrically about (1.0, 2.0) so the MAD scale is
    # non-degenerate and the keep/reject decision genuinely depends on radius k.
    good_dx = np.linspace(0.8, 1.2, 20)   # mean exactly 1.0
    good_dy = np.linspace(1.6, 2.4, 20)   # mean exactly 2.0
    df = _single_window_df(good_dx, good_dy, outlier_dx=100.0, outlier_dy=-100.0, n_outlier=2)
    grid = dict(n_windows=1, x_range=(0.0, 100.0), y_range=(0.0, 100.0))

    # Normal radius: the two extreme tracks are rejected, so the window mean
    # equals the outlier-free truth (1.0, 2.0).
    kept = compute_velocity_field(df, outlier_k=4.0, **grid)
    assert kept.mean_dx[0, 0] == pytest.approx(1.0)
    assert kept.mean_dy[0, 0] == pytest.approx(2.0)

    # Disabling rejection (huge radius) lets the outliers through and corrupts
    # the mean, proving the rejection above is load-bearing rather than a no-op
    # the uniform-input integration test would ever have caught.
    contaminated = compute_velocity_field(df, outlier_k=1e9, **grid)
    assert contaminated.mean_dx[0, 0] == pytest.approx((20 * 1.0 + 2 * 100.0) / 22)
    assert contaminated.mean_dy[0, 0] == pytest.approx((20 * 2.0 + 2 * -100.0) / 22)
