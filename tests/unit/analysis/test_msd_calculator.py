"""Unit tests for ``opennta.analysis.msd_calculator.msd_calculator``."""
from __future__ import annotations

import numpy as np
import pandas as pd

from opennta.analysis.msd_calculator import MSDCalculator
from opennta.tests._helpers.synth import brownian_tracks_df_corr


def test_calculate_returns_lag_msd_pairs(default_analysis_config):
    """The output structure: ``{pid: (lags, msds)}`` where ``lags[0] == 1``
    and ``lags.shape == msds.shape``."""
    cfg = default_analysis_config
    df = brownian_tracks_df_corr(D_um2_per_s=0.5, dt=cfg.dt, n_tracks=20, n_frames=120)
    msd_by_id = MSDCalculator(cfg).calculate(df, lag_frame=4)
    assert len(msd_by_id) == df["ID"].nunique()
    for pid, (lags, msds) in msd_by_id.items():
        assert lags.shape == msds.shape
        assert int(lags[0]) == 1


def test_calculate_empty_dataframe(default_analysis_config):
    empty = pd.DataFrame({"ID": [], "X_diff_corr": [], "Y_diff_corr": [], "FRAME": []})
    msd_by_id = MSDCalculator(default_analysis_config).calculate(empty, lag_frame=2)
    assert msd_by_id == {}


def test_calculate_single_frame_track_yields_no_msd(default_analysis_config):
    """A track with only one observation has no pairs to compute MSD."""
    df = pd.DataFrame({
        "ID": [1], "X_diff_corr": [0.0], "Y_diff_corr": [0.0], "FRAME": [0],
    })
    msd_by_id = MSDCalculator(default_analysis_config).calculate(df, lag_frame=2)
    assert msd_by_id == {}


def test_calculate_handles_nan_coordinates(default_analysis_config):
    """A track with NaN coordinates yields NaN MSDs but the calculator
    must not crash, and other tracks should still be returned."""
    df = pd.DataFrame({
        "ID": [1, 1, 1, 2, 2, 2],
        "X_diff_corr": [0.0, 1.0, 2.0, 0.0, np.nan, 2.0],
        "Y_diff_corr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "FRAME": [0, 1, 2, 0, 1, 2],
    })
    msd_by_id = MSDCalculator(default_analysis_config).calculate(df, lag_frame=2)
    assert 1 in msd_by_id
    assert np.all(np.isfinite(msd_by_id[1][1]))
    assert 2 in msd_by_id
    assert np.isnan(msd_by_id[2][1]).any()


def test_r2_per_track_in_unit_range(default_analysis_config):
    cfg = default_analysis_config
    df = brownian_tracks_df_corr(D_um2_per_s=0.3, dt=cfg.dt, n_tracks=20, n_frames=200)
    msd_by_id = MSDCalculator(cfg).calculate(df, lag_frame=4)
    r2 = MSDCalculator(cfg).calculate_r2_per_track(msd_by_id, lag_frame=4)

    finite = [v for v in r2.values() if np.isfinite(v)]
    assert finite, "expected at least one finite R^2 value"
    assert all(-1.0 <= v <= 1.0 for v in finite)


def test_r2_too_few_points_is_nan(default_analysis_config):
    msd_by_id = {1: (np.array([1, 2]), np.array([0.1, 0.2]))}
    r2 = MSDCalculator(default_analysis_config).calculate_r2_per_track(
        msd_by_id, lag_frame=2, min_points=3
    )
    assert np.isnan(r2[1])


def test_r2_high_for_linear_msd(default_analysis_config):
    """Pure Brownian + a linear MSD ground truth gives R^2 near 1 on the
    fitted lag window. Catches regressions that silently flatten R^2."""
    cfg = default_analysis_config
    df = brownian_tracks_df_corr(
        D_um2_per_s=0.3, dt=cfg.dt, n_tracks=60, n_frames=400, seed=7
    )
    msd_by_id = MSDCalculator(cfg).calculate(df, lag_frame=4)
    r2 = MSDCalculator(cfg).calculate_r2_per_track(msd_by_id, lag_frame=4)
    finite = np.array([v for v in r2.values() if np.isfinite(v)])
    assert finite.size > 0
    assert np.median(finite) > 0.85


def test_r2_flat_msd_yields_nan(default_analysis_config):
    """If all sampled MSD points are equal, R^2 is undefined → NaN."""
    msd_by_id = {1: (np.array([1, 2, 3, 4, 5]), np.array([0.5, 0.5, 0.5, 0.5, 0.5]))}
    r2 = MSDCalculator(default_analysis_config).calculate_r2_per_track(
        msd_by_id, lag_frame=2
    )
    assert np.isnan(r2[1])
