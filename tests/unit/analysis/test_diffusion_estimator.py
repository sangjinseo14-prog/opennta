"""Unit tests for ``opennta.analysis.diffusion_estimator.diffusion_estimator``."""
from __future__ import annotations

import numpy as np
import pytest

from opennta.analysis.diffusion_estimator import DiffusionEstimator
from opennta.analysis.msd_calculator import MSDCalculator
from opennta.analysis.types import AnalysisConfig
from opennta.tests._helpers.synth import brownian_tracks_df_corr


def test_estimate_recovers_known_diffusion(default_analysis_config):
    cfg = default_analysis_config
    D_true = 0.5
    df = brownian_tracks_df_corr(D_um2_per_s=D_true, dt=cfg.dt, n_tracks=80, n_frames=300)
    msd_by_id = MSDCalculator(cfg).calculate(df, lag_frame=4)

    D_by_id, diameters_by_id, insufficient, nonpositive_d = DiffusionEstimator(cfg).estimate(
        msd_by_id, lag_frame=4
    )
    assert nonpositive_d == []
    D_values = np.fromiter(D_by_id.values(), dtype=float)
    assert D_values.size > 0
    assert np.median(D_values) == pytest.approx(D_true, rel=0.2)
    assert insufficient == []
    assert len(diameters_by_id) == len(D_by_id)
    assert all(d > 0 for d in diameters_by_id.values())

    # Diameter must be the Stokes-Einstein value, not merely positive: verify
    # the exact per-track conversion d_nm = KB*T / (3*pi*eta*D_m2s) * 1e9, and
    # that the recovered diameter distribution centres on the value implied by
    # D_true. (diameter is a monotone fn of D, so median(d) = SE(median(D)).)
    def stokes_einstein_nm(D_um2_s: float) -> float:
        return (cfg.KB * cfg.temp) / (3 * np.pi * cfg.eta * (D_um2_s * 1e-12)) * 1e9

    for pid, D in D_by_id.items():
        assert diameters_by_id[pid] == pytest.approx(stokes_einstein_nm(D))
    diam_values = np.fromiter(diameters_by_id.values(), dtype=float)
    assert np.median(diam_values) == pytest.approx(stokes_einstein_nm(D_true), rel=0.25)


def test_estimate_buckets_each_track_by_fit_outcome(default_analysis_config):
    """``estimate`` partitions a per-id MSD map into four outputs; one mixed
    input with a single track per bucket pins the whole contract at once.

    A non-positive slope is *recorded* in ``nonpositive_d`` rather than dropped:
    silently folding such tracks away (common for slow/large or noisy tracks)
    would bias the recovered size distribution toward smaller sizes.
    """
    msd_by_id = {
        1: (np.array([1]), np.array([0.1])),                          # 1 pt < 2 -> insufficient
        2: (np.array([1, 2, 3, 4]), np.array([0.1, 0.2, 0.3, 0.4])),  # slope > 0 -> D + diameter
        99: (np.array([1, 2, 3, 4]), np.array([1.0, 0.9, 0.8, 0.7])),  # slope <= 0 -> nonpositive_d
    }

    D_by_id, diameters_by_id, insufficient, nonpositive_d = DiffusionEstimator(
        default_analysis_config
    ).estimate(msd_by_id, lag_frame=4)

    # Only the positive-slope track yields a diffusion coefficient, and a
    # diameter is produced for exactly those surviving ids.
    assert set(D_by_id) == {2} and D_by_id[2] > 0
    assert set(diameters_by_id) == {2} and diameters_by_id[2] > 0
    # The 1-point track is too short to fit: use_n = min(4, 1) = 1 < 2.
    assert insufficient == [(1, 1)]
    # The decreasing-MSD track is surfaced with its non-positive D, not dropped.
    assert [pid for pid, _ in nonpositive_d] == [99]
    assert nonpositive_d[0][1] <= 0.0


def test_empty_msd_returns_empty(default_analysis_config):
    D_by_id, diameters_by_id, insufficient, nonpositive_d = DiffusionEstimator(
        default_analysis_config
    ).estimate({}, lag_frame=4)
    assert D_by_id == {}
    assert diameters_by_id == {}
    assert insufficient == []
    assert nonpositive_d == []


def test_uses_at_most_lag_frame_points(default_analysis_config):
    """Slope is fitted over the first ``lag_frame`` lags. Extending the
    MSD beyond ``lag_frame`` must not change the recovered D."""
    short = {1: (np.array([1, 2]), np.array([0.10, 0.20]))}
    long = {1: (np.array([1, 2, 3, 4, 5]),
                np.array([0.10, 0.20, 99.0, -99.0, 50.0]))}
    D_short, *_ = DiffusionEstimator(default_analysis_config).estimate(short, lag_frame=2)
    D_long, *_ = DiffusionEstimator(default_analysis_config).estimate(long, lag_frame=2)
    assert D_short[1] == pytest.approx(D_long[1])


def test_blur_shifted_multilag_msd_recovers_diffusion():
    config = AnalysisConfig(fps=25.0, exposure_time=0.040)
    diffusion = 0.5
    lags = np.arange(1, 5)
    msds = 4.0 * diffusion * (
        lags * config.dt - config.exposure_time / 3.0
    ) + 0.02

    estimated, *_ = DiffusionEstimator(config).estimate(
        {1: (lags, msds)}, lag_frame=4
    )

    assert estimated[1] == pytest.approx(diffusion)
