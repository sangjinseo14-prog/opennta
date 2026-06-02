"""Unit tests for ``opennta.analysis.drift_analyzer.drift_analyzer``."""
from __future__ import annotations

import pandas as pd
import pytest

from opennta.analysis.drift_analyzer import DriftAnalyzer
from opennta.tests._helpers import synth as _synth


def test_drift_analyzer_recovers_known_velocity(analysis_config, tmp_path):
    df = _synth.drifting_displacements_df(analysis_config, tmp_path, vx_um_s=2.0, vy_um_s=-1.0)
    analyzer = DriftAnalyzer(analysis_config)
    mean_vx, mean_vy, speeds, drift_by_id = analyzer.calculate(df)

    assert speeds.shape[1] == 2
    assert speeds.shape[0] == df["ID"].nunique()
    assert mean_vx == pytest.approx(2.0, abs=0.1)
    assert mean_vy == pytest.approx(-1.0, abs=0.1)
    assert len(drift_by_id) == df["ID"].nunique()


def test_empty_dataframe_returns_zeros(default_analysis_config):
    empty = pd.DataFrame({
        "ID": pd.Series(dtype=int), "FRAME": pd.Series(dtype=int),
        "X_diff": pd.Series(dtype=float), "Y_diff": pd.Series(dtype=float),
    })
    mean_vx, mean_vy, speeds, drift_by_id = DriftAnalyzer(default_analysis_config).calculate(empty)
    assert mean_vx == 0.0 and mean_vy == 0.0
    assert speeds.shape == (0, 2)
    assert drift_by_id == {}


def test_single_row_track_skipped(default_analysis_config):
    df = pd.DataFrame({
        "ID": [1], "FRAME": [0], "X_diff": [0.0], "Y_diff": [0.0],
    })
    mean_vx, mean_vy, speeds, drift_by_id = DriftAnalyzer(default_analysis_config).calculate(df)
    # Tracks with <2 frames are skipped; we get the empty-array shape back.
    assert speeds.shape == (0, 2)
    assert drift_by_id == {}
    assert mean_vx == 0.0 and mean_vy == 0.0


def test_column_override(default_analysis_config):
    """Passing x_col/y_col should drive the velocity calculation off the
    chosen columns; otherwise the post-correction residual-drift check
    elsewhere in the suite would silently use the wrong columns."""
    cfg = default_analysis_config
    dt = cfg.dt
    df = pd.DataFrame({
        "ID": [1, 1, 1, 2, 2, 2],
        "FRAME": [0, 1, 2, 0, 1, 2],
        "X_diff": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Y_diff": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "X_diff_corr": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        "Y_diff_corr": [0.0, -1.0, -2.0, 0.0, -1.0, -2.0],
    })
    mean_vx, mean_vy, *_ = DriftAnalyzer(cfg).calculate(
        df, x_col="X_diff_corr", y_col="Y_diff_corr",
    )
    expected_vx = 2.0 / (2.0 * dt)
    expected_vy = -2.0 / (2.0 * dt)
    assert mean_vx == pytest.approx(expected_vx)
    assert mean_vy == pytest.approx(expected_vy)
