"""End-to-end check for the registered ``"numerical"`` drift corrector.

Replaces per-component unit tests of FieldSampler / FieldSmoother /
VelocityField with a single result-oriented assertion: the registered
corrector, given a synthetic spots CSV with a known uniform drift,
produces corrected per-spot displacements whose ensemble mean is close
to zero. Anything broken inside the numerical-field stack
(window-grid stats, smoothing, sampler interpolation) shows up here as
residual drift in the output.
"""
from __future__ import annotations

import numpy as np
import pytest

from opennta.analysis.data_preprocessor import DataPreprocessor
from opennta.analysis.drift_analyzer import DriftAnalyzer
from opennta.analysis.drift_corrector import (
    get_drift_corrector,
    registered_drift_correction_modes,
)
from opennta.analysis.numerical_field.types import NumericalFieldParams
from opennta.tests._helpers import synth as _synth

pytestmark = pytest.mark.integration


def _headless_params() -> NumericalFieldParams:
    return NumericalFieldParams(
        n_windows=4,
        x_range=(0.0, 500.0),
        y_range=(0.0, 500.0),
        outlier_k=3.0,
        min_count=1,
        ksize=3,
        n_iter=1,
        sigma=1.0,
    )


def test_numerical_mode_is_registered():
    assert "numerical" in registered_drift_correction_modes()


def test_numerical_corrector_removes_uniform_drift(analysis_config, tmp_path):
    df_spots = _synth.brownian_spots_df(
        analysis_config,
        drift_vx_um_per_s=2.0,
        drift_vy_um_per_s=-1.5,
        seed=3,
    )
    csv_path = _synth.write_spots_csv(df_spots, tmp_path, filename="numerical.csv")

    df, _ = DataPreprocessor(analysis_config).load_and_compute_displacements(str(csv_path))
    mean_vx, mean_vy, _, _ = DriftAnalyzer(analysis_config).calculate(df)

    corrector = get_drift_corrector(analysis_config, mode="numerical")
    corrector.set_params(_headless_params())
    # The *_field.csv export reads get_export_field(), which is populated by
    # fit() — not by set_params(). The Analysis tab must therefore export the
    # field after the worker runs, not right after configuring the dialog.
    assert corrector.get_export_field() is None
    corrector.fit(df, mean_vx, mean_vy)
    corrected = corrector.apply(df)

    assert {"X_diff_corr", "Y_diff_corr"}.issubset(corrected.columns)
    assert corrector.get_export_field() is not None

    post_vx, post_vy, _, _ = DriftAnalyzer(analysis_config).calculate(
        corrected, x_col="X_diff_corr", y_col="Y_diff_corr",
    )
    pre_speed = float(np.hypot(mean_vx, mean_vy))
    post_speed = float(np.hypot(post_vx, post_vy))

    assert post_speed < 0.25 * pre_speed
