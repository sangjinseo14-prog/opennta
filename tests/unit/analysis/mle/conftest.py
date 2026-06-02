"""Shared fixtures for the MLE distributor tests.

``monodisperse_pipeline_state`` runs the deterministic analysis pipeline up
to the (msd_by_id, diameters_by_id) stage once per module so both the
iterative and FTLA distributor test files can be exercised against the
same ground-truth Brownian ensemble.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from opennta.analysis.data_preprocessor import DataPreprocessor
from opennta.analysis.diffusion_estimator import DiffusionEstimator
from opennta.analysis.drift_corrector import NoDriftCorrector
from opennta.analysis.msd_calculator import MSDCalculator
from opennta.analysis.types import AnalysisConfig
from opennta.tests._helpers import synth as _synth


@pytest.fixture(scope="module")
def analysis_config_module() -> AnalysisConfig:
    """Module-scoped clone of the session ``analysis_config`` so that the
    module-scoped ``monodisperse_pipeline_state`` fixture below can depend
    on it."""
    return AnalysisConfig(
        sensor_size=6.5, magnification=20, fps=25.0,
        temp=298.0, eta=0.00089,
    )


@pytest.fixture(scope="module")
def monodisperse_pipeline_state(analysis_config_module):
    cfg = analysis_config_module
    spots = _synth.brownian_spots_df(
        cfg,
        target_diameter_nm=100.0,
        n_tracks=120,
        n_frames=200,
        drift_vx_um_per_s=0.0,
        drift_vy_um_per_s=0.0,
        seed=11,
    )
    with tempfile.TemporaryDirectory() as td:
        csv = _synth.write_spots_csv(spots, Path(td))
        df, _ = DataPreprocessor(cfg).load_and_compute_displacements(str(csv))

    corr = NoDriftCorrector(cfg)
    corr.fit(df, mean_vx=0.0, mean_vy=0.0)
    df_corr = corr.apply(df)

    lag_frame = 4
    msd_by_id = MSDCalculator(cfg).calculate(df_corr, lag_frame=lag_frame)
    _, diameters_by_id, _, _ = DiffusionEstimator(cfg).estimate(msd_by_id, lag_frame=lag_frame)

    return {
        "config": cfg,
        "df": df_corr,
        "msd_by_id": msd_by_id,
        "diameters_by_id": diameters_by_id,
    }
