"""End-to-end check for the registered ``"unet"`` (U-Net) drift corrector.

Replaces per-component unit tests of preprocessing / coordinate
transforms / cuda runtime / io / model weights with a single
result-oriented assertion: the registered corrector, given a synthetic
spots CSV, runs inference end-to-end and produces corrected per-spot
columns.

The U-Net runtime is optional — TensorFlow + the bundled weights are
not present in lean environments — so the test skips cleanly rather
than failing when those aren't installed.
"""
from __future__ import annotations

import pytest

from opennta.analysis.data_preprocessor import DataPreprocessor
from opennta.analysis.drift_corrector import (
    get_drift_corrector,
    registered_drift_correction_modes,
)
from opennta.tests._helpers import synth as _synth

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_unet_mode_is_registered():
    assert "unet" in registered_drift_correction_modes()


def test_unet_corrector_emits_correction_columns(analysis_config, tmp_path):
    pytest.importorskip("tensorflow")

    from opennta.analysis.unet_field.unet_bundle.config import WEIGHTS_PATH
    if not WEIGHTS_PATH.exists():
        pytest.skip(f"U-Net weights not available at {WEIGHTS_PATH}")

    df_spots = _synth.brownian_spots_df(
        analysis_config,
        drift_vx_um_per_s=1.0,
        drift_vy_um_per_s=-0.5,
        seed=11,
    )
    csv_path = _synth.write_spots_csv(df_spots, tmp_path, filename="unet.csv")

    df, _ = DataPreprocessor(analysis_config).load_and_compute_displacements(str(csv_path))

    corrector = get_drift_corrector(analysis_config, mode="unet")
    corrector.set_params(vector_n=8)
    corrector.fit(df, mean_vx=0.0, mean_vy=0.0)
    corrected = corrector.apply(df)

    assert {"X_diff_corr", "Y_diff_corr"}.issubset(corrected.columns)
    assert len(corrected) == len(df)
    assert corrected["X_diff_corr"].notna().all()
    assert corrected["Y_diff_corr"].notna().all()
