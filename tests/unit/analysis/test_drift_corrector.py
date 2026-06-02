"""Unit tests for ``opennta.analysis.drift_corrector`` (registry + plugins)."""
from __future__ import annotations

import numpy as np
import pytest

from opennta.analysis.drift_analyzer import DriftAnalyzer
from opennta.analysis.drift_corrector import (
    DriftCorrector,
    MeanCorrector,
    NoDriftCorrector,
    get_drift_corrector,
    registered_drift_correction_modes,
)
from opennta.tests._helpers import synth as _synth
from opennta.tests._helpers.factories import DRIFT_FACTORIES, SKIPPED_MODES

ALL_DRIFT_MODES = sorted(registered_drift_correction_modes())


def test_no_drift_corrector_passthrough(analysis_config, tmp_path):
    df = _synth.drifting_displacements_df(analysis_config, tmp_path, vx_um_s=1.0, vy_um_s=0.5)
    corr = NoDriftCorrector(analysis_config)
    corr.fit(df, mean_vx=0.0, mean_vy=0.0)
    out = corr.apply(df)

    assert np.allclose(out["X_diff_corr"], out["X_diff"])
    assert np.allclose(out["Y_diff_corr"], out["Y_diff"])


# Mode-specific: only `mean` can be checked headlessly. `no` is passthrough
# (covered by test_no_drift_corrector_passthrough); `numerical`/`unet` need a
# configurator/checkpoint and are listed in SKIPPED_MODES.
def test_mean_corrector_zeroes_residual_drift(analysis_config, tmp_path):
    df = _synth.drifting_displacements_df(analysis_config, tmp_path, vx_um_s=3.0, vy_um_s=-2.0)
    analyzer = DriftAnalyzer(analysis_config)
    mean_vx, mean_vy, _, _ = analyzer.calculate(df)

    corr = MeanCorrector(analysis_config)
    corr.fit(df, mean_vx=mean_vx, mean_vy=mean_vy)
    out = corr.apply(df)

    post_vx, post_vy, _, _ = analyzer.calculate(
        out, x_col="X_diff_corr", y_col="Y_diff_corr"
    )
    assert post_vx == pytest.approx(0.0, abs=1e-6)
    assert post_vy == pytest.approx(0.0, abs=1e-6)


# Note: the fit→apply lifecycle guard (apply-before-fit raises "not fitted")
# is asserted for every registered mode by test_registry_contract below, so a
# MeanCorrector-specific test is not duplicated here.


def test_get_drift_corrector_unknown_mode_raises(default_analysis_config):
    with pytest.raises(ValueError, match="Unknown drift correction mode"):
        get_drift_corrector(default_analysis_config, "does_not_exist")


@pytest.mark.parametrize("mode", ALL_DRIFT_MODES)
def test_registry_returns_subclass(default_analysis_config, mode):
    corr = get_drift_corrector(default_analysis_config, mode)
    assert isinstance(corr, DriftCorrector)
    assert corr.mode_key == mode
    assert corr.get_display_name()


@pytest.mark.parametrize("mode", ALL_DRIFT_MODES)
def test_registry_contract(default_analysis_config, tmp_path, mode):
    """Every registered corrector must honor the fit→apply lifecycle."""
    if mode in SKIPPED_MODES:
        pytest.skip(SKIPPED_MODES[mode])

    df = _synth.drifting_displacements_df(default_analysis_config, tmp_path, vx_um_s=1.0, vy_um_s=0.5)
    factory = DRIFT_FACTORIES.get(
        mode, lambda cfg, _mode=mode: get_drift_corrector(cfg, _mode)
    )
    corr = factory(default_analysis_config)

    with pytest.raises(RuntimeError, match="not fitted"):
        corr.apply(df)

    corr.fit(df, mean_vx=1.0, mean_vy=0.5)
    out = corr.apply(df)
    assert {"X_diff_corr", "Y_diff_corr"}.issubset(out.columns)
    assert len(out) == len(df)
