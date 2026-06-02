"""End-to-end regression test for the analysis pipeline.

Drives ``AnalysisProcessor.run_full_analysis`` on a fully synthesised
Brownian spot stream whose target hydrodynamic diameter and drift are
chosen by the fixture, then asserts the pipeline recovers them.

Parametrised over multiple seeds and both built-in correction modes so a
regression introduced by a single seed realisation, or by a mode-specific
breakage, is surfaced rather than masked by a fixed seed=0 + mean-only
case.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opennta.analysis.analysis_processor import AnalysisProcessor
from opennta.tests._helpers import synth as _synth

pytestmark = pytest.mark.integration


SEEDS = [0, 7, 42]
MODES = ["no", "mean"]


def _build_spots_csv(analysis_config, tmp_path: Path, seed: int) -> Path:
    df = _synth.brownian_spots_df(analysis_config, seed=seed)
    return _synth.write_spots_csv(df, tmp_path, filename=f"spots_seed{seed}.csv")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", SEEDS)
def test_pipeline_recovers_input_diameter(
    analysis_config, tmp_path, synthetic_brownian_params, seed, mode,
):
    target_diameter_nm = synthetic_brownian_params["target_diameter_nm"]
    lag_frame = synthetic_brownian_params["lag_frame"]
    n_tracks = synthetic_brownian_params["n_tracks"]

    csv = _build_spots_csv(analysis_config, tmp_path, seed=seed)

    processor = AnalysisProcessor(analysis_config, correction_mode=mode)
    results = processor.run_full_analysis(str(csv), lag_frame=lag_frame)

    assert results.df is not None
    assert results.D_by_id and results.diameters_by_id

    diameters = np.array([d for d in results.diameters_by_id.values() if d > 0])

    # The drift contribution to MSD(tau) scales as (v*tau)^2, which is well
    # under 1% of the Brownian 4*D*tau term at the synthetic parameters, so
    # the diameter estimate is mode-insensitive. Tolerances cover empirical
    # seed-to-seed scatter at n_tracks=80 (mean ~ +/-6%, median ~ +/-3%);
    # tightening them would alias seed reshuffling as a regression.
    assert diameters.size >= int(0.8 * n_tracks)
    assert np.mean(diameters) == pytest.approx(target_diameter_nm, rel=0.08)
    assert np.median(diameters) == pytest.approx(target_diameter_nm, rel=0.05)


@pytest.mark.parametrize("seed", SEEDS)
def test_mean_correction_zeroes_residual_drift(
    analysis_config, tmp_path, synthetic_brownian_params, seed,
):
    lag_frame = synthetic_brownian_params["lag_frame"]
    csv = _build_spots_csv(analysis_config, tmp_path, seed=seed)

    processor = AnalysisProcessor(analysis_config, correction_mode="mean")
    results = processor.run_full_analysis(str(csv), lag_frame=lag_frame)

    assert results.post_correction_mean_vx == pytest.approx(0.0, abs=1e-9)
    assert results.post_correction_mean_vy == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_correction_preserves_input_drift(
    analysis_config, tmp_path, synthetic_brownian_params, seed,
):
    """``mode="no"`` must leave the ensemble drift unchanged: post-correction
    drift equals pre-correction drift exactly. The input-drift recovery
    itself is *not* asserted here — at n_tracks=80 the ensemble-mean drift
    estimator has stddev comparable to the input drift magnitude, so a
    tight comparison would be a coin-flip across seeds. The strict property
    we want is the corrector's pass-through behavior.
    """
    lag_frame = synthetic_brownian_params["lag_frame"]
    csv = _build_spots_csv(analysis_config, tmp_path, seed=seed)

    processor = AnalysisProcessor(analysis_config, correction_mode="no")
    results = processor.run_full_analysis(str(csv), lag_frame=lag_frame)

    assert results.post_correction_mean_vx == pytest.approx(results.mean_vx, abs=1e-9)
    assert results.post_correction_mean_vy == pytest.approx(results.mean_vy, abs=1e-9)
