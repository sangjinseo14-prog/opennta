"""Unit tests for ``FittingEngine`` — the truncated-MLE threshold fitter.

The engine fits a noise-bulk distribution to TrackMate peak qualities (upper-
truncated at a fit-range cutoff so genuine particles in the tail don't bias
the fit) and solves for the per-peak quality threshold at tail probability
``alpha``. This is pure numerics with no Fiji dependency; the test feeds it the
synthetic ``quality_samples`` bulk-plus-spike and asserts it recovers the known
bulk parameters and places the threshold between the bulk and the spike.
"""
from __future__ import annotations

import numpy as np
import pytest

from opennta.tests._helpers import synth as _synth
from opennta.tracking.trackmate.fitting import FittingEngine
from opennta.tracking.trackmate.types import FitResult, ThresholdResult


def test_gaussian_fit_recovers_bulk_parameters():
    values, _n_spike, bg_mu, bg_sigma, _spike = _synth.quality_samples(seed=1)
    engine = FittingEngine("Gaussian")

    result = engine.fit(values)

    assert isinstance(result, FitResult)
    assert result.converged
    # Truncated MLE recovers the parent bulk despite the high-quality spike,
    # which sits above the fit-range cutoff and is excluded from the likelihood.
    assert result["mu"] == pytest.approx(bg_mu, abs=0.5)
    assert result["sigma"] == pytest.approx(bg_sigma, rel=0.1)


@pytest.mark.parametrize("model_name", ["Gaussian", "Cheng-Schwartzman", "Poly2"])
def test_calculate_threshold_separates_bulk_from_spike(model_name):
    values, n_spike, bg_mu, bg_sigma, spike_value = _synth.quality_samples(seed=2)
    engine = FittingEngine(model_name)

    result = engine.calculate_threshold(values, alpha=1e-4)

    assert isinstance(result, ThresholdResult)
    assert result.ok and result.converged
    assert np.isfinite(result.u_star_alpha)
    # Threshold sits above the noise bulk and below the genuine-particle spike.
    assert bg_mu + 2.0 * bg_sigma < result.u_star_alpha < spike_value
    # Every spike sample is recovered; background leakage at alpha=1e-4 is tiny.
    assert result.n_ge_thresh >= n_spike
    assert result.n_ge_thresh <= n_spike + 5
    assert result.n_total == values.size
    assert result.n_ge_thresh + result.n_lt_thresh == result.n_total


def test_no_fitting_frac_uses_empirical_quantile_as_threshold():
    values = np.arange(1.0, 101.0)
    engine = FittingEngine("No Fitting (FRAC)")

    result = engine.calculate_threshold(values, frac=0.85, alpha=1e-4)

    assert result.ok and result.converged
    assert result.params == {}
    assert result.u_star_alpha == pytest.approx(np.quantile(values, 0.85))
    assert result.u_cut_fit_range == result.u_star_alpha
    assert result.n_ge_thresh == 15
    assert result.n_lt_thresh == 85
    assert result.plot_path is None
    assert result.fit_csv_path is None


def test_no_fitting_frac_rejects_out_of_range_fraction():
    engine = FittingEngine("No Fitting (FRAC)")

    with pytest.raises(ValueError, match="FRAC must be between 0 and 1"):
        engine.calculate_threshold(np.arange(10.0), frac=1.1, alpha=0.05)


def test_neg_log_likelihood_rejects_invalid_sigma():
    values, *_ = _synth.quality_samples(seed=3)
    engine = FittingEngine("Gaussian")
    u_upper = float(np.quantile(values, 0.9))

    # sigma <= 0 is invalid → sentinel penalty rather than a crash / -inf.
    bad = np.array([50.0, -1.0])
    assert engine.neg_log_likelihood_truncated(bad, values, u_upper) == pytest.approx(1e12)


def test_neg_log_likelihood_rejects_tiny_sample():
    engine = FittingEngine("Gaussian")
    tiny = np.linspace(40.0, 60.0, 10)  # < 30 points below the cutoff
    params = np.array([50.0, 5.0])
    assert engine.neg_log_likelihood_truncated(params, tiny, u_upper=100.0) == pytest.approx(1e12)


def test_solve_threshold_returns_nan_on_unbracketable_root():
    engine = FittingEngine("Gaussian")
    params = {"mu": 50.0, "sigma": 5.0}
    # On [lo, hi] both far in the upper tail, tail_prob - alpha never changes
    # sign for a large alpha, so brentq cannot bracket → graceful NaN.
    out = engine.solve_threshold_for_tail_prob(0.9, 80.0, 100.0, **params)
    assert np.isnan(out)


def test_exported_fit_csv_uses_unique_decimal_tenths(tmp_path):
    values, *_ = _synth.quality_samples(seed=4)
    engine = FittingEngine("Gaussian")
    result = engine.calculate_threshold(
        values,
        alpha=1e-4,
        quality_csv_path=str(tmp_path / "frame_quality.csv"),
    )
    exported = np.loadtxt(result.fit_csv_path, delimiter=",")
    grid = exported[:, 0]

    assert np.unique(grid).size == grid.size
    assert np.diff(grid) == pytest.approx(0.1)
    assert grid * 10 == pytest.approx(np.round(grid * 10))
