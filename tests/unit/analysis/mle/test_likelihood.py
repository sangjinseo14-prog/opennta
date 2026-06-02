"""Unit tests for ``opennta.analysis.mle.likelihood``.

The Gamma model is z_k | n_k, d ~ Gamma(shape=n_k, scale=s(d)/n_k) with
mean s(d) = 4 D(d) tau + noise, so for n_k fixed and noise=0 the
likelihood as a function of d peaks at d* such that s(d*) = z_k, which we
use as analytic ground truth.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from opennta.analysis.mle.likelihood import (
    chi_squared,
    em_iterate,
    em_step,
    log_likelihood,
    log_prob_matrix,
)
from opennta.analysis.mle.track_stats import TrackStats
from opennta.analysis.mle.types import PhysicalParams


def _phys(noise_m2: float = 0.0) -> PhysicalParams:
    return PhysicalParams(
        temp_K=298.0, eta_Pa_s=8.9e-4, tau_s=1.0 / 30.0, noise_m2=noise_m2
    )


def _d_grid_nm(lo: float = 10.0, hi: float = 1000.0, n: int = 200) -> np.ndarray:
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _z_for_diameter_nm(phys: PhysicalParams, d_nm: float) -> float:
    return float(phys.msd_scale(np.asarray([d_nm * 1e-9]))[0])


def test_log_prob_matrix_peaks_at_true_diameter():
    """log P(z|n,d) as a function of d peaks at d such that s(d) = z."""
    phys = _phys()
    d_grid_nm = _d_grid_nm()
    d_grid_m = d_grid_nm * 1e-9
    d_true_nm = 120.0
    z_m2 = _z_for_diameter_nm(phys, d_true_nm)

    stats = TrackStats(
        pids=np.array([1], dtype=np.int64),
        z_m2=np.array([z_m2], dtype=np.float64),
        n_steps=np.array([50], dtype=np.int64),
    )
    log_p = log_prob_matrix(stats, d_grid_m, phys)
    assert log_p.shape == (1, d_grid_nm.size)

    argmax = int(np.argmax(log_p[0]))
    # The peak should be within one logarithmic bin of d_true.
    bin_width = np.log(d_grid_nm[1]) - np.log(d_grid_nm[0])
    assert abs(np.log(d_grid_nm[argmax]) - np.log(d_true_nm)) <= bin_width


def test_log_prob_matrix_empty_inputs():
    phys = _phys()
    empty_stats = TrackStats(
        pids=np.empty(0, dtype=np.int64),
        z_m2=np.empty(0, dtype=np.float64),
        n_steps=np.empty(0, dtype=np.int64),
    )
    assert log_prob_matrix(empty_stats, np.array([100e-9]), phys).shape == (0, 1)

    stats = TrackStats(
        pids=np.array([1], dtype=np.int64),
        z_m2=np.array([1e-16], dtype=np.float64),
        n_steps=np.array([10], dtype=np.int64),
    )
    assert log_prob_matrix(stats, np.empty(0, dtype=np.float64), phys).shape == (1, 0)


def test_log_likelihood_matches_logsumexp_definition():
    """sum_k log( sum_b P_kb * w_b ) reproduces a direct logsumexp."""
    rng = np.random.default_rng(0)
    log_p = rng.standard_normal((5, 3))
    w = np.array([0.5, 0.3, 0.2])
    log_w = np.log(w)

    expected = float(np.sum(logsumexp(log_p + log_w[None, :], axis=1)))
    assert log_likelihood(log_p, log_w) == pytest.approx(expected)


def test_log_likelihood_empty_matrix_is_zero():
    log_p = np.empty((0, 0))
    assert log_likelihood(log_p, np.empty(0)) == 0.0


def test_chi_squared_zero_when_weights_concentrate_on_argmax():
    """If P_m is a delta at each track's argmax bin (so weights are uniform
    across that single bin only), chi^2 collapses to 0 by construction."""
    log_p = np.array([
        [-1.0, -10.0, -10.0],
        [-10.0, -1.0, -10.0],
        [-10.0, -10.0, -1.0],
    ])
    weights = np.array([1.0, 1.0, 1.0]) / 3.0
    val = chi_squared(log_p, weights)
    assert val > 0.0

    log_p_concentrated = np.array([
        [-1.0, -10.0],
        [-1.0, -10.0],
        [-1.0, -10.0],
    ])
    weights_delta = np.array([1.0, 0.0])
    assert chi_squared(log_p_concentrated, weights_delta) == pytest.approx(0.0, abs=1e-9)


def test_chi_squared_non_negative():
    rng = np.random.default_rng(1)
    log_p = rng.standard_normal((10, 4))
    w = rng.dirichlet(np.ones(4))
    assert chi_squared(log_p, w) >= 0.0


def test_em_step_is_log_likelihood_monotone():
    """Walker Eq. 14 must not decrease the log-likelihood."""
    rng = np.random.default_rng(2)
    K, B = 30, 6
    log_p = rng.standard_normal((K, B))
    w = rng.dirichlet(np.ones(B))

    ll_before = log_likelihood(log_p, np.log(np.maximum(w, 1e-300)))
    w_next = em_step(log_p, w)
    ll_after = log_likelihood(log_p, np.log(np.maximum(w_next, 1e-300)))

    assert w_next.sum() == pytest.approx(1.0)
    assert ll_after + 1e-9 >= ll_before


def test_em_iterate_recovers_monodisperse_distribution():
    """Build many tracks all from the same diameter; EM should put almost
    all of the weight on the corresponding grid bin."""
    phys = _phys()
    d_grid_nm = _d_grid_nm(n=60)
    d_grid_m = d_grid_nm * 1e-9
    d_true_nm = 150.0
    z_true = _z_for_diameter_nm(phys, d_true_nm)

    K = 100
    stats = TrackStats(
        pids=np.arange(1, K + 1, dtype=np.int64),
        z_m2=np.full(K, z_true, dtype=np.float64),
        n_steps=np.full(K, 50, dtype=np.int64),
    )
    log_p = log_prob_matrix(stats, d_grid_m, phys)
    weights, info = em_iterate(
        log_p,
        max_iter=500,
        tol_rel=0.01,
        chi2_fn=lambda w: chi_squared(log_p, w),
    )
    assert weights.shape == (d_grid_nm.size,)
    assert weights.sum() == pytest.approx(1.0)
    assert info["n_iter"] > 0

    peak_nm = d_grid_nm[int(np.argmax(weights))]
    assert peak_nm == pytest.approx(d_true_nm, rel=0.10)

    in_band = (d_grid_nm > d_true_nm / 1.5) & (d_grid_nm < d_true_nm * 1.5)
    assert weights[in_band].sum() > 0.5


def test_em_iterate_empty_grid_returns_empty_weights():
    log_p = np.empty((10, 0))
    weights, info = em_iterate(log_p)
    assert weights.shape == (0,)
    assert info["converged"] is True
    assert info["n_iter"] == 0


def test_em_iterate_no_tracks_returns_initial_weights():
    log_p = np.empty((0, 5))
    weights, info = em_iterate(log_p)
    assert weights.shape == (5,)
    assert info["n_iter"] == 0
