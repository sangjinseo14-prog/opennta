"""Tests for the parametric size-distribution families used by FTLA."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from opennta.analysis.mle.families import (
    LognormalFamily,
    available_families,
    get_family,
)


def _log_d_grid(lo: float = 10.0, hi: float = 1000.0, n: int = 200) -> np.ndarray:
    return np.logspace(np.log10(lo), np.log10(hi), n)


def test_available_families_contains_lognormal():
    assert "lognormal" in available_families()


def test_get_family_unknown_raises():
    with pytest.raises(ValueError, match="Unknown FTLA family"):
        get_family("does_not_exist", d_bin_nm=_log_d_grid())


def test_get_family_default_is_lognormal():
    fam = get_family("lognormal", d_bin_nm=_log_d_grid())
    assert isinstance(fam, LognormalFamily)
    assert fam.n_params == 2


def test_lognormal_rejects_non_positive_grid():
    with pytest.raises(ValueError):
        LognormalFamily(d_bin_nm=np.array([10.0, -5.0, 100.0]))


def test_lognormal_log_weights_normalize_to_one():
    fam = get_family("lognormal", d_bin_nm=_log_d_grid())
    theta = np.array([np.log(100.0), np.log(0.3)])
    log_w = fam.log_weights(theta)
    assert log_w.shape == (200,)
    assert logsumexp(log_w) == pytest.approx(0.0, abs=1e-9)


def test_lognormal_log_weights_peak_at_mu_on_log_grid():
    """On a log-spaced grid the discrete distribution peaks at d == exp(mu)."""
    d_grid = _log_d_grid()
    fam = LognormalFamily(d_bin_nm=d_grid, log_spaced_grid=True)
    mu = np.log(150.0)
    sigma = 0.2
    theta = np.array([mu, np.log(sigma)])
    log_w = fam.log_weights(theta)
    peak = d_grid[int(np.argmax(log_w))]
    # The peak is the bin closest to exp(mu) on a log grid.
    log_step = np.log(d_grid[1]) - np.log(d_grid[0])
    assert abs(np.log(peak) - mu) <= log_step


def test_lognormal_initial_guess_uses_hint():
    fam = get_family("lognormal", d_bin_nm=_log_d_grid())
    theta = fam.initial_guess({"log_d_mean": np.log(80.0), "log_d_std": 0.4})
    assert theta[0] == pytest.approx(np.log(80.0))
    assert theta[1] == pytest.approx(np.log(0.4))


def test_lognormal_linear_grid_changes_parametrization():
    """The 1/d factor in the log-normal pdf must be included on a linear-d
    grid and dropped on a log-d grid; the two grids therefore produce
    measurably different weight vectors for the same theta."""
    d_lin = np.linspace(50.0, 500.0, 200)
    d_log = _log_d_grid(50.0, 500.0, 200)
    theta = np.array([np.log(150.0), np.log(0.3)])

    w_lin = np.exp(LognormalFamily(d_lin, log_spaced_grid=False).log_weights(theta))
    w_log = np.exp(LognormalFamily(d_log, log_spaced_grid=True).log_weights(theta))
    # Same theta, different discretization → distributions differ.
    assert w_lin.sum() == pytest.approx(1.0)
    assert w_log.sum() == pytest.approx(1.0)
    # The argmax bin is in a similar location but the lin and log grids do
    # not share bin centers; verify the distributions are not identical.
    common_n = min(w_lin.size, w_log.size)
    assert not np.allclose(w_lin[:common_n], w_log[:common_n])
