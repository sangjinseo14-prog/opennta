"""Gamma likelihood shared by the MLE-based size distributors.

For a 2D Brownian tracer of diameter ``d`` the mean squared per-frame
displacement is

    s(d) = 4 D(d) tau + noise_m2,    D(d) = k_B T / (3 pi eta d)

where ``noise_m2`` aggregates the localization-error contribution. Given
``n`` independent squared-displacement samples averaged into ``z``,

    z | n, d  ~  Gamma(shape = n, scale = s(d) / n).

We precompute a track-by-bin log-probability matrix so the outer optimizer
only has to mix bins with a size-distribution weight vector.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp

from .track_stats import TrackStats
from .types import PhysicalParams


def log_prob_matrix(
    stats: TrackStats,
    d_bin_m: NDArray[np.floating],
    phys: PhysicalParams,
) -> NDArray[np.floating]:
    # Compute log P(z_k | n_k, d_b) for every (track, bin) pair, shape (K, B).
    # Numerically stable Gamma log-pdf:
    #     log P = n*log(n) - n*log(s) + (n-1)*log(z) - n*z/s - lgamma(n)
    # The (n-1)*log(z) - lgamma(n) term depends only on the track and would
    # cancel in any per-track normalization, but it is retained so the returned
    # log-likelihood matches a true Gamma density.
    if len(stats) == 0 or d_bin_m.size == 0:
        return np.empty((len(stats), d_bin_m.size), dtype=np.float64)

    n = stats.n_steps.astype(np.float64)[:, None]   # (K, 1)
    z = stats.z_m2.astype(np.float64)[:, None]      # (K, 1)
    s = phys.msd_scale(d_bin_m.astype(np.float64))[None, :]  # (1, B)

    # Per-track constants
    track_const = (n - 1.0) * np.log(z) - gammaln(n)  # (K, 1)

    # Per-bin * per-track pieces
    log_p = (
        n * np.log(n)
        - n * np.log(s)
        - n * z / s
        + track_const
    )
    return log_p


def log_likelihood(
    log_p_matrix: NDArray[np.floating],
    log_weights: NDArray[np.floating],
) -> float:
    # log_weights must satisfy logsumexp(log_weights) == 0.
    # Returns sum_k log( sum_b P_kb * w_b ).
    if log_p_matrix.size == 0:
        return 0.0
    mixed = logsumexp(log_p_matrix + log_weights[None, :], axis=1)
    return float(np.sum(mixed))


def chi_squared(
    log_p_matrix: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> float:
    # Walker 2012 Eq. 15, implemented as the deviance against the per-track
    # saturated model that puts all mass on each track's best-fitting bin:
    #   chi^2(P) = 2 * sum_n [ max_m log P(eps_n | k_n, r_m)
    #                          - log sum_m P_m P(eps_n | k_n, r_m) ].
    # This form is strictly non-negative, guaranteed non-increasing under EM
    # (since log L is non-decreasing and the saturated term is constant in P),
    # and asymptotically chi^2 with B - 1 dof (Wilks). Therefore Walker's
    # Eq. 16 "1% decrease" rule is well-defined on it and fires only when
    # the fit has genuinely stopped improving.
    if log_p_matrix.size == 0 or weights.size == 0:
        return 0.0
    safe = np.where(weights > 0.0, weights, np.finfo(np.float64).tiny)
    log_w = np.log(safe)
    ll_sat = float(np.max(log_p_matrix, axis=1).sum())
    ll_p = log_likelihood(log_p_matrix, log_w)
    return float(max(2.0 * (ll_sat - ll_p), 0.0))


def em_step(
    log_p_matrix: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    # Walker 2012 Eq. 14 in log space (standard finite-mixture EM):
    #   gamma_km  = P_m * P_km / sum_m'(P_m' * P_km')        # E-step
    #   P_m_new   = (1/K) * sum_k gamma_km                    # M-step
    # logsumexp is used throughout for numerical stability.
    K, B = log_p_matrix.shape
    if K == 0 or B == 0:
        return weights.copy()
    safe = np.where(weights > 0.0, weights, np.finfo(np.float64).tiny)
    log_w = np.log(safe)                                     # (B,)
    joint = log_p_matrix + log_w[None, :]                    # (K, B)
    row_norm = logsumexp(joint, axis=1, keepdims=True)       # (K, 1)
    log_gamma = joint - row_norm                             # (K, B)
    # logsumexp over tracks (axis=0) then subtract log(K)
    log_new = logsumexp(log_gamma, axis=0) - np.log(float(K))
    new = np.exp(log_new)
    total = new.sum()
    if total > 0.0:
        new /= total
    return new


def em_iterate(
    log_p_matrix: NDArray[np.floating],
    *,
    initial_weights: NDArray[np.floating] | None = None,
    max_iter: int = 2000,
    tol_rel: float = 0.01,
    chi2_fn: Callable[[NDArray[np.floating]], float] | None = None,
) -> tuple[NDArray[np.floating], dict]:
    # Iterates Walker 2012 Eq. 14 until the Eq. 16 criterion fires.
    #
    # tol_rel:  Walker uses 0.01 (1%). The stop condition requires the
    #           relative decrease in the monitored statistic to be both
    #           non-negative and below tol_rel; a clearly negative move
    #           indicates numerical instability and never counts as
    #           converged.
    # chi2_fn:  Pass chi_squared() for the faithful Walker monitor. If
    #           None, the routine falls back to negative log-likelihood,
    #           which is also EM-monotone.
    #
    # Returns the best iterate seen (EM is theoretically monotone, but
    # tracking the best iterate makes the routine robust to fp drift) and
    # diagnostics: n_iter, converged, log_likelihood, chi2_history.
    K, B = log_p_matrix.shape
    if B == 0:
        return np.empty(0, dtype=np.float64), {
            "n_iter": 0, "converged": True, "log_likelihood": 0.0,
            "chi2_history": [],
        }
    if initial_weights is None:
        weights = np.full(B, 1.0 / B, dtype=np.float64)
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        total = weights.sum()
        if total <= 0.0 or not np.isfinite(total):
            weights = np.full(B, 1.0 / B, dtype=np.float64)
        else:
            weights /= total

    def _monitor(w: NDArray[np.floating]) -> float:
        if chi2_fn is not None:
            return float(chi2_fn(w))
        safe = np.where(w > 0.0, w, np.finfo(np.float64).tiny)
        return -float(log_likelihood(log_p_matrix, np.log(safe)))

    history: list[float] = []
    if K == 0:
        return weights, {
            "n_iter": 0, "converged": True, "log_likelihood": 0.0,
            "chi2_history": [],
        }

    stat_prev = _monitor(weights)
    history.append(stat_prev)

    # Track the best iterate so a late fp bump in the monitor cannot poison
    # the returned distribution.
    best_weights = weights.copy()
    best_stat = stat_prev

    # Tolerate fp jitter near the converged regime so a statistic wobbling
    # below the "truly converged" level still terminates rather than burning
    # iterations until max_iter.
    _FP_SLACK = 1e-12

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - final value reported as iteration count
        weights = em_step(log_p_matrix, weights)
        stat = _monitor(weights)
        history.append(stat)

        if stat < best_stat:
            best_stat = stat
            best_weights = weights.copy()

        # Walker Eq. 16: require relative improvement non-negative AND below
        # tol_rel. The non-negativity guard rejects fp upward wobbles that
        # would otherwise falsely report convergence after a few iterations.
        denom = abs(stat_prev) if stat_prev != 0.0 else 1.0
        rel_change = (stat_prev - stat) / denom
        if -_FP_SLACK <= rel_change < tol_rel:
            converged = True
            break
        stat_prev = stat

    weights = best_weights
    safe = np.where(weights > 0.0, weights, np.finfo(np.float64).tiny)
    ll = log_likelihood(log_p_matrix, np.log(safe))
    return weights, {
        "n_iter": n_iter,
        "converged": converged,
        "log_likelihood": ll,
        "chi2_history": history,
    }


__all__ = [
    "PhysicalParams",
    "chi_squared",
    "em_iterate",
    "em_step",
    "log_likelihood",
    "log_prob_matrix",
]
