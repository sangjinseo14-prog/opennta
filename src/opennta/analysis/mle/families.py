"""Parameterized size-distribution families for the parametric FTLA distributor.

Each family maps an unconstrained parameter vector to per-bin log-weights.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


class DistributionFamily(ABC):
    # ``log_spaced_grid`` distinguishes linear-d grids (constant Δd) from
    # log-d grids (Δd_b scales as d_b). Per-bin probability is f(d_b) * Δd_b,
    # so families like log-normal whose pdf has a 1/d factor must change
    # parameterization between the two grid types to return correct weights.

    name: str = ""

    def __init__(
        self,
        d_bin_nm: NDArray[np.floating],
        *,
        log_spaced_grid: bool = True,
    ):
        d = np.asarray(d_bin_nm, dtype=np.float64)
        if d.size == 0 or np.any(d <= 0):
            raise ValueError("d_bin_nm must be positive and non-empty")
        self.d_bin_nm = d
        self.log_d = np.log(d)
        self.log_spaced_grid = bool(log_spaced_grid)

    @property
    @abstractmethod
    def n_params(self) -> int: ...

    @abstractmethod
    def initial_guess(self, stats_hint: dict | None = None) -> NDArray[np.floating]: ...

    @abstractmethod
    def log_weights(self, theta: NDArray[np.floating]) -> NDArray[np.floating]:
        # Returns per-bin log-probabilities normalized so logsumexp == 0.
        ...


def _normalize_log(log_w: NDArray[np.floating]) -> NDArray[np.floating]:
    return log_w - logsumexp(log_w)


class LognormalFamily(DistributionFamily):
    # mu = mean of ln(d) (d in nm); sigma = exp(log_sigma) = std of ln(d).
    # Both parameters are real-valued so the optimizer stays unconstrained.

    name = "lognormal"
    n_params = 2

    def initial_guess(self, stats_hint: dict | None = None) -> NDArray[np.floating]:
        if stats_hint and "log_d_mean" in stats_hint:
            mu0 = float(stats_hint["log_d_mean"])
        else:
            mu0 = float(np.mean(self.log_d))
        if stats_hint and "log_d_std" in stats_hint:
            sig0 = max(float(stats_hint["log_d_std"]), 1e-2)
        else:
            sig0 = 0.5
        return np.array([mu0, np.log(sig0)], dtype=np.float64)

    def log_weights(self, theta: NDArray[np.floating]) -> NDArray[np.floating]:
        mu = float(theta[0])
        log_sigma = float(theta[1])
        # Clamp sigma to avoid underflow for extreme optimizer moves.
        sigma = float(np.exp(np.clip(log_sigma, -6.0, 4.0)))
        z = (self.log_d - mu) / sigma
        # Per-bin probability is ``f(d_b) * Delta d_b``. The log-normal pdf
        # gives ``f(d) ~ (1/d) * exp(-z^2/2)``; on a log-spaced grid
        # ``Delta d_b ~ d_b`` cancels the ``1/d`` and the ``-log(d)`` term
        # must be dropped. On a linear grid ``Delta d_b`` is constant and
        # ``-log(d)`` belongs. Mixing these up gives a mu shift of sigma^2.
        if self.log_spaced_grid:
            log_w = -0.5 * z * z
        else:
            log_w = -0.5 * z * z - self.log_d
        return _normalize_log(log_w)


_REGISTRY: dict[str, type[DistributionFamily]] = {
    LognormalFamily.name: LognormalFamily,
}


def get_family(
    name: str,
    d_bin_nm: NDArray[np.floating],
    *,
    log_spaced_grid: bool = True,
) -> DistributionFamily:
    key = (name or "lognormal").lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown FTLA family {name!r}. Available: {available}")
    return _REGISTRY[key](d_bin_nm, log_spaced_grid=log_spaced_grid)


def available_families() -> list[str]:
    return list(_REGISTRY)


__all__ = [
    "DistributionFamily",
    "LognormalFamily",
    "get_family",
    "available_families",
]
