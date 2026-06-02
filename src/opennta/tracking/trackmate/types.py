"""Shared TrackMate value types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy import optimize


@dataclass
class FitResult:
    params: dict[str, float]
    u_cut: float
    converged: bool
    raw_result: optimize.OptimizeResult

    def __getitem__(self, key: str) -> float:
        return self.params[key]


@dataclass
class ThresholdResult:
    ok: bool
    converged: bool
    params: dict[str, float]
    u_cut_fit_range: float
    u_star_alpha: float
    alpha: float
    plot_path: str | None
    fit_csv_path: str | None
    n_total: int
    n_ge_thresh: int
    n_lt_thresh: int
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "ok": self.ok,
            "converged": self.converged,
            "u_cut_fit_range": self.u_cut_fit_range,
            "u_star_alpha": self.u_star_alpha,
            "alpha": self.alpha,
            "plot_path": self.plot_path,
            "fit_csv_path": self.fit_csv_path,
            "n_total": self.n_total,
            "n_ge_thresh": self.n_ge_thresh,
            "n_lt_thresh": self.n_lt_thresh,
            "model_name": self.model_name,
        }
        result.update(self.params)
        return result


__all__ = ["FitResult", "ThresholdResult"]
