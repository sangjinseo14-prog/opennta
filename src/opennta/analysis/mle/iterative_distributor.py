"""Iterative size-distribution reconstruction (Walker 2012).

Walker, J.G. "Improved nano-particle tracking analysis",
Meas. Sci. Technol. 23, 065605 (2012).

Non-parametric MLE: recovers the discrete distribution P_m over a diameter
grid by iterating Walker Eq. 14 until Eq. 16 (relative chi-squared change
< 1%) fires. Multimodal-friendly; use :class:`FTLADistributor` for smoother
but uni-modal fits.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..size_distributor import SizeDistributor, register_distributor
from .likelihood import (
    chi_squared,
    em_iterate,
    log_prob_matrix,
)
from .track_stats import extract_track_stats
from .types import PhysicalParams

if TYPE_CHECKING:
    from ..types import AnalysisConfig

logger = logging.getLogger(__name__)

_DEFAULT_LOC_ERR_NM = 5.0
_DEFAULT_TOL_REL = 0.01       # Walker Eq. 16: 1%
_DEFAULT_MAX_ITER = 2000

_EMPTY_RESULT: dict[str, Any] = {
    "diameters": None,
    "bin_edges": None,
    "counts": None,
    "bin_centers": None,
}


@register_distributor
class IterativeDistributor(SizeDistributor):

    mode_key = "iterative"
    ui_label = "Iterative"

    loc_err_nm: float = _DEFAULT_LOC_ERR_NM
    tol_rel: float = _DEFAULT_TOL_REL
    max_iter: int = _DEFAULT_MAX_ITER

    def compute(
        self,
        diameters_by_id: dict[int, float] | None,
        min_d: float = 0.1,
        max_d: float = 1000.0,
        num_bins: int = 100,
        *,
        msd_by_id: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]] | None = None,
        df: pd.DataFrame | None = None,
        config: AnalysisConfig | None = None,
        log_scale: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        if not diameters_by_id:
            return dict(_EMPTY_RESULT)
        if msd_by_id is None or df is None or config is None:
            raise ValueError(
                "Iterative method requires msd_by_id, df, and config. "
                "Call distributor.compute(..., msd_by_id=..., df=..., config=...)."
            )

        # 1) Per-track (z_k, n_k), restricted to IDs that survived R^2 filtering.
        valid_ids = set(diameters_by_id.keys())
        msd_filtered = {pid: v for pid, v in msd_by_id.items() if pid in valid_ids}
        stats = extract_track_stats(msd_filtered, df)
        if len(stats) == 0:
            logger.warning("Iterative: no usable tracks after filtering")
            return dict(_EMPTY_RESULT)

        # 2) Diameter grid. Walker's convergence analysis only requires equal
        # spacing in the grid variable, so a linear-d grid (Walker original)
        # or a linear-log(d) grid both work. NTA diameters span multiple
        # decades, so a linear-d grid with 100 bins on [1, 10000] nm makes
        # the first bin ~100 nm wide and collapses all sub-100-nm particles
        # into bin 0; the log-spaced grid keeps relative resolution constant
        # and matches the display grid, avoiding any rebinning step.
        if log_scale:
            lo = max(float(min_d), 1e-4)
            hi = max(float(max_d), lo + 1.0)
            algo_edges_nm = np.logspace(
                np.log10(lo), np.log10(hi), int(num_bins) + 1
            )
            algo_centers_nm = np.sqrt(algo_edges_nm[:-1] * algo_edges_nm[1:])
        else:
            lo = max(float(min_d), 0.0)
            hi = max(float(max_d), lo + 1.0)
            algo_edges_nm = np.linspace(lo, hi, int(num_bins) + 1)
            algo_centers_nm = 0.5 * (algo_edges_nm[:-1] + algo_edges_nm[1:])
        d_bin_m = algo_centers_nm * 1e-9

        # 3) Gamma log-probability matrix (constant over iterations).
        loc_err_m = self.loc_err_nm * 1e-9
        phys = PhysicalParams(
            temp_K=float(config.temp),
            eta_Pa_s=float(config.eta),
            frame_interval_s=float(config.dt),
            exposure_time_s=float(config.exposure_time),
            kB=float(config.KB),
            # 2D displacement with per-coord SD sigma_e contributes 4 sigma_e^2.
            noise_m2=4.0 * loc_err_m * loc_err_m,
        )
        log_p = log_prob_matrix(stats, d_bin_m, phys)

        # 4) Walker Eq. 14 with uniform init; stop on Eq. 16.
        weights, info = em_iterate(
            log_p,
            initial_weights=None,
            max_iter=int(self.max_iter),
            tol_rel=float(self.tol_rel),
            chi2_fn=lambda w: chi_squared(log_p, w),
        )
        if not info["converged"]:
            logger.info(
                "Iterative EM did not converge within max_iter=%d (final chi2=%.3g)",
                self.max_iter,
                info["chi2_history"][-1] if info["chi2_history"] else float("nan"),
            )

        # 5) Per-bin expected counts K * P_m on the algorithm grid; the
        # algorithm grid already matches the display grid so no rebinning.
        algo_counts = float(len(stats)) * weights
        disp_edges_nm = algo_edges_nm
        disp_centers_nm = algo_centers_nm
        disp_counts = algo_counts

        return {
            "diameters": None,
            "bin_edges": disp_edges_nm,
            "bin_centers": disp_centers_nm,
            "counts": disp_counts,
            "fit": {
                "method": "walker2012-em",
                "algo_grid": "log" if log_scale else "linear",
                "display_grid": "log" if log_scale else "linear",
                "n_tracks": int(len(stats)),
                "n_bins": int(weights.size),
                "n_iter": int(info["n_iter"]),
                "converged": bool(info["converged"]),
                "log_likelihood": float(info["log_likelihood"]),
                "chi2_final": (
                    float(info["chi2_history"][-1]) if info["chi2_history"] else None
                ),
            },
        }


__all__ = ["IterativeDistributor"]
