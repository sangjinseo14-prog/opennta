"""Finite Track Length Adjustment (Saveyn et al. 2010).

Saveyn, H. et al. "Accurate particle size distribution determination by
nanoparticle tracking analysis based on 2-D Brownian dynamics simulation",
J. Colloid Interface Sci. 352, 593-600 (2010).

Parametric MLE: fits an assumed size-distribution family (default: log-normal
in diameter) by maximizing the Gamma likelihood over its parameters. Smooth
but uni-modal; use :class:`IterativeDistributor` for polydisperse samples.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..size_distributor import SizeDistributor, register_distributor
from .families import get_family
from .likelihood import log_likelihood, log_prob_matrix
from .track_stats import extract_track_stats
from .types import PhysicalParams

if TYPE_CHECKING:
    from ..types import AnalysisConfig

logger = logging.getLogger(__name__)

_DEFAULT_LOC_ERR_NM = 5.0
_DEFAULT_FAMILY = "lognormal"

_EMPTY_RESULT: dict[str, Any] = {
    "diameters": None,
    "bin_edges": None,
    "counts": None,
    "bin_centers": None,
}


@register_distributor
class FTLADistributor(SizeDistributor):

    mode_key = "ftla"
    ui_label = "FTLA"

    loc_err_nm: float = _DEFAULT_LOC_ERR_NM
    family_name: str = _DEFAULT_FAMILY

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
                "FTLA requires msd_by_id, df, and config. "
                "Call distributor.compute(..., msd_by_id=..., df=..., config=...)."
            )

        # 1) Per-track (z_k, n_k) from MSD + preprocessed df.
        valid_ids = set(diameters_by_id.keys())
        msd_filtered = {pid: v for pid, v in msd_by_id.items() if pid in valid_ids}
        stats = extract_track_stats(msd_filtered, df)
        if len(stats) == 0:
            logger.warning("FTLA: no usable tracks after filtering")
            return dict(_EMPTY_RESULT)

        # 2) Diameter grid (log-spaced by default; log-normal has natural
        # support on log scale).
        lo = max(float(min_d), 1e-4)
        hi = max(float(max_d), lo + 1.0)
        if log_scale:
            bin_edges = np.logspace(np.log10(lo), np.log10(hi), int(num_bins) + 1)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        else:
            bin_edges = np.linspace(lo, hi, int(num_bins) + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        d_bin_m = bin_centers * 1e-9

        # 3) Gamma log-probability matrix log P(z_k | n_k, d_b).
        loc_err_m = self.loc_err_nm * 1e-9
        phys = PhysicalParams(
            temp_K=float(config.temp),
            eta_Pa_s=float(config.eta),
            frame_interval_s=float(config.dt),
            exposure_time_s=float(config.exposure_time),
            kB=float(config.KB),
            noise_m2=4.0 * loc_err_m * loc_err_m,
        )
        log_p = log_prob_matrix(stats, d_bin_m, phys)

        # 4) Build family and optimize parameters with L-BFGS-B.
        family = get_family(
            self.family_name,
            d_bin_nm=bin_centers,
            log_spaced_grid=bool(log_scale),
        )

        # Seed the optimizer with moments of the direct-conversion diameters.
        raw_diam = np.asarray(
            [d for d in diameters_by_id.values() if d > 0], dtype=np.float64
        )
        if raw_diam.size >= 2:
            log_d = np.log(raw_diam)
            stats_hint = {
                "log_d_mean": float(np.mean(log_d)),
                "log_d_std": float(max(np.std(log_d), 1e-2)),
            }
        else:
            stats_hint = None
        theta0 = family.initial_guess(stats_hint)

        def neg_ll(theta: np.ndarray) -> float:
            log_w = family.log_weights(np.asarray(theta, dtype=np.float64))
            return -log_likelihood(log_p, log_w)

        result = minimize(neg_ll, theta0, method="L-BFGS-B")
        theta_opt = np.asarray(result.x, dtype=np.float64)
        if not result.success:
            logger.info(
                "FTLA optimizer did not converge (message=%r); using best iterate",
                result.message,
            )

        log_w_opt = family.log_weights(theta_opt)
        weights = np.exp(log_w_opt)
        total = weights.sum()
        if total > 0 and np.isfinite(total):
            weights = weights / total

        # 5) Per-bin expected counts K * P_m.
        counts = float(len(stats)) * weights
        ll = -float(result.fun) if np.isfinite(result.fun) else float("nan")

        return {
            "diameters": None,
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "counts": counts,
            "fit": {
                "method": "saveyn2010-parametric",
                "family": self.family_name,
                "display_grid": "log" if log_scale else "linear",
                "n_tracks": int(len(stats)),
                "n_bins": int(weights.size),
                "converged": bool(result.success),
                "theta": theta_opt.tolist(),
                "log_likelihood": ll,
            },
        }


__all__ = ["FTLADistributor"]
