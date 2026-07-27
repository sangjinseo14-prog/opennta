from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .fitting_models import TAIL_SUPPORT_Z_UPPER, get_model_class
from .types import FitResult, ThresholdResult

ModelType = Any
logger = logging.getLogger(__name__)
FIT_CURVE_X_STEP = 0.1


class FittingEngine:

    def __init__(self, model: str | ModelType = "Cheng-Schwartzman"):
        if isinstance(model, str):
            model_class = get_model_class(model)
            self.model = model_class()
        else:
            self.model = model

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def param_names(self) -> list[str]:
        return self.model.param_names

    def _params_to_dict(self, params_array: NDArray[np.floating]) -> dict[str, float]:
        return {name: float(val) for name, val in zip(self.param_names, params_array, strict=False)}

    def log_pdf_untruncated(self, u: NDArray[np.floating], **params) -> NDArray[np.floating]:
        dens = self.model.pdf_untruncated(u, **params)
        dens_safe = np.maximum(dens, 1e-300)
        return np.log(dens_safe)

    def _get_truncation_normalizer(self, u_upper: float, **params) -> float:
        if hasattr(self.model, 'truncation_normalizer'):
            return self.model.truncation_normalizer(u_upper, **params)

        try:
            result, _ = integrate.quad(
                lambda x: float(self.model.pdf_untruncated(np.array([x]), **params)[0]),
                -np.inf, float(u_upper),
                limit=200
            )
            return max(float(result), 1e-12)
        except Exception as exc:
            logger.warning("Truncated normalizer integration failed at u_upper=%s: %s", u_upper, exc)
            return 1e-12

    def _validate_params(self, params: dict[str, float]) -> bool:
        if hasattr(self.model, 'validate_params'):
            return self.model.validate_params(params)
        return all(np.isfinite(v) for v in params.values())

    def _regularization_penalty(self, params: dict[str, float]) -> float:
        return 1e-6 * sum(v*v for v in params.values())

    def neg_log_likelihood_truncated(self, params_array: NDArray[np.floating], data: NDArray[np.floating], u_upper: float) -> float:
        params = self._params_to_dict(params_array)

        if not self._validate_params(params):
            return 1e12

        y = data[data <= u_upper]
        if y.size < 30:
            return 1e12

        log_dens = self.log_pdf_untruncated(y, **params)
        if np.any(~np.isfinite(log_dens)):
            return 1e12

        Z_trunc = self._get_truncation_normalizer(u_upper, **params)
        if not np.isfinite(Z_trunc) or Z_trunc <= 1e-12:
            return 1e12

        ll = float(np.sum(log_dens) - y.size * np.log(Z_trunc))
        penalty = self._regularization_penalty(params)

        return -(ll - penalty)

    def fit(self, u: NDArray[np.floating], frac: float | None = None) -> FitResult:
        frac = frac if frac is not None else self.model.default_frac
        u_c = float(np.quantile(u, frac))

        init = self.model.get_initial_params(u, u_c)
        bounds = self.model.get_bounds()

        res = optimize.minimize(
            lambda p: self.neg_log_likelihood_truncated(p, u, u_c),
            x0=init,
            bounds=bounds,
            method="L-BFGS-B",
        )

        if not res.success or not np.isfinite(res.fun):
            for attempt in range(1, 4):
                retry_init = self.model.get_retry_params(u, u_c, attempt)
                if retry_init is None:
                    break

                res2 = optimize.minimize(
                    lambda p: self.neg_log_likelihood_truncated(p, u, u_c),
                    x0=retry_init,
                    bounds=bounds,
                    method="L-BFGS-B",
                )
                if res2.success and np.isfinite(res2.fun):
                    res = res2
                    break

        return FitResult(
            params=self._params_to_dict(res.x),
            u_cut=u_c,
            converged=res.success,
            raw_result=res,
        )

    def solve_threshold_for_tail_prob(self, alpha: float, lo: float, hi: float, **params) -> float:
        def f(u):
            return self.model.tail_probability(u, **params) - float(alpha)

        try:
            return float(optimize.brentq(f, float(lo), float(hi), maxiter=200))
        except Exception as exc:
            logger.warning("solve_threshold_for_tail_prob brentq failed for alpha=%s lo=%s hi=%s: %s", alpha, lo, hi, exc)
            return float("nan")

    def calculate_threshold(
        self,
        u: NDArray[np.floating],
        alpha: float,
        frac: float | None = None,
        quality_csv_path: str | None = None,
    ) -> ThresholdResult:
        fit_result = self.fit(u, frac)
        params = fit_result.params

        lo = float(np.min(u) - 1.0)
        mu = params["mu"]
        sig = params["sigma"]
        # Bracket the brentq root at the tail-support cap: tail prob is 0 at
        # mu + TAIL_SUPPORT_Z_UPPER*sigma, so f(hi) = -alpha < 0 always brackets.
        hi = float(mu + TAIL_SUPPORT_Z_UPPER * sig)

        u_star = self.solve_threshold_for_tail_prob(alpha, lo, hi, **params)

        plot_path = None
        fit_csv_path = None
        if quality_csv_path:
            base = os.path.splitext(os.path.basename(quality_csv_path))[0]
            folder = os.path.dirname(quality_csv_path)
            plot_path = os.path.join(folder, f"{base}_{self.model_name}_fit.png")
            grid, dens = self.save_fit_plot(u, fit_result.u_cut, u_star, alpha, plot_path, **params)
            fit_csv_path = os.path.join(folder, f"{base}_{self.model_name}_fit.csv")
            out = np.column_stack((grid, dens))
            np.savetxt(
                fit_csv_path,
                out,
                delimiter=",",
                comments=""
            )

        n_total = int(u.size)
        n_hi = int(np.sum(u >= u_star)) if np.isfinite(u_star) else 0
        n_lo = n_total - n_hi

        return ThresholdResult(
            ok=fit_result.converged and np.isfinite(u_star),
            converged=fit_result.converged,
            params=params,
            u_cut_fit_range=fit_result.u_cut,
            u_star_alpha=u_star,
            alpha=alpha,
            plot_path=plot_path,
            fit_csv_path=fit_csv_path,
            n_total=n_total,
            n_ge_thresh=n_hi,
            n_lt_thresh=n_lo,
            model_name=self.model_name,
        )

    def save_fit_plot(
        self,
        u: NDArray[np.floating],
        u_c: float,
        u_star: float,
        alpha: float,
        out_path: str,
        **params,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        xlo = float(np.quantile(u, 0.01))
        xmed = float(np.median(u))
        xhi = float(xmed + 3 * (xmed - xlo))
        # A fixed point count made the exported x spacing depend on each
        # dataset's quality range.  In narrow ranges those densely packed
        # values can also look duplicated after spreadsheet/display rounding.
        # Anchor the samples to decimal tenths so the CSV has one unambiguous
        # point per x value and a consistent resolution across datasets.
        grid_start = np.ceil(xlo / FIT_CURVE_X_STEP) * FIT_CURVE_X_STEP
        grid_stop = np.floor(xhi / FIT_CURVE_X_STEP) * FIT_CURVE_X_STEP
        grid = np.arange(
            grid_start,
            grid_stop + FIT_CURVE_X_STEP / 2,
            FIT_CURVE_X_STEP,
        )
        if grid.size == 0:
            grid = np.array([grid_start])
        grid = np.round(grid, decimals=10)
        dens = self.model.pdf_untruncated(grid, **params)

        n_total = int(u.size)
        if np.isfinite(u_star):
            n_hi = int(np.sum(u >= u_star))
            n_lo = n_total - n_hi
        else:
            n_hi = n_lo = 0

        r2_full = self.compute_histogram_pdf_r2(u, xlo, xhi, bins=500, normalize_pdf_on_interval=True, **params)
        r2_tail = self.compute_histogram_pdf_r2(u, u_c, xhi, bins=500, normalize_pdf_on_interval=True, **params)

        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            ax.hist(u, bins=1000, range=(xlo, xhi), density=True, alpha=0.35,
                    label="Empirical (raw) [bins=1000]")
            ax.plot(grid, dens, lw=2, label=self.model.get_plot_label(params))
            ax.axvline(u_c, linestyle="--", linewidth=1, color="blue",
                       label=f"fit range cutoff ≈ {u_c:.2f}")

            if np.isfinite(u_star):
                ax.axvline(u_star, linestyle="-.", linewidth=1, color="red",
                           label=f"per-peak α={alpha:.3f} thr ≈ {u_star:.2f}")
                pct_hi = (n_hi / n_total) * 100.0
                pct_lo = (n_lo / n_total) * 100.0
                txt = (
                    f"n(total) = {n_total}\n"
                    f"n(u ≥ thr) = {n_hi} ({pct_hi:.1f}%)\n"
                    f"n(u < thr) = {n_lo} ({pct_lo:.1f}%)\n"
                    f"R^2 ({xlo:.4f}-{xhi:.4f}) = {r2_full:.4f}\n"
                    f"R^2 ({u_c:.4f}-{xhi:.4f}) = {r2_tail:.4f}"
                )
                ax.text(
                    0.98, 0.98, txt, transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
                )

            ax.set_xlabel("Peak quality (raw units)")
            ax.set_ylabel("Density")
            ax.set_title(f"Noise-only fit: {self.model_name}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_path, dpi=300)
        finally:
            plt.close(fig)

        return grid, dens

    def compute_histogram_pdf_r2(
        self,
        u: NDArray[np.floating],
        x_min: float,
        x_max: float,
        bins: int = 300,
        normalize_pdf_on_interval: bool = True,
        **params,
    ) -> float:
        # Pseudo-R^2 against the binned density estimate, not raw observations.
        u = np.asarray(u, dtype=float)
        u = u[np.isfinite(u)]
        if u.size < 50 or not np.isfinite([x_min, x_max]).all() or x_max <= x_min:
            return float("nan")

        hist, edges = np.histogram(u, bins=bins, range=(x_min, x_max), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        y_emp = hist
        y_fit = self.model.pdf_untruncated(centers, **params)

        if normalize_pdf_on_interval:
            z = float(np.sum(np.maximum(y_fit, 0.0) * widths))
            if np.isfinite(z) and z > 1e-12:
                y_fit = y_fit / z
            else:
                return float("nan")

        mask = np.isfinite(y_emp) & np.isfinite(y_fit)
        if np.sum(mask) < 10:
            return float("nan")

        y_emp = y_emp[mask]
        y_fit = y_fit[mask]

        ss_res = float(np.sum((y_emp - y_fit) ** 2))
        ss_tot = float(np.sum((y_emp - float(np.mean(y_emp))) ** 2))
        if ss_tot <= 0:
            return float("nan")

        return float(1.0 - ss_res / ss_tot)
