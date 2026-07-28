from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.interpolate import CubicSpline

CHENG_SCHWARTZMAN_DEFAULT_KAPPA = 0.999  # near 1.0 freezes kappa; lower values relax the constraint

# Upper bound (in standardized z = (u - mu)/sigma) of the tail-integration
# support when solving for the auto-threshold: beyond z = 6 values are
# numerical/extreme outliers, not background fluctuations. Applied identically
# by every model so alpha means the same thing whichever background was fit.
TAIL_SUPPORT_Z_UPPER = 6.0


def _capped_upper_tail(cdf_standardized, z0: float,
                       z_cap: float = TAIL_SUPPORT_Z_UPPER) -> float:
    """Upper-tail probability of a standardized background density, with the
    support truncated at ``z_cap``.

        tail(z0) = P(z0 < Z <= z_cap) / P(Z <= z_cap)
                 = (F(z_cap) - F(z0)) / F(z_cap)

    ``cdf_standardized`` is the model's standardized CDF F(z) (any positive
    scaling cancels in the ratio). Values at or above ``z_cap`` have zero tail.
    """
    z0 = float(z0)
    if z0 >= z_cap:
        return 0.0
    cdf_cap = float(cdf_standardized(z_cap))
    if not np.isfinite(cdf_cap) or cdf_cap <= 1e-12:
        return float("nan")
    cdf_0 = float(cdf_standardized(z0))
    return float(np.clip((cdf_cap - cdf_0) / cdf_cap, 0.0, 1.0))


class ChengSchwartzmanModel:
    """Cheng & Schwartzman (2015), eq. (2.10). Params: mu, sigma, kappa."""

    model_name = "Cheng-Schwartzman"
    default_frac = 0.80

    def __init__(self):
        self._cdf_cache: dict[float, CubicSpline] = {}

    @property
    def param_names(self) -> list[str]:
        return ["mu", "sigma", "kappa"]

    @staticmethod
    def _std_normal_pdf(z: NDArray[np.floating]) -> NDArray[np.floating]:
        return stats.norm.pdf(z)

    @staticmethod
    def _std_normal_cdf(z: NDArray[np.floating]) -> NDArray[np.floating]:
        return stats.norm.cdf(z)

    def _cheng_schwartzman_standardized_density(self, x: NDArray[np.floating], kappa: float) -> float:
        x = np.asarray(x, dtype=float)
        k = float(np.clip(kappa, 1e-6, 0.999))

        d1 = max(1e-6, 2.0 - k*k)
        d2 = max(1e-6, 3.0 - k*k)

        A = (np.sqrt(3.0) * (k**2) * (x*x - 1.0)
             * self._std_normal_pdf(x)
             * self._std_normal_cdf((k * x) / np.sqrt(d1)))

        exp_arg_B = np.clip(-x*x / d1, -100, 100)
        B = (k * x * np.sqrt(3.0 * d1) / (2.0 * np.pi)) * np.exp(exp_arg_B)

        exp_arg_C = np.clip(-3.0 * x*x / (2.0 * d2), -100, 100)
        C = ((np.sqrt(6.0) / np.sqrt(np.pi * d2)) * np.exp(exp_arg_C)
             * self._std_normal_cdf((k * x) / np.sqrt(d2 * d1)))

        h = A + B + C
        return np.maximum(h, 0.0)

    def pdf_untruncated(self, u: NDArray[np.floating], mu: float, sigma: float, kappa: float, **_) -> NDArray[np.floating]:
        if sigma <= 0 or not np.isfinite(sigma):
            return np.full_like(np.asarray(u, dtype=float), np.nan)
        z = (np.asarray(u, dtype=float) - float(mu)) / float(sigma)
        return (1.0 / abs(float(sigma))) * self._cheng_schwartzman_standardized_density(z, kappa)

    def tail_probability(self, u0: float, mu: float, sigma: float, kappa: float, **_) -> float:
        if sigma <= 0 or not np.isfinite(sigma):
            return float("nan")
        z0 = (float(u0) - float(mu)) / float(sigma)
        cdf = self._get_cached_cdf(float(kappa))
        return _capped_upper_tail(lambda z: float(cdf(z)), z0)

    def get_initial_params(self, data: NDArray[np.floating], u_cut: float) -> NDArray[np.floating]:
        subset: NDArray[np.floating] = data[data <= u_cut]
        mu0 = float(np.median(subset))
        mad0 = float(stats.median_abs_deviation(subset, scale=1.0))
        sig0 = max(1e-3, 1.4826 * mad0)
        return np.array([mu0, sig0, CHENG_SCHWARTZMAN_DEFAULT_KAPPA], dtype=float)

    def get_bounds(self) -> list[tuple[float | None, float | None]]:
        return [
            (None, None),
            (1e-6, None),
            (CHENG_SCHWARTZMAN_DEFAULT_KAPPA, CHENG_SCHWARTZMAN_DEFAULT_KAPPA),
        ]

    def get_retry_params(self, data: NDArray[np.floating], u_cut: float, attempt: int) -> NDArray[np.floating] | None:
        if attempt > 1:
            return None
        subset = data[data <= u_cut]
        mu0 = float(np.median(subset))
        mad0 = float(stats.median_abs_deviation(subset, scale=1.0))
        sig0 = max(1e-3, 1.4826 * mad0 * 0.7)
        return np.array([mu0, sig0, CHENG_SCHWARTZMAN_DEFAULT_KAPPA], dtype=float)

    def get_plot_label(self, params: dict[str, float]) -> str:
        kappa = params.get("kappa", CHENG_SCHWARTZMAN_DEFAULT_KAPPA)
        return f"C&S (2.10) fit, kappa={kappa:.3f}"

    def validate_params(self, params: dict[str, float]) -> bool:
        mu, sigma, kappa = params["mu"], params["sigma"], params["kappa"]
        if not all(np.isfinite([mu, sigma, kappa])):
            return False
        if sigma <= 0:
            return False
        if not (1e-6 <= kappa <= 1.0):
            return False
        return True

    def truncation_normalizer(
        self,
        u_upper: float,
        mu: float,
        sigma: float,
        kappa: float,
        **_,
    ) -> float:

        if sigma <= 0 or not np.isfinite(sigma):
            return 1e-12

        z_upper = (float(u_upper) - float(mu)) / float(sigma)
        cdf = self._get_cached_cdf(float(kappa))
        return float(np.clip(cdf(z_upper), 1e-12, 1.0))

    def _get_cached_cdf(self, kappa: float):
        key = round(kappa, 6)
        cached = self._cdf_cache.get(key)
        if cached is not None:
            return cached

        from scipy.integrate import cumulative_trapezoid

        z_grid = np.linspace(-12.0, 20.0, 4001)
        pdf_vals = self._cheng_schwartzman_standardized_density(z_grid, kappa)
        cdf_vals = cumulative_trapezoid(pdf_vals, z_grid, initial=0.0)

        spline = CubicSpline(z_grid, cdf_vals, extrapolate=True)
        self._cdf_cache[key] = spline
        return spline


class Poly2Model:
    """φ(z) * (a0 + a1*z + a2*z²). Params: mu, sigma, a0, a1, a2."""

    model_name = "Poly2"
    default_frac = 0.85

    @property
    def param_names(self) -> list[str]:
        return ["mu", "sigma", "a0", "a1", "a2"]

    @staticmethod
    def _std_normal_pdf(z: NDArray[np.floating]) -> NDArray[np.floating]:
        return stats.norm.pdf(z)

    @staticmethod
    def _std_normal_cdf(z: NDArray[np.floating]) -> NDArray[np.floating]:
        return stats.norm.cdf(z)

    @staticmethod
    def _full_range_normalization_constant(a0: float, a1: float, a2: float) -> float:
        return max(float(a0 + a2), 1e-12)

    def pdf_untruncated(self, u: NDArray[np.floating], mu: float, sigma: float,
                        a0: float, a1: float, a2: float, **_) -> NDArray[np.floating]:
        z = (np.asarray(u, dtype=float) - mu) / sigma
        P = a0 + a1 * z + a2 * z * z
        Z = self._full_range_normalization_constant(a0, a1, a2)
        return (1.0 / abs(sigma)) * self._std_normal_pdf(z) * P / Z

    def tail_probability(self, u0: float, mu: float, sigma: float,
                         a0: float, a1: float, a2: float, **_) -> float:
        if sigma <= 0 or not np.isfinite(sigma):
            return float("nan")
        Z = self._full_range_normalization_constant(a0, a1, a2)

        def cdf_std(z: float) -> float:
            # Standardized CDF F(z) = int_{-inf}^{z} phi(t)(a0+a1 t+a2 t^2)/Z dt.
            phi = float(self._std_normal_pdf(np.array([z]))[0])
            Phi = float(self._std_normal_cdf(np.array([z]))[0])
            num = a0 * Phi + a1 * (-phi) + a2 * (Phi - z * phi)
            return num / Z

        z0 = (float(u0) - mu) / sigma
        return _capped_upper_tail(cdf_std, z0)

    def truncation_normalizer(self, u_upper: float, mu: float, sigma: float,
                              a0: float, a1: float, a2: float, **_) -> float:
        z = (u_upper - mu) / sigma
        Z = self._full_range_normalization_constant(a0, a1, a2)
        phi_z = float(self._std_normal_pdf(np.array([z]))[0])
        cdf_z = float(self._std_normal_cdf(np.array([z]))[0])

        num = a0 * cdf_z + a1 * (-phi_z) + a2 * (cdf_z - z * phi_z)
        return max(float(num / Z), 1e-12)

    def get_initial_params(self, data: NDArray[np.floating], u_cut: float) -> NDArray[np.floating]:
        subset = data[data <= u_cut]
        mu0 = float(np.median(subset))
        mad0 = float(stats.median_abs_deviation(subset, scale=1.0))
        sig0 = max(mad0 * 1.4826, 1e-3)
        return np.array([mu0, sig0, 0.8, 0.0, 0.3], dtype=float)

    def get_bounds(self) -> list[tuple[float | None, float | None]]:
        return [
            (None, None),
            (1e-6, None),
            (1e-6, None),
            (None, None),
            (1e-6, None),
        ]

    def get_retry_params(self, data: NDArray[np.floating], u_cut: float, attempt: int) -> NDArray[np.floating] | None:
        if attempt > 1:
            return None
        subset = data[data <= u_cut]
        mu0 = float(np.median(subset))
        mad0 = float(stats.median_abs_deviation(subset, scale=1.0))
        sig0 = max(mad0 * 1.4826 * 0.7, 1e-3)
        return np.array([mu0, sig0, 0.6, 0.0, 0.6], dtype=float)

    def get_plot_label(self, params: dict[str, float]) -> str:
        a0, a1, a2 = params["a0"], params["a1"], params["a2"]
        return f"Poly2 fit (a0={a0:.2f}, a1={a1:.2f}, a2={a2:.2f})"

    def validate_params(self, params: dict[str, float]) -> bool:
        if params["sigma"] <= 0:
            return False
        if params["a0"] <= 1e-6 or params["a2"] <= 1e-6:
            return False
        return True


class GaussianModel:

    model_name = "Gaussian"
    default_frac = 0.90

    @property
    def param_names(self) -> list[str]:
        return ["mu", "sigma"]

    def pdf_untruncated(self, u: NDArray[np.floating], mu: float, sigma: float, **_) -> NDArray[np.floating]:
        return stats.norm.pdf(u, loc=mu, scale=sigma)

    def tail_probability(self, u0: float, mu: float, sigma: float, **_) -> float:
        if sigma <= 0 or not np.isfinite(sigma):
            return float("nan")
        z0 = (float(u0) - float(mu)) / float(sigma)
        return _capped_upper_tail(lambda z: float(stats.norm.cdf(z)), z0)

    def truncation_normalizer(self, u_upper: float, mu: float, sigma: float, **_) -> float:
        return float(stats.norm.cdf(u_upper, loc=mu, scale=sigma))

    def get_initial_params(self, data: NDArray[np.floating], u_cut: float) -> NDArray[np.floating]:
        subset = data[data <= u_cut]
        mu0 = float(np.mean(subset))
        sig0 = float(np.std(subset))
        return np.array([mu0, max(sig0, 1e-3)], dtype=float)

    def get_bounds(self) -> list[tuple[float | None, float | None]]:
        return [(None, None), (1e-6, None)]

    def get_retry_params(self, data: NDArray[np.floating], u_cut: float, attempt: int) -> NDArray[np.floating] | None:
        return None

    def get_plot_label(self, params: dict[str, float]) -> str:
        return f"Gaussian (μ={params['mu']:.2f}, σ={params['sigma']:.2f})"

    def validate_params(self, params: dict[str, float]) -> bool:
        return params["sigma"] > 0


class NoFittingFracModel:
    """Marker model that uses the empirical FRAC quantile as the threshold."""

    model_name = "No Fitting (FRAC)"
    default_frac = 0.85
    uses_empirical_frac = True

    @property
    def param_names(self) -> list[str]:
        return []


MODEL_CLASSES = {
    "No Fitting (FRAC)": NoFittingFracModel,
    "Cheng-Schwartzman": ChengSchwartzmanModel,
    "Poly2": Poly2Model,
    "Gaussian": GaussianModel,
}


def get_model_class(name: str):
    if name not in MODEL_CLASSES:
        available = ", ".join(MODEL_CLASSES.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODEL_CLASSES[name]
