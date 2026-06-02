from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ...analysis.types import AnalysisResults


class StatisticsUtils:

    def __init__(self, results: AnalysisResults):
        self.results = results

    @classmethod
    def from_results(cls, results: AnalysisResults) -> dict[str, Any]:
        return cls(results).calculate_comprehensive_statistics()

    def calculate_comprehensive_statistics(self) -> dict[str, Any]:
        results = self.results

        if results is None:
            return {}

        # Section insertion order (size, drift, correction, msd) drives the
        # display order downstream, since Python dicts preserve insertion order.
        stats_dict: dict[str, Any] = {}

        diameters = np.array([d for d in results.diameters_by_id.values() if d > 0])

        passed_ids: set[int] = (
            set(results.diameters_by_id.keys())
            if results.diameters_by_id else set()
        )

        if diameters is not None and diameters.size > 0:
            stats_dict["diameter"] = self._calculate_distribution_stats(
                diameters, kind="diameter"
            )
            log_d = np.log(diameters)
            stats_dict["diameter"]["log_std_mean"] = float(np.std(log_d))
            # MAD * 1.4826 -> sigma assuming log-normal; robust to outliers in
            # the tails that distort plain np.std on heavy-tailed diameters.
            stats_dict["diameter"]["log_std_median"] = float(
                np.median(np.abs(log_d - np.median(log_d))) * 1.4826
            )

        stats_dict["drift"] = self._build_drift_vector_summary(
            results.drift_by_id, passed_ids,
            name="Drift (Pre-Correction)",
        )

        stats_dict["correction"] = self._build_correction_summary(
            results, passed_ids,
        )

        stats_dict["msd"] = self._build_msd_summary(results, passed_ids)

        return stats_dict

    @staticmethod
    def _build_drift_vector_summary(
        drift_by_id: dict[int, tuple[float, float]] | None,
        passed_ids: set[int],
        name: str = "Drift",
    ) -> dict[str, Any]:
        if not drift_by_id:
            vectors = np.empty((0, 2))
        else:
            vectors = np.array(
                [
                    drift_by_id[pid] for pid in drift_by_id
                    if pid in passed_ids
                ],
                dtype=float,
            )
            if vectors.size == 0:
                vectors = np.empty((0, 2))

        if vectors.shape[0] == 0:
            mean_xy = (float("nan"), float("nan"))
            median_xy = (float("nan"), float("nan"))
            std_xy = (float("nan"), float("nan"))
        else:
            mean_xy = (float(np.mean(vectors[:, 0])), float(np.mean(vectors[:, 1])))
            median_xy = (float(np.median(vectors[:, 0])), float(np.median(vectors[:, 1])))
            std_xy = (float(np.std(vectors[:, 0])), float(np.std(vectors[:, 1])))

        return {
            "kind": "drift_vector",
            "name": name,
            "n_tracks": int(vectors.shape[0]),
            "mean_xy": mean_xy,
            "median_xy": median_xy,
            "std_xy": std_xy,
            "unit": "um/s",
        }

    @staticmethod
    def _build_correction_summary(
        results: AnalysisResults,
        passed_ids: set[int],
    ) -> dict[str, Any]:
        post_drift_stats = StatisticsUtils._build_drift_vector_summary(
            results.post_correction_drift_by_id,
            passed_ids,
            name="Post-Correction Drift",
        )
        return {
            "name": "Drift Correction",
            "kind": "correction",
            "mode": getattr(results, "correction_mode", "no"),
            "label": getattr(results, "correction_label", "No Correction"),
            "params": dict(getattr(results, "correction_params", {}) or {}),
            "post_drift": post_drift_stats,
            "unit": "um/s",
        }

    @staticmethod
    def _build_msd_summary(
        results: AnalysisResults,
        passed_ids: set[int],
    ) -> dict[str, Any]:
        total = (
            len(results.msd_r2_by_id)
            if results.msd_r2_by_id is not None else 0
        )
        passed = len(passed_ids)
        failed = max(total - passed, 0)
        failed_d = len(getattr(results, "last_nonpositive_d_tracks", None) or [])
        failed_r2 = max(failed - failed_d, 0)
        return {
            "kind": "msd",
            "name": "MSD Fit Settings",
            "lag_frame": int(getattr(results, "lag_frame", 0)),
            "msd_max_lag": int(getattr(results, "msd_max_lag", 0)),
            "extra_points": int(getattr(results, "msd_extra_points", 0)),
            "min_points": int(getattr(results, "msd_min_points", 0)),
            "r2_threshold": float(getattr(results, "r2_threshold", 0.0)),
            "passed": int(passed),
            "failed": int(failed),
            "failed_d": int(failed_d),
            "failed_r2": int(failed_r2),
            "total": int(total),
        }

    @staticmethod
    def _calculate_distribution_stats(data: ArrayLike, kind: str = "diameter", bin_size=None) -> dict[str, Any]:

        arr = np.asarray(data, dtype=float)

        if arr.size == 0:
            return {
                "data": arr,
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "mode": None,
                "peaks": [] if kind == "diameter" else None,
                "unit": "nm" if kind == "diameter" else "",
                "name": "Particle Size" if kind == "diameter" else kind,
            }

        if kind == "diameter":
            unit = "nm"
            name = "Particle Size"
            default_bin_size = 10
        elif kind == "diffusion":
            unit = "um^2/s"
            name = "Brownian Diffusion"
            default_bin_size = None
        elif kind == "drift_magnitude":
            unit = "um/s"
            name = "Drift Velocity"
            default_bin_size = None
        elif kind == "msd_r2":
            unit = ""
            name = "MSD Fit Quality (R^2)"
            default_bin_size = 0.02
        else:
            unit = ""
            name = kind
            default_bin_size = None

        actual_bin_size = bin_size if bin_size is not None else default_bin_size

        mode = StatisticsUtils._histogram_mode(arr, bin_size=actual_bin_size)
        peaks = (
            StatisticsUtils._detect_peaks_from_samples(arr, bin_size=actual_bin_size)
            if kind == "diameter"
            else None
        )

        return {
            "data": arr,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "mode": mode,
            "peaks": peaks,
            "unit": unit,
            "name": name,
        }

    @staticmethod
    def _histogram_mode(arr: NDArray[np.floating], bin_size: float = None):
        if len(arr) == 0 or np.allclose(np.ptp(arr), 0):
            return None
        bins = StatisticsUtils._auto_bin_count(arr, bin_size=bin_size, min_bins=20)
        hist, bin_edges = np.histogram(arr, bins=bins)
        if hist.size == 0:
            return None
        max_bin_idx = int(np.argmax(hist))
        mode_value = 0.5 * (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1])
        return float(mode_value)

    @staticmethod
    def _auto_bin_count(arr: NDArray[np.floating], bin_size: float = None, min_bins: int = 20, max_bins: int = 1000):
        n = len(arr)
        if n == 0:
            return min_bins
        rng = float(np.ptp(arr))
        if rng <= 0:
            return min_bins
        if bin_size is None:
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = float(q75 - q25)
            if iqr > 0:
                bw = 2.0 * iqr / np.cbrt(n)
            else:
                std = float(np.std(arr))
                bw = 3.49 * std / np.cbrt(n) if std > 0 else rng / max(np.cbrt(n), 1.0)
            bw = max(bw, 1e-9)
            bins = int(np.ceil(rng / bw))
        else:
            bins = int(np.ceil(rng / max(float(bin_size), 1e-9)))
        return int(np.clip(bins, min_bins, max_bins))

    @staticmethod
    def _detect_peaks_from_samples(
        arr: NDArray[np.floating],
        bin_size: float = None,
        prominence_rel: float = 0.10,
        smooth_nm: float = 5.0,
        distance_nm: float = 5.0,
        width_bins: int = 1,
        min_bins=20,
    ):
        if len(arr) == 0 or np.allclose(np.ptp(arr), 0):
            return []

        bins = StatisticsUtils._auto_bin_count(arr, bin_size=bin_size, min_bins=min_bins)
        hist, bin_edges = np.histogram(arr, bins=bins)
        if hist.size < 3 or np.max(hist) == 0:
            return []

        bin_width = (bin_edges[-1] - bin_edges[0]) / len(hist)
        sigma_bins = max(1.0, smooth_nm / bin_width)
        dist_bins = max(2, int(distance_nm / bin_width))

        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=sigma_bins)

        prom_abs = float(prominence_rel) * float(np.max(hist_smooth))
        if prom_abs <= 0:
            return []

        peaks, props = find_peaks(
            hist_smooth,
            prominence=prom_abs,
            distance=dist_bins,
            width=max(int(width_bins), 1),
            plateau_size=(1, None),
        )

        if peaks.size == 0:
            return []

        out = []
        prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
        for i, idx in enumerate(peaks):
            mode_value = 0.5 * (bin_edges[idx] + bin_edges[idx + 1])
            count_val = int(hist[min(idx, hist.size - 1)])
            out.append(
                {
                    "mode": float(mode_value),
                    "count": count_val,
                    "prominence": float(prominences[i]),
                }
            )

        out.sort(key=lambda x: x["prominence"], reverse=True)
        return out

    @staticmethod
    def detect_peaks_from_histogram(
        bin_centers: ArrayLike,
        bin_edges: ArrayLike,
        counts: ArrayLike,
        prominence_rel: float = 0.10,
        smooth_nm: float = 5.0,
        distance_nm: float = 5.0,
        width_bins: int = 1,
    ):
        # Operate on the plotted histogram directly so FTLA/Iterative weight
        # distributions produce peaks matching the visible bars; rebuilding
        # the histogram from raw samples would not.
        counts = np.asarray(counts, dtype=float)
        bin_centers = np.asarray(bin_centers, dtype=float)
        bin_edges = np.asarray(bin_edges, dtype=float)

        if counts.size < 3 or not np.any(counts > 0):
            return []
        if bin_centers.size != counts.size or bin_edges.size != counts.size + 1:
            return []

        bin_width = float((bin_edges[-1] - bin_edges[0]) / len(counts))
        if bin_width <= 0:
            return []

        sigma_bins = max(1.0, smooth_nm / bin_width)
        dist_bins = max(2, int(distance_nm / bin_width))

        hist_smooth = gaussian_filter1d(counts, sigma=sigma_bins)
        prom_abs = float(prominence_rel) * float(np.max(hist_smooth))
        if prom_abs <= 0:
            return []

        peaks, props = find_peaks(
            hist_smooth,
            prominence=prom_abs,
            distance=dist_bins,
            width=max(int(width_bins), 1),
            plateau_size=(1, None),
        )
        if peaks.size == 0:
            return []

        out = []
        prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
        for i, idx in enumerate(peaks):
            mode_value = float(bin_centers[idx])
            count_val = float(counts[idx])
            out.append(
                {
                    "mode": mode_value,
                    "count": count_val,
                    "prominence": float(prominences[i]),
                }
            )

        out.sort(key=lambda x: x["prominence"], reverse=True)
        return out
