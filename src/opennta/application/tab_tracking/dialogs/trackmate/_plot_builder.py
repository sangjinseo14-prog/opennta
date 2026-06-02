from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import _plot_style as _ps
from .preview_worker import TrackMatePreviewResult


@dataclass
class _HistogramRenderCache:
    centers: NDArray[np.floating]
    bin_width: float
    counts: NDArray[np.floating]
    smoothed: NDArray[np.floating] | None
    fit_cutoff: float | None


class _PlotBuilderMixin:

    _SMOOTH_WINDOW = 11  # ±5 neighbors + center

    def _init_plot_state(self) -> None:
        self._histogram_caches: dict[str, _HistogramRenderCache] = {}

    def _reset_preview_axes(self) -> None:
        self.ax_image.clear()
        _ps.style_image_axis(self.ax_image)
        self.ax_hist.clear()
        _ps.style_histogram_axis(self.ax_hist)
        self.canvas.draw_idle()

    def _on_display_toggle(self, _checked: bool) -> None:
        idx = self.combo_sample.currentIndex()
        if idx < 0 or idx >= len(self._folder_items):
            return
        key = self._format_sample_key(self._folder_items[idx])
        result = self._results.get(key)
        if result is not None:
            self._render_preview(result)

    def _build_histogram_cache(
        self, result: TrackMatePreviewResult
    ) -> _HistogramRenderCache | None:
        qualities = result.qualities
        if not qualities.size:
            return None
        q01 = float(np.quantile(qualities, 0.01))
        qmed = float(np.median(qualities))
        qhi = float(qmed + 3 * (qmed - q01))
        counts, edges = np.histogram(
            qualities, bins=_ps.HIST_BINS, range=(q01, qhi), density=True,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_width = float(edges[1] - edges[0])
        smoothed: NDArray[np.floating] | None = None
        if counts.size >= self._SMOOTH_WINDOW:
            kernel = np.ones(self._SMOOTH_WINDOW) / self._SMOOTH_WINDOW
            smoothed = np.convolve(counts, kernel, mode="same")
        fit_cutoff: float | None = None
        if np.isfinite(result.fit_fraction) and 0.0 < result.fit_fraction < 1.0:
            fit_cutoff = float(np.quantile(qualities, result.fit_fraction))
        return _HistogramRenderCache(
            centers=centers,
            bin_width=bin_width,
            counts=counts,
            smoothed=smoothed,
            fit_cutoff=fit_cutoff,
        )

    def _render_preview(self, result: TrackMatePreviewResult) -> None:
        cache = self._histogram_caches.get(result.sample_key)
        if cache is None:
            cache = self._build_histogram_cache(result)
            if cache is not None:
                self._histogram_caches[result.sample_key] = cache
        self._render_image_panel(result)
        self._render_histogram_panel(result, cache)
        self.canvas.draw_idle()

    def _render_image_panel(self, result: TrackMatePreviewResult) -> None:
        self.ax_image.clear()
        image = result.image
        vmin = float(np.percentile(image, _ps.IMAGE_VMIN_PCT))
        vmax = float(np.percentile(image, _ps.IMAGE_VMAX_PCT))
        if vmax <= vmin:
            vmax = vmin + 1.0
        self.ax_image.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        if (
            self.check_show_overlay.isChecked()
            and result.spots_x.size
            and np.isfinite(result.threshold)
        ):
            above = result.qualities >= result.threshold
            self.ax_image.scatter(
                result.spots_x[above], result.spots_y[above],
                s=_ps.SPOT_MARKER_SIZE,
                facecolors="none", edgecolors=_ps.SPOT_OVERLAY,
                linewidths=_ps.SPOT_MARKER_LW,
                label=f"above threshold ({int(above.sum())})",
            )
            self.ax_image.legend(loc="upper right", fontsize=_ps.FS_LEGEND)
        _ps.style_image_axis(
            self.ax_image, f"{result.sample_key} - frame {result.frame_index}",
        )

    def _render_histogram_panel(
        self,
        result: TrackMatePreviewResult,
        cache: _HistogramRenderCache | None,
    ) -> None:
        self.ax_hist.clear()
        if cache is not None:
            self.ax_hist.bar(
                cache.centers, cache.counts, width=cache.bin_width,
                alpha=_ps.HIST_BAR_ALPHA, color=_ps.HIST_BAR, linewidth=0,
            )
            if self.check_show_smoothed.isChecked() and cache.smoothed is not None:
                self.ax_hist.plot(
                    cache.centers, cache.smoothed,
                    lw=_ps.HIST_SMOOTHED_LW, color=_ps.HIST_SMOOTHED,
                    label="smoothed (±5)",
                )
            if result.fit_grid is not None and result.fit_density is not None:
                self.ax_hist.plot(
                    result.fit_grid, result.fit_density,
                    lw=_ps.HIST_FIT_LW, color=_ps.HIST_FIT,
                    label="Noise fit",
                )
            if cache.fit_cutoff is not None:
                self.ax_hist.axvline(
                    cache.fit_cutoff,
                    color=_ps.HIST_CUTOFF, lw=_ps.HIST_CUTOFF_LW, linestyle=":",
                    label=f"FRAC cutoff = {cache.fit_cutoff:.3f}",
                )
            if np.isfinite(result.threshold):
                self.ax_hist.axvline(
                    result.threshold,
                    color=_ps.HIST_THRESHOLD, lw=_ps.HIST_THRESHOLD_LW, linestyle="--",
                    label=f"threshold = {result.threshold:.3f}",
                )
            self.ax_hist.legend(loc="upper right", fontsize=_ps.FS_LEGEND)
        _ps.style_histogram_axis(self.ax_hist)
