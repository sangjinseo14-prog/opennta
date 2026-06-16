from __future__ import annotations

import numpy as np
from matplotlib.gridspec import GridSpec
from PyQt5.QtCore import QTimer

from . import _plot_style as _ps


class _PlotBuilderMixin:

    _FIG_LEFT = 0.04
    _FIG_RIGHT = 0.88
    _FIG_TOP = 0.92
    _FIG_BOTTOM = 0.08
    _FIG_HSPACE = 0.35
    _FIG_WSPACE = 0.55

    _CBAR_W_PX = 20
    _CBAR_H_PX = 200
    _CBAR_GAP_PX = 20

    _TRACKS_INDEX = _ps.TRACKS_INDEX
    _PLOT_SPECS = _ps.PLOT_SPECS

    def _init_plot_state(self) -> None:
        self._cbars = [None] * len(self._PLOT_SPECS)
        self._cbar_repositioning = False

        self._show_vectors = False
        self._vector_artists: list = []

    def _init_plot_slots(self) -> None:
        # Three-column layout:
        #   col 0 (2 rows): tracks
        #   col 1 row 0  : mean drift
        #   col 1 row 1  : SE
        #   col 2 (2 rows): SE-weighted smoothed
        gs = GridSpec(
            2,
            3,
            figure=self.fig,
            left=self._FIG_LEFT,
            right=self._FIG_RIGHT,
            top=self._FIG_TOP,
            bottom=self._FIG_BOTTOM,
            hspace=self._FIG_HSPACE,
            wspace=self._FIG_WSPACE,
        )
        self._slot_axes = [
            self.fig.add_subplot(gs[:, 0]),
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[1, 1]),
            self.fig.add_subplot(gs[:, 2]),
        ]
        for ax in self._slot_axes:
            ax.set_aspect("equal", adjustable="box", anchor="C")

    def _redraw(self) -> None:
        # _draw_slot calls ax.clear(), which drops any existing quiver artists.
        self._vector_artists = []

        for idx, (data, cmap, title, unit) in enumerate(self._iter_field_panel_specs()):
            self._draw_slot(idx, data, cmap, title, unit)

        self._redraw_vector_overlay()
        self._sync_cbar_positions()
        self.canvas.draw_idle()

    def _redraw_tracks_only(self) -> None:
        idx = self._TRACKS_INDEX
        title, _cmap, unit = self._PLOT_SPECS[idx]
        self._draw_slot(idx, None, None, title, unit)
        self.canvas.draw_idle()

    def _iter_field_panel_specs(self):
        specs = self._PLOT_SPECS
        # Tracks slot is driven by the raw df; passing None lets _draw_slot
        # render the polylines directly instead of an image array.
        track_title, track_cmap, track_unit = specs[self._TRACKS_INDEX]

        if self.stats is None:
            payloads = [(None, cmap, title, unit) for title, cmap, unit in specs]
            payloads[self._TRACKS_INDEX] = (None, track_cmap, track_title, track_unit)
            return payloads

        # Stored field is in um/frame; display in um/s via fps scaling.
        fps = float(self.config.fps)
        vbar = np.hypot(self.stats.mean_dx, self.stats.mean_dy) * fps
        se_vec = self.stats.se_vec * fps
        vbar_smoothed = None

        if self._use_smoothed and self.smoothed_u is not None and self.smoothed_v is not None:
            vbar_smoothed = np.hypot(self.smoothed_u, self.smoothed_v) * fps

        return [
            (None, track_cmap, track_title, track_unit),
            (vbar, specs[1][1], specs[1][0], specs[1][2]),
            (se_vec, specs[2][1], specs[2][0], specs[2][2]),
            (vbar_smoothed, specs[3][1], specs[3][0], specs[3][2]),
        ]

    def _draw_slot(self, idx: int, data, cmap, title: str, unit: str = "") -> None:
        ax = self._slot_axes[idx]
        self._remove_cbar(idx)

        ax.clear()
        ax.set_aspect("equal", adjustable="box", anchor="C")
        _ps.style_panel_axes(ax, title)

        if idx == self._TRACKS_INDEX:
            self._draw_tracks(ax, idx)
            return

        if data is None:
            self._draw_placeholder(ax)
            return

        im = ax.imshow(data, cmap=cmap, origin="upper", interpolation="nearest")
        ny, nx = data.shape[:2]
        ax.set_xticks(np.arange(0, nx, max(1, nx // 10)))
        ax.set_yticks(np.arange(0, ny, max(1, ny // 10)))
        cax = self.fig.add_axes([0.0, 0.0, 0.01, 0.01])
        cb = self.fig.colorbar(im, cax=cax)
        _ps.style_colorbar(cb, unit)
        self._cbars[idx] = cb

    def _draw_tracks(self, ax, idx: int) -> None:
        n = max(int(self.sb_nwin.value()), 2)
        x_range = (float(self.sb_xmin.value()), float(self.sb_xmax.value()))
        y_range = (float(self.sb_ymin.value()), float(self.sb_ymax.value()))

        sm = _ps.draw_tracks_on_axes(ax, self.df, n, x_range, y_range)
        if sm is None:
            return

        cax = self.fig.add_axes([0.0, 0.0, 0.01, 0.01])
        cb = self.fig.colorbar(sm, cax=cax)
        _ps.style_colorbar(cb, _ps.TRACK_TIME_LABEL)
        self._cbars[idx] = cb

    def _draw_placeholder(self, ax) -> None:
        n_windows = max(self.sb_nwin.value(), 2)
        ax.set_xlim(-0.5, n_windows - 0.5)
        ax.set_ylim(n_windows - 0.5, -0.5)
        ax.text(
            0.5,
            0.5,
            "(not computed)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=_ps.FG_MUTED,
            fontsize=_ps.FS_LABEL,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def _remove_cbar(self, idx: int) -> None:
        cb = self._cbars[idx]
        if cb is None:
            return
        try:
            cb.remove()
        except Exception:
            pass
        self._cbars[idx] = None

    def _sync_cbar_positions(self, redraw: bool = True) -> None:
        if redraw:
            self.fig.canvas.draw()

        canvas_w = max(1, self.canvas.width())
        canvas_h = max(1, self.canvas.height())

        cbar_w = self._CBAR_W_PX / canvas_w
        cbar_h = self._CBAR_H_PX / canvas_h
        cbar_gap = self._CBAR_GAP_PX / canvas_w

        for ax, cb in zip(self._slot_axes, self._cbars, strict=False):
            if cb is None:
                continue

            bb = ax.get_position()
            y0 = bb.y0 + (bb.height - cbar_h) * 0.5
            cb.ax.set_position([bb.x1 + cbar_gap, y0, cbar_w, cbar_h])

    def _on_canvas_resize(self, event) -> None:
        if self._cbar_repositioning:
            return

        self._cbar_repositioning = True
        QTimer.singleShot(0, self._finish_resize)

    def _finish_resize(self) -> None:
        try:
            self._sync_cbar_positions(redraw=False)
            self.canvas.draw_idle()
        finally:
            self._cbar_repositioning = False

    def _toggle_vectors(self, checked: bool) -> None:
        self._show_vectors = bool(checked)
        self._redraw_vector_overlay()
        self.canvas.draw_idle()

    def _clear_vector_overlay(self) -> None:
        for artist in self._vector_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._vector_artists = []

    def _redraw_vector_overlay(self) -> None:
        self._clear_vector_overlay()
        if not self._show_vectors or self.stats is None:
            return

        q_mean = _ps.draw_quiver_overlay(
            self._slot_axes[_ps.MEAN_INDEX],
            self.stats.mean_dx, self.stats.mean_dy,
        )
        if q_mean is not None:
            self._vector_artists.append(q_mean)

        if (
            self._use_smoothed
            and self.smoothed_u is not None
            and self.smoothed_v is not None
        ):
            q_sm = _ps.draw_quiver_overlay(
                self._slot_axes[_ps.SMOOTHED_INDEX],
                self.smoothed_u, self.smoothed_v,
            )
            if q_sm is not None:
                self._vector_artists.append(q_sm)
