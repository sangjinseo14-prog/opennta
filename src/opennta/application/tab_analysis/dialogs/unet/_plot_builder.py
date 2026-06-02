from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PyQt5.QtCore import QTimer

from . import _plot_style as _ps


class _PlotBuilderMixin:

    def _init_plot_slots(self, fig: Figure, axes_list: list, specs) -> None:
        fig.clear()
        axes_list.clear()
        gs = GridSpec(len(specs), 1, figure=fig, **_ps.FIG_LAYOUT)
        for i in range(len(specs)):
            ax = fig.add_subplot(gs[i])
            ax.set_aspect("equal", adjustable="box", anchor="C")
            axes_list.append(ax)

    def _draw_placeholders(self, axes_list, specs, canvas) -> None:
        for ax, (title, _cmap, _unit) in zip(axes_list, specs, strict=False):
            ax.clear()
            ax.set_aspect("equal", adjustable="box", anchor="C")
            ax.set_title(title, fontsize=10)
            self._apply_axis_ticks(ax, (128, 128))
            ax.text(
                0.5, 0.5, "(not computed)",
                transform=ax.transAxes, ha="center", va="center",
                color=_ps.FG_MUTED, fontsize=10,
            )
        canvas.draw_idle()

    def _draw_input_panels(self, sample) -> None:
        from opennta.analysis.unet_field.unet_bundle.visualization import symmetric_percentile_limit

        X = sample["X"]
        lim_u = symmetric_percentile_limit(X[..., 0], pct=99.0)
        lim_v = symmetric_percentile_limit(X[..., 1], pct=99.0)
        # Arrays are in physics y-up layout (row 0 = smallest ys_ref = bottom
        # of the camera image). Flip vertically + sign-flip V to match the
        # top-left-origin screen convention used by the panel.
        payloads = (
            (X[..., 0][::-1, :],  -lim_u, lim_u),
            (-X[..., 1][::-1, :], -lim_v, lim_v),
            (X[..., 2][::-1, :],   0.0,    1.0),
        )
        self._draw_panels(
            self.fig_inputs, self._input_axes, self._input_cbars,
            _ps.INPUT_SPECS, payloads, self.canvas_inputs,
        )

    def _draw_output_panels(self, pred_u, pred_v) -> None:
        from opennta.analysis.unet_field.unet_bundle.visualization import symmetric_percentile_limit

        scale = _ps.UNET_DISPLAY_SCALE
        # Same y-up -> top-left flip as _draw_input_panels.
        pred_u_disp = np.asarray(pred_u, dtype=np.float32)[::-1, :] * scale
        pred_v_disp = -np.asarray(pred_v, dtype=np.float32)[::-1, :] * scale
        pred_speed_disp = np.sqrt(pred_u_disp ** 2 + pred_v_disp ** 2)
        lim_u = symmetric_percentile_limit(pred_u_disp, pct=99.0)
        lim_v = symmetric_percentile_limit(pred_v_disp, pct=99.0)
        vmax_sp = float(max(np.percentile(pred_speed_disp, 99.0), 1e-12))
        payloads = (
            (pred_u_disp,     -lim_u, lim_u),
            (pred_v_disp,     -lim_v, lim_v),
            (pred_speed_disp,  0.0,    vmax_sp),
        )
        self._draw_panels(
            self.fig_outputs, self._output_axes, self._output_cbars,
            _ps.OUTPUT_SPECS, payloads, self.canvas_outputs,
        )
        self._redraw_vector_overlay()
        self.canvas_outputs.draw_idle()
        if hasattr(self, "btn_vectors"):
            self.btn_vectors.setEnabled(True)

    def _draw_panels(self, fig, axes_list, cbars, specs, payloads, canvas) -> None:
        self._init_plot_slots(fig, axes_list, specs)
        for idx, ((title, cmap, unit), (data, vmin, vmax)) in enumerate(zip(specs, payloads, strict=False)):
            self._draw_slot(axes_list, cbars, fig, idx, data, cmap, title, unit, vmin, vmax)
        self._sync_cbar_positions(canvas, axes_list, cbars)
        canvas.draw_idle()

    def _draw_slot(self, axes_list, cbars, fig, idx, data, cmap, title, unit, vmin, vmax) -> None:
        ax = axes_list[idx]
        cb = cbars[idx]
        if cb is not None:
            try:
                cb.remove()
            except Exception:
                pass
            cbars[idx] = None

        ax.clear()
        ax.set_aspect("equal", adjustable="box", anchor="C")
        ax.set_title(title, fontsize=10)
        self._apply_axis_ticks(ax, data.shape)

        im = ax.imshow(data, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax, interpolation="nearest")
        cax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
        cb = fig.colorbar(im, cax=cax)
        if unit:
            cb.set_label(unit, fontsize=9)
        cbars[idx] = cb
        # Lift the "1e-6"-style offset label above the top tick so it no
        # longer crowds the topmost tick label.
        ot = cb.ax.yaxis.get_offset_text()
        ot.set_fontsize(8)
        ot.set_horizontalalignment("left")
        ot.set_verticalalignment("bottom")
        ot.set_position((0.0, 1.04))
        cb.ax.tick_params(labelsize=8)

    def _apply_axis_ticks(self, ax, shape) -> None:
        h, w = int(shape[0]), int(shape[1])
        xticks = [t for t in _ps.AXIS_TICKS if 0 <= t <= w]
        yticks = [t for t in _ps.AXIS_TICKS if 0 <= t <= h]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        # Default imshow extent is (-0.5, w - 0.5); pad so a tick sitting on
        # the pixel edge (e.g. 128 on a 128-wide array) is still visible.
        x_hi = max(w - 0.5, max(xticks) if xticks else 0)
        y_hi = max(h - 0.5, max(yticks) if yticks else 0)
        ax.set_xlim(-0.5, x_hi)
        ax.set_ylim(y_hi, -0.5)
        ax.tick_params(axis="both", labelsize=8)

    def _sync_cbar_positions(self, canvas, axes_list, cbars, redraw: bool = True) -> None:
        if redraw:
            canvas.draw()

        canvas_w = max(1, canvas.width())
        canvas_h = max(1, canvas.height())

        cbar_w = _ps.CBAR_W_PX / canvas_w
        cbar_h = _ps.CBAR_H_PX / canvas_h
        cbar_gap = _ps.CBAR_GAP_PX / canvas_w

        for ax, cb in zip(axes_list, cbars, strict=False):
            if cb is None:
                continue
            bb = ax.get_position()
            y0 = bb.y0 + (bb.height - cbar_h) * 0.5
            cb.ax.set_position([bb.x1 + cbar_gap, y0, cbar_w, cbar_h])

    def _on_input_resize(self, event) -> None:
        if self._input_resizing:
            return
        self._input_resizing = True
        QTimer.singleShot(0, self._finish_input_resize)

    def _finish_input_resize(self) -> None:
        try:
            self._sync_cbar_positions(
                self.canvas_inputs, self._input_axes, self._input_cbars, redraw=False
            )
            self.canvas_inputs.draw_idle()
        finally:
            self._input_resizing = False

    def _on_output_resize(self, event) -> None:
        if self._output_resizing:
            return
        self._output_resizing = True
        QTimer.singleShot(0, self._finish_output_resize)

    def _finish_output_resize(self) -> None:
        try:
            self._sync_cbar_positions(
                self.canvas_outputs, self._output_axes, self._output_cbars, redraw=False
            )
            self.canvas_outputs.draw_idle()
        finally:
            self._output_resizing = False

    def _toggle_vectors(self, checked: bool) -> None:
        self._show_vectors = bool(checked)
        self._redraw_vector_overlay()
        self.canvas_outputs.draw_idle()

    def _on_vector_n_changed(self, _value: int) -> None:
        if self._show_vectors:
            self._redraw_vector_overlay()
            self.canvas_outputs.draw_idle()

    def _redraw_vector_overlay(self) -> None:
        from opennta.analysis.unet_field.unet_bundle.visualization import normalized_quiver

        for artist in self._vector_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._vector_artists = []

        if not self._show_vectors or self._last_result is None:
            return
        if len(self._output_axes) < 3:
            return

        ax = self._output_axes[2]  # speed-prediction panel
        n = int(self.sb_vector_n.value()) if hasattr(self, "sb_vector_n") else _ps.VECTOR_N_DEFAULT
        q = normalized_quiver(
            ax,
            self._last_result["pred_u_sc"],
            self._last_result["pred_v_sc"],
            n,
            color=_ps.VECTOR_COLOR,
            flip_y=True,
        )
        if q is not None:
            self._vector_artists.append(q)
