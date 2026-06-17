from __future__ import annotations

import pandas as pd
from PyQt5.QtWidgets import QDialog, QMessageBox

from opennta.analysis.numerical_field.field_smoother import (
    ci95_weighted_gaussian_smooth,
)
from opennta.analysis.numerical_field.field_stats import component_stats
from opennta.analysis.numerical_field.types import (
    FieldStats,
    NumericalFieldParams,
)
from opennta.analysis.numerical_field.velocity_field import compute_velocity_field

from ....common.fonts import get_app_font
from ._plot_builder import _PlotBuilderMixin
from ._plot_style import apply_dark_plot_theme
from ._ui_builder import _UIBuilderMixin


class NumericalFieldDialog(
    _UIBuilderMixin,
    _PlotBuilderMixin,
    QDialog,
):

    def __init__(self, df: pd.DataFrame, config, parent=None):
        apply_dark_plot_theme()
        super().__init__(parent)

        self.df = df
        self.config = config

        self.stats: FieldStats | None = None
        self.smoothed_u = None
        self.smoothed_v = None
        self._use_smoothed = False

        self._init_plot_state()

        self._build_ui()
        self.setFont(get_app_font())
        self._reserve_uv_stats_height()
        self._init_plot_slots()

        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)
        self._redraw()

    def get_export_csv(self) -> bool:
        return self.chk_export_csv.isChecked()

    def get_params(self) -> NumericalFieldParams | None:

        return NumericalFieldParams(
            n_windows=self.sb_nwin.value(),
            x_range=(self.sb_xmin.value(), self.sb_xmax.value()),
            y_range=(self.sb_ymin.value(), self.sb_ymax.value()),
            outlier_k=self.sb_k.value(),
            min_count=self.sb_mincount.value(),
            ksize=self.sb_ksize.value(),
            n_iter=self.sb_niter.value(),
            sigma=self.sb_sigma.value(),
            node_count=self._node_count(),
        )

    def _node_count(self) -> int:
        # Nodes are the window centers (n_windows) unless interpolation is
        # enabled and overrides the resolution. 10 windows -> 10 nodes.
        default = self.sb_nwin.value()
        if not self.chk_interp.isChecked():
            return default
        txt = self.le_nodes.text().strip()
        try:
            return max(int(txt), 2)
        except ValueError:
            return default

    def _compute_field(self) -> None:
        try:
            self.stats = compute_velocity_field(
                df=self.df,
                n_windows=self.sb_nwin.value(),
                x_range=(self.sb_xmin.value(), self.sb_xmax.value()),
                y_range=(self.sb_ymin.value(), self.sb_ymax.value()),
                outlier_k=self.sb_k.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Field computation failed:\n{exc}")
            return

        self.smoothed_u = None
        self.smoothed_v = None
        self._use_smoothed = False
        self._update_uv_stats()
        self._redraw()

    def _update_uv_stats(self) -> None:
        # Stored values are um/frame; scale by fps so the readout matches the
        # µm/s colorbars on the velocity panels.
        if self.stats is None:
            self._clear_uv_stats()
            return
        fps = float(self.config.fps)
        u_stats = component_stats(self.stats.mean_dx * fps)
        v_stats = component_stats(self.stats.mean_dy * fps)
        self._set_uv_stats(u_stats, v_stats)

    def _apply_smoothing(self) -> None:
        if self.stats is None:
            QMessageBox.warning(self, "Warning", "Compute the field first.")
            return

        try:
            self.smoothed_u, self.smoothed_v = ci95_weighted_gaussian_smooth(
                mean_dx=self.stats.mean_dx,
                mean_dy=self.stats.mean_dy,
                ci95_vec=self.stats.ci95_vec,
                count=self.stats.count,
                min_count=self.sb_mincount.value(),
                ksize=self.sb_ksize.value(),
                n_iter=self.sb_niter.value(),
                sigma=self.sb_sigma.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Smoothing failed:\n{exc}")
            return

        self._use_smoothed = True
        self._redraw()

    def _accept_field(self) -> None:
        self.accept()
