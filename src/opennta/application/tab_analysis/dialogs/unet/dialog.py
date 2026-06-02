from __future__ import annotations

import logging

import pandas as pd
from PyQt5.QtWidgets import QDialog, QMessageBox

from opennta.analysis.unet_field.unit_convert import check_config_compatibility

from ....common.fonts import get_app_font
from . import _plot_style as _ps
from ._plot_builder import _PlotBuilderMixin
from ._plot_style import apply_dark_plot_theme
from ._ui_builder import _UIBuilderMixin

logger = logging.getLogger(__name__)


class UNetFieldDialog(
    _UIBuilderMixin,
    _PlotBuilderMixin,
    QDialog,
):
    # HLeft horizontal layout: controls, preprocessing panel, inference panel.
    # Figures use an explicit GridSpec + manual colorbar positioning so image
    # panels stay put across reruns instead of drifting via tight_layout.

    def __init__(self, df: pd.DataFrame, config, parent=None):
        apply_dark_plot_theme()
        super().__init__(parent)

        self.df = df
        self.config = config

        self._prep = None           # {"flow_df", "meta", "sample"}
        self._last_result = None    # full inference result dict

        self._input_axes: list = []
        self._input_cbars = [None, None, None]
        self._output_axes: list = []
        self._output_cbars = [None, None, None]
        self._input_resizing = False
        self._output_resizing = False
        self._show_vectors = False
        self._vector_artists: list = []
        self._config_warning_checked = False

        self._build_ui()
        self.setFont(get_app_font())

        self._init_plot_slots(self.fig_inputs, self._input_axes, _ps.INPUT_SPECS)
        self._init_plot_slots(self.fig_outputs, self._output_axes, _ps.OUTPUT_SPECS)

        self.canvas_inputs.mpl_connect("resize_event", self._on_input_resize)
        self.canvas_outputs.mpl_connect("resize_event", self._on_output_resize)

        self._draw_placeholders(self._input_axes, _ps.INPUT_SPECS, self.canvas_inputs)
        self._draw_placeholders(self._output_axes, _ps.OUTPUT_SPECS, self.canvas_outputs)

    def showEvent(self, event):
        # Pop the config-mismatch warning once, after the dialog is visible.
        super().showEvent(event)
        if not self._config_warning_checked:
            self._config_warning_checked = True
            self._check_and_show_config_warning()

    def get_export_csv(self) -> bool:
        return self.chk_export_csv.isChecked()

    def get_vector_n(self) -> int:
        if hasattr(self, "sb_vector_n"):
            return int(self.sb_vector_n.value())
        return _ps.VECTOR_N_DEFAULT

    def get_image_size(self) -> int:
        # Side length N (px) of the N×N source image the spots came from.
        try:
            n = int(self.le_image_size.text())
        except (AttributeError, TypeError, ValueError):
            n = 0
        return n if n > 0 else self._image_size_default()

    def _check_and_show_config_warning(self):
        try:
            from opennta.analysis.unet_field.unet_bundle.config import (
                ACTUAL_FPS,
                PIXEL_SIZE_X_METERS,
            )
        except Exception:
            return
        ok, msg = check_config_compatibility(self.config, PIXEL_SIZE_X_METERS, ACTUAL_FPS)
        if not ok:
            QMessageBox.warning(self, "Config mismatch with trained model", msg)

    def _run_preprocess(self):
        from opennta.analysis.unet_field.unet_bundle import get_shared_runner

        self.btn_preprocess.setEnabled(False)
        self.btn_inference.setEnabled(False)
        self.btn_accept.setEnabled(False)
        n = self.get_image_size()
        try:
            runner = get_shared_runner()
            prep = runner.prepare_sample_from_df(
                spots_df=self.df, image_width_px=n, image_height_px=n,
            )
        except FileNotFoundError as exc:
            QMessageBox.critical(
                self, "Missing model assets",
                f"Model assets are missing:\n{exc}\n\n"
                "Install the assets bundle (best.weights.h5, norm.json, "
                "reference_field.npz, meta.csv) into\n"
                "  opennta/analysis/unet_field/unet_bundle/unet_assets/"
            )
            self.btn_preprocess.setEnabled(True)
            return
        except Exception as exc:
            logger.exception("OpenCV preprocessing failed")
            QMessageBox.critical(self, "Preprocessing failed", f"{type(exc).__name__}: {exc}")
            self.btn_preprocess.setEnabled(True)
            return

        self._prep = prep
        self._last_result = None
        self._vector_artists = []

        self._draw_input_panels(prep["sample"])
        # Old predictions no longer match this sample; reset output side.
        self._init_plot_slots(self.fig_outputs, self._output_axes, _ps.OUTPUT_SPECS)
        self._draw_placeholders(self._output_axes, _ps.OUTPUT_SPECS, self.canvas_outputs)

        if hasattr(self, "btn_vectors"):
            self.btn_vectors.setChecked(False)
            self.btn_vectors.setEnabled(False)

        self.btn_preprocess.setEnabled(True)
        self.btn_inference.setEnabled(True)

    def _run_inference(self):
        from opennta.analysis.unet_field.unet_bundle import get_shared_runner

        if self._prep is None:
            QMessageBox.warning(self, "Preprocess first", "Run preprocessing (OpenCV) before inference.")
            return

        self.btn_inference.setEnabled(False)
        self.btn_accept.setEnabled(False)
        try:
            runner = get_shared_runner()
            result = runner.infer_from_sample(self._prep["sample"], self._prep["meta"])
        except Exception as exc:
            logger.exception("U-Net inference failed")
            QMessageBox.critical(self, "Inference failed", f"{type(exc).__name__}: {exc}")
            self.btn_inference.setEnabled(True)
            return

        self._last_result = result
        self._draw_output_panels(result["pred_u_sc"], result["pred_v_sc"])

        self.btn_inference.setEnabled(True)
        self.btn_accept.setEnabled(True)

    def _accept_field(self):
        if self._last_result is None:
            return
        self.accept()
