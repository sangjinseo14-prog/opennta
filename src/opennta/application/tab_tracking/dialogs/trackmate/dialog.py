from __future__ import annotations

from collections.abc import Sequence

from PyQt5.QtWidgets import QDialog, QMessageBox

from opennta.tracking.types import FolderItem, ProcessingParams

from ....common.fonts import get_app_font
from ._plot_builder import _PlotBuilderMixin
from ._ui_builder import _UIBuilderMixin
from .preview_worker import TrackMatePreviewResult, TrackMatePreviewWorker


class TrackMateDialog(_PlotBuilderMixin, _UIBuilderMixin, QDialog):

    # (ProcessingParams field, dialog widget attr, value cast)
    _NUMERIC_FIELDS: tuple[tuple[str, str, type], ...] = (
        ("normalization_percentile", "spin_normalization_percentile", float),
        ("particle_radius_pixels", "spin_particle_radius", float),
        ("fit_fraction", "spin_fit_fraction", float),
        ("significance_alpha", "spin_significance_alpha", float),
        ("linking_max_distance", "spin_linking_max_distance", int),
        ("gap_closing_max_distance", "spin_gap_closing_max_distance", int),
        ("max_frame_gap", "spin_max_frame_gap", int),
        ("min_track_frames", "spin_min_track_frames", int),
        ("border_margin_pixels", "spin_border_margin_pixels", int),
    )

    def __init__(
        self,
        initial_params: ProcessingParams,
        fiji_path: str | None,
        folder_items: Sequence[FolderItem],
        default_index: int,
        script_threshold_path: str,
        parent=None,
    ):
        super().__init__(parent)

        self._fiji_path = fiji_path
        self._folder_items: list[FolderItem] = list(folder_items)
        self._default_index = max(0, min(int(default_index), max(0, len(self._folder_items) - 1)))
        self._script_threshold_path = script_threshold_path

        self._initial_params = initial_params
        self._worker: TrackMatePreviewWorker | None = None
        self._results: dict[str, TrackMatePreviewResult] = {}

        self._init_plot_state()
        self._build_ui()
        self.setFont(get_app_font())
        self._reset_preview_axes()
        self._load_processing_params(initial_params)
        if self._folder_items:
            self.combo_sample.setCurrentIndex(self._default_index)

    def _format_sample_key(self, item: FolderItem) -> str:
        _sub_path, _tiffs, nta, sub = item
        return f"{nta}_{sub}"

    def _load_processing_params(self, p: ProcessingParams) -> None:
        for field, widget_attr, cast in self._NUMERIC_FIELDS:
            getattr(self, widget_attr).setValue(cast(getattr(p, field)))
        idx = self.combo_quality_model.findText(p.quality_model)
        if idx < 0:
            idx = self.combo_quality_model.findText("Cheng-Schwartzman")
        self.combo_quality_model.setCurrentIndex(max(idx, 0))

    def get_processing_params(self) -> ProcessingParams:
        values = {f: getattr(self, w).value() for f, w, _ in self._NUMERIC_FIELDS}
        return ProcessingParams(
            **values,
            method=self._initial_params.method,
            detection_method=self._initial_params.detection_method,
            linking_method=self._initial_params.linking_method,
            quality_model=self.combo_quality_model.currentText() or "Cheng-Schwartzman",
        )

    def _set_preview_controls_enabled(self, enabled: bool) -> None:
        self.button_preview.setEnabled(enabled)
        self.button_run.setEnabled(enabled)
        self.combo_sample.setEnabled(enabled)
        self.spin_frame_index.setEnabled(enabled)

    def _on_preview(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Preview", "Preview is already running.")
            return
        if not self._fiji_path:
            QMessageBox.warning(self, "Preview", "Fiji path not set.")
            return
        if not self._folder_items:
            QMessageBox.warning(self, "Preview", "No samples available.")
            return

        idx = self.combo_sample.currentIndex()
        if idx < 0 or idx >= len(self._folder_items):
            QMessageBox.warning(self, "Preview", "Invalid sample selection.")
            return

        sub_path, tiff_paths, nta, sub = self._folder_items[idx]
        sample_key = self._format_sample_key(self._folder_items[idx])
        frame_index = int(self.spin_frame_index.value())

        params = self.get_processing_params()
        self.log_view.clear()
        self.status_label.setText(f"Running preview on {sample_key} (frame {frame_index})...")
        self._set_preview_controls_enabled(False)

        self._worker = TrackMatePreviewWorker(
            fiji_path=self._fiji_path,
            tiff_paths=tiff_paths,
            sub_path=sub_path,
            sample_key=sample_key,
            nta_folder=nta,
            sub=sub,
            frame_index=frame_index,
            params=params,
            script_threshold_path=self._script_threshold_path,
            parent=self,
        )
        self._worker.log.connect(self._on_worker_log)
        self._worker.finished_ok.connect(self._on_worker_ok)
        self._worker.finished_error.connect(self._on_worker_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_log(self, message: str) -> None:
        self.log_view.append(message)

    def _on_worker_ok(self, result: TrackMatePreviewResult) -> None:
        self.status_label.setText(
            f"{result.sample_key}  frame={result.frame_index}   "
            f"threshold = {result.threshold:.4f}   "
            f"spots above = {result.n_above}/{result.n_total}   "
            f"converged = {result.converged}"
        )
        self._results[result.sample_key] = result
        self._histogram_caches.pop(result.sample_key, None)
        self._render_preview(result)

    def _on_worker_error(self, message: str) -> None:
        self.status_label.setText(f"Preview failed: {message}")
        QMessageBox.warning(self, "Preview", message)

    def _on_worker_finished(self) -> None:
        self._set_preview_controls_enabled(True)
        self._worker = None

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(10000)
        super().reject()
