from __future__ import annotations

import logging
import traceback
from collections.abc import Sequence

from PyQt5.QtCore import QThread, pyqtSignal

from ...tracking.detection_method import (
    detector_ui_label_to_mode,
    get_detector_ui_labels,
)
from ...tracking.linking_method import (
    get_linker_ui_labels,
    linker_ui_label_to_mode,
)
from ...tracking.status_codes import TrackingStatus
from ...tracking.tracking_method import (
    get_combined_method_ui_labels as get_tracking_ui_labels,
    method_ui_label_to_mode,
)
from ...tracking.tracking_processor import FolderItem, ProcessingParams, TrackingProcessor

__all__ = [
    "TrackingWorker",
    "read_tracking_modes_from_ui",
    "get_tracking_ui_labels",
    "get_detector_ui_labels",
    "get_linker_ui_labels",
]

logger = logging.getLogger(__name__)


def _read_combo_mode(parent, attr: str, to_mode, default: str = "") -> str:
    combo = getattr(parent, attr, None)
    if combo is None:
        return default
    label = combo.currentText().strip()
    if not label:
        return default
    return to_mode(label) or default


def read_tracking_modes_from_ui(parent):
    if parent.tracking_radioButton_split.isChecked():
        method_mode = "split"
        detection_mode = _read_combo_mode(
            parent, "tracking_comboBox_detector", detector_ui_label_to_mode, ""
        )
        linking_mode = _read_combo_mode(
            parent, "tracking_comboBox_linker", linker_ui_label_to_mode, ""
        )
    else:
        method_mode = _read_combo_mode(
            parent, "tracking_comboBox_method", method_ui_label_to_mode, "trackmate"
        )
        detection_mode = ""
        linking_mode = ""

    return ProcessingParams(
        method=method_mode,
        detection_method=detection_mode,
        linking_method=linking_mode,
    )


class TrackingWorker(QThread):

    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    save_path_created = pyqtSignal(str)

    def __init__(
        self,
        folder_items: Sequence[FolderItem],
        fiji_path: str,
        params: ProcessingParams | None = None,
        save_path: str | None = None,
        root_folder: str | None = None,
    ):
        super().__init__()
        self.folder_items = folder_items
        self.fiji_path = fiji_path
        self.params = params
        self.save_path = save_path
        self.root_folder = root_folder
        self.processor: TrackingProcessor | None = None

    PROGRESS_MAP = {
        TrackingStatus.STARTED: 0,
        TrackingStatus.PROCESSING_TIFF: 10,
        TrackingStatus.SAVING_IMAGES: 15,
        TrackingStatus.THRESHOLD_DETECTION_START: 20,
        TrackingStatus.THRESHOLD_DETECTION_COMPLETE: 25,
        TrackingStatus.CALCULATING_THRESHOLD: 30,
        TrackingStatus.QUALITY_LOADED: 35,
        TrackingStatus.THRESHOLD_CALCULATED: 40,
        TrackingStatus.TRACKING_START: 45,
        TrackingStatus.FULL_TRACKMATE_START: 50,
        TrackingStatus.FULL_TRACKMATE_COMPLETED: 85,
        TrackingStatus.TRACKING_COMPLETE: 100,
        TrackingStatus.ITEM_COMPLETE: 100,
        TrackingStatus.ALL_COMPLETE: 100,
    }

    def _on_progress(self, status_code: int) -> None:
        progress = self.PROGRESS_MAP.get(status_code)
        if progress is not None:
            self.progress.emit(progress)

    def _on_log(self, message: str) -> None:
        self.log.emit(message)

    def cancel(self) -> None:
        self.requestInterruption()
        if self.processor is not None:
            self.processor.cancel()

    def run(self) -> None:
        try:
            self.processor = TrackingProcessor(
                folder_items=self.folder_items,
                fiji_path=self.fiji_path,
                params=self.params,
                save_path=self.save_path,
                root_folder=self.root_folder,
                progress_callback=self._on_progress,
                log_callback=self._on_log,
            )

            if self.processor.save_path:
                self.save_path_created.emit(self.processor.save_path)

            success = self.processor.run_tracking_batch()
            save_path = getattr(self.processor, "save_path", None)

            if success:
                summary = "Tracking completed"
                if save_path:
                    summary += f"\nResults saved to: {save_path}"
                logger.info(summary)
                self.finished.emit(True, "All processing completed successfully")
            else:
                error_msg = getattr(self.processor, "error_message", None) or "Processing failed"
                self.finished.emit(False, error_msg)

        except Exception as e:
            logger.exception("Tracking: failed (exception)")
            tb = traceback.format_exc()
            self.log.emit(tb)
            self.finished.emit(False, f"{type(e).__name__}: {e}")
