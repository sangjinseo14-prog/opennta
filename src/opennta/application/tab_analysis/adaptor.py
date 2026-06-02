"""Qt adapters that bridge the Qt-free analysis processors to Qt signals."""

from __future__ import annotations

import logging
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from ...analysis.analysis_processor import AnalysisConfig, AnalysisProcessor, AnalysisResults
from ...analysis.drift_corrector import (
    DriftCorrector,
    drift_correction_ui_label_to_mode,
    get_drift_correction_ui_labels,
)
from ...analysis.size_distributor import (
    get_size_distribution_ui_labels,
    size_distribution_ui_label_to_mode,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisWorker",
    "drift_correction_ui_label_to_mode",
    "get_drift_correction_ui_labels",
    "get_size_distribution_ui_labels",
    "size_distribution_ui_label_to_mode",
]


class AnalysisWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    results_ready = pyqtSignal(object)

    def __init__(
        self,
        csv_path: str,
        config: AnalysisConfig,
        lag_frame: int,
        correction_mode: str | None = None,
        pre_configured_corrector: DriftCorrector | None = None,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.config = config
        self.results = AnalysisResults()
        self.correction_mode = correction_mode
        self.lag_frame = lag_frame
        self.pre_configured_corrector = pre_configured_corrector

    PROGRESS_MAP = {
        2000: 5, 2001: 10, 2002: 20, 2003: 30, 2004: 40,
        2005: 50, 2006: 60, 2007: 70, 2008: 80, 2009: 85,
        2010: 90, 2011: 100,
    }

    def _on_progress(self, status_code: int) -> None:
        progress = self.PROGRESS_MAP.get(status_code)
        if progress is not None:
            self.progress.emit(progress)

    def _on_log(self, message: str) -> None:
        self.log.emit(message)

    def cancel(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        try:
            self.processor = AnalysisProcessor(
                config=self.config,
                correction_mode=self.correction_mode,
                progress_callback=self._on_progress,
                log_callback=self._on_log,
                pre_configured_corrector=self.pre_configured_corrector,
            )
            results = self.processor.run_full_analysis(
                self.csv_path,
                lag_frame=self.lag_frame,
            )

            if results:
                self.results = results
                self.finished.emit(True, "Analysis completed successfully")
            else:
                self.finished.emit(False, "Analysis failed - no results")
        except Exception as e:
            logger.exception("AnalysisWorker failed")
            tb = traceback.format_exc()
            self.log.emit(tb)
            self.finished.emit(False, f"{type(e).__name__}: {e}")

