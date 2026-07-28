from __future__ import annotations

import logging
import os
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tifffile as tiff
from numpy.typing import NDArray
from PyQt5.QtCore import QThread, pyqtSignal

from opennta.common.progress import ProgressEmitter
from opennta.tracking.trackmate.fiji_runner import FijiRunner
from opennta.tracking.trackmate.fitting import FittingEngine
from opennta.tracking.trackmate.image_processor import ImageProcessor
from opennta.tracking.types import ProcessingParams

logger = logging.getLogger(__name__)


@dataclass
class TrackMatePreviewResult:
    sample_key: str
    nta_folder: str
    sub: str
    frame_index: int
    image: NDArray[np.unsignedinteger]
    spots_x: NDArray[np.floating]
    spots_y: NDArray[np.floating]
    qualities: NDArray[np.floating]
    threshold: float
    fit_fraction: float
    converged: bool
    fit_grid: NDArray[np.floating] | None
    fit_density: NDArray[np.floating] | None
    n_total: int
    n_above: int


class TrackMatePreviewWorker(QThread):

    log = pyqtSignal(str)
    finished_ok = pyqtSignal(object)         # TrackMatePreviewResult
    finished_error = pyqtSignal(str)

    def __init__(
        self,
        fiji_path: str,
        tiff_paths: Sequence[str],
        sub_path: str,
        sample_key: str,
        nta_folder: str,
        sub: str,
        frame_index: int,
        params: ProcessingParams,
        script_threshold_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.fiji_path = fiji_path
        self.tiff_paths = list(tiff_paths)
        self.sub_path = sub_path
        self.sample_key = sample_key
        self.nta_folder = nta_folder
        self.sub = sub
        self.frame_index = int(frame_index)
        self.params = params
        self.script_threshold_path = script_threshold_path
        self._runner: FijiRunner | None = None

    def _emit_log(self, message: str) -> None:
        self.log.emit(message)

    def cancel(self) -> None:
        # Kill the in-flight Fiji subprocess so the worker thread unwinds
        # promptly instead of being force-terminated.
        if self._runner is not None:
            self._runner.cancel()

    def run(self) -> None:
        try:
            emitter = ProgressEmitter(
                save_path=None,
                progress_callback=None,
                log_callback=self._emit_log,
            )

            self._emit_log(f"Preview: processing TIFF stack for {self.sample_key}")
            tiff_processor = ImageProcessor(emitter)
            stack = tiff_processor.normalize_stack(
                self.tiff_paths, self.params.normalization_percentile
            )

            n_frames = int(stack.shape[0])
            if self.frame_index < 0 or self.frame_index >= n_frames:
                self.finished_error.emit(
                    f"Frame index {self.frame_index} out of range (stack has {n_frames} frames)"
                )
                return

            frame_image = stack[self.frame_index].copy()
            del stack

            with tempfile.TemporaryDirectory(prefix="opennta_preview_") as tmp_dir:
                frame_path = os.path.join(tmp_dir, f"preview_frame_{self.frame_index}.tiff")
                tiff.imwrite(frame_path, frame_image)

                self._emit_log(
                    f"Preview: running Fiji threshold detection on frame {self.frame_index}"
                )
                runner = FijiRunner(self.fiji_path, emitter)
                self._runner = runner
                ok = runner.run_threshold_detection(
                    self.script_threshold_path,
                    frame_path,
                    self.params.particle_radius_pixels,
                )
                quality_csv = os.path.splitext(frame_path)[0] + "_quality.csv"

                if not ok or not os.path.isfile(quality_csv):
                    self.finished_error.emit("Detection failed or quality CSV missing")
                    return

                df = pd.read_csv(quality_csv)

            cols_lower = {str(c).lower(): c for c in df.columns}
            qcol = cols_lower.get("quality")
            xcol = cols_lower.get("x")
            ycol = cols_lower.get("y")
            if qcol is None or xcol is None or ycol is None:
                self.finished_error.emit(
                    f"Preview CSV missing X/Y/QUALITY columns: {list(df.columns)}"
                )
                return

            qualities = df[qcol].to_numpy(dtype=float)
            spots_x = df[xcol].to_numpy(dtype=float)
            spots_y = df[ycol].to_numpy(dtype=float)

            if self.params.quality_model == "No Fitting (FRAC)":
                self._emit_log("Preview: using the empirical FRAC quantile")
            else:
                self._emit_log(
                    f"Preview: fitting noise distribution ({self.params.quality_model})"
                )
            engine = FittingEngine(self.params.quality_model)
            fit_result = engine.calculate_threshold(
                u=qualities,
                alpha=self.params.significance_alpha,
                frac=self.params.fit_fraction,
                quality_csv_path=None,
            )

            grid = None
            density = None
            if (
                not engine.uses_empirical_frac
                and fit_result.converged
                and np.isfinite(fit_result.u_star_alpha)
            ):
                xlo = float(np.quantile(qualities, 0.01))
                xmed = float(np.median(qualities))
                xhi = float(xmed + 3 * (xmed - xlo))
                grid = np.linspace(xlo, xhi, 500)
                density = engine.model.pdf_untruncated(grid, **fit_result.params)

            threshold = float(fit_result.u_star_alpha)
            n_above = int(np.sum(qualities >= threshold)) if np.isfinite(threshold) else 0

            self._emit_log(
                f"Preview: threshold={threshold:.3f}, "
                f"{n_above}/{qualities.size} spots above"
            )

            result = TrackMatePreviewResult(
                sample_key=self.sample_key,
                nta_folder=self.nta_folder,
                sub=self.sub,
                frame_index=self.frame_index,
                image=frame_image,
                spots_x=spots_x,
                spots_y=spots_y,
                qualities=qualities,
                threshold=threshold,
                fit_fraction=float(self.params.fit_fraction),
                converged=bool(fit_result.converged),
                fit_grid=grid,
                fit_density=density,
                n_total=int(qualities.size),
                n_above=n_above,
            )
            self.finished_ok.emit(result)

        except Exception as e:
            logger.exception("TrackMate preview failed")
            tb = traceback.format_exc()
            self._emit_log(tb)
            self.finished_error.emit(f"{type(e).__name__}: {e}")
