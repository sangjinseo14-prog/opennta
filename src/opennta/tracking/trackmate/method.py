from __future__ import annotations

import gc
import logging
import os
import time
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from ..status_codes import TrackingStatus
from ..tracking_method import TrackingMethod, register_method
from ..types import ProcessingParams
from .fiji_runner import FijiRunner
from .image_processor import ImageProcessor
from .threshold_calculator import ThresholdCalculator

logger = logging.getLogger(__name__)


_PACKAGE_DIR = Path(__file__).resolve().parent
SCRIPT_THRESHOLD = _PACKAGE_DIR / "scripts" / "thresholding.py"
SCRIPT_TRACKING = _PACKAGE_DIR / "scripts" / "tracking.py"


@register_method
class TrackMateMethod(TrackingMethod):

    mode_key = "trackmate"
    ui_label = "TrackMate (Fiji)"

    def __init__(
        self,
        params: ProcessingParams,
        emitter,
        save_path: str | None = None,
        fiji_path: str | None = None,
    ):
        super().__init__(params, emitter, save_path, fiji_path)
        if not fiji_path:
            raise ValueError("TrackMateMethod requires a fiji_path")

        self.tiff_processor = ImageProcessor(emitter)
        self.threshold_calculator = ThresholdCalculator(emitter, params.quality_model)
        self.fiji_runner = FijiRunner(fiji_path, emitter)

        self.script_tracking_path: str = str(SCRIPT_TRACKING)
        self.script_threshold_path: str = str(SCRIPT_THRESHOLD)

    def requires_configuration(self) -> bool:
        return True

    def cancel(self) -> None:
        self.fiji_runner.cancel()

    def get_parameter_log_lines(self) -> list[str]:
        return [
            "[ Quality Fitting Model ]",
            f"  Name: {self.threshold_calculator.model_name}",
            "  Parameter: Auto",
            "",
            "[ Tracker Settings ]",
            f"  Linking Max Distance: {self.params.linking_max_distance}",
            f"  Gap Closing Max Distance: {self.params.gap_closing_max_distance}",
            f"  Max Frame Gap: {self.params.max_frame_gap}",
            "",
            "[ Track Filtering ]",
            f"  Min Track Frames: {self.params.min_track_frames}",
            f"  Border Margin: {self.params.border_margin_pixels}",
            "",
        ]

    def track_subfolder(
        self,
        sub_path: str,
        tiff_paths: Sequence[str],
        nta_folder: str,
        sub: str,
    ) -> bool:
        try:
            start_time = time.time()
            sample_name = f"{nta_folder}_{sub}"

            self.emitter.emit(code=TrackingStatus.PROCESSING_TIFF)
            result_stack = self.tiff_processor.normalize_stack(
                tiff_paths, self.params.normalization_percentile
            )

            self.emitter.emit(code=TrackingStatus.SAVING_IMAGES)
            stack_path, first_frame_path = self.tiff_processor.save_stack_and_first_frame(
                result_stack, sub_path, sample_name
            )

            del result_stack
            gc.collect()

            self.emitter.emit(
                code=TrackingStatus.THRESHOLD_DETECTION_START,
                msg="Starting threshold detection",
            )
            if not self.fiji_runner.run_threshold_detection(
                self.script_threshold_path,
                first_frame_path,
                self.params.particle_radius_pixels,
            ):
                self.emitter.emit(
                    code=TrackingStatus.ERROR_THRESHOLD_DETECTION,
                    msg=f"Threshold detection failed for {first_frame_path}",
                )
                return False

            quality_csv = os.path.splitext(first_frame_path)[0] + "_quality.csv"
            self.emitter.emit(
                code=TrackingStatus.THRESHOLD_DETECTION_COMPLETE,
            )
            self.emitter.emit(
                code=TrackingStatus.CALCULATING_THRESHOLD,
                msg="Calculating optimal threshold",
            )

            threshold_result = self.threshold_calculator.calculate_threshold_from_csv(
                quality_csv_path=quality_csv,
                frac=self.params.fit_fraction,
                alpha=self.params.significance_alpha,
            )

            if not threshold_result:
                self.emitter.emit(
                    code=TrackingStatus.ERROR_THRESHOLD_CALCULATION,
                    msg="Threshold estimation failed",
                )
                return False
            if not bool(threshold_result.get("ok", False)):
                self.emitter.emit(
                    code=TrackingStatus.ERROR_THRESHOLD_CALCULATION,
                    msg="Threshold fit not converged/invalid",
                )
                return False

            threshold = float(threshold_result["u_star_alpha"])
            self.emitter.emit(
                code=TrackingStatus.THRESHOLD_CALCULATED,
                msg=f"    Threshold: {threshold:.3f}",
            )

            self._move_output_file(nta_folder, quality_csv)
            self._move_output_file(nta_folder, threshold_result.get("plot_path"))
            self._move_output_file(nta_folder, threshold_result.get("fit_csv_path"))

            self.emitter.emit(
                code=TrackingStatus.FULL_TRACKMATE_START,
                msg="Starting TrackMate tracking",
            )
            if not self.fiji_runner.run_full_tracking(
                self.script_tracking_path, stack_path, threshold, self.params
            ):
                self.emitter.emit(
                    code=TrackingStatus.ERROR_TRACKING,
                    msg=f"TrackMate failed for {stack_path}",
                )
                return False

            self.emitter.emit(code=TrackingStatus.FULL_TRACKMATE_COMPLETED)

            spots_csv = os.path.splitext(stack_path)[0] + "_spots.csv"
            if os.path.exists(spots_csv):
                self._move_output_file(nta_folder, spots_csv)

            elapsed = time.time() - start_time
            formatted = str(timedelta(seconds=int(elapsed)))

            self.emitter.emit(
                code=TrackingStatus.TRACKING_COMPLETE,
                msg=f"Tracking complete in {formatted}",
            )
            self.emitter.emit(msg="=" * 80)
            return True

        except Exception as e:
            logger.exception("Tracking item processing failed")
            self.emitter.emit(msg=str(e))
            return False


__all__ = [
    "TrackMateMethod",
    "SCRIPT_THRESHOLD",
    "SCRIPT_TRACKING",
]
