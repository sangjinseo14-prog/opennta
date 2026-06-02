"""Qt adapter that bridges the Qt-free analysis processor to Qt signals for
batch (multi-file) runs."""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ...analysis.analysis_processor import AnalysisConfig, AnalysisProcessor
from ...analysis.drift_corrector import DriftCorrector
from ...analysis.result_filters import filter_by_r2
from ..common.statistics import StatisticsUtils
from .types import FileInfo

logger = logging.getLogger(__name__)

# Diagnostic PNG writers live in the optional `dialogs` sub-module of the
# analysis tab. If it is unavailable, fall back to a silent no-op so batch
# analysis still completes — diagnostics are non-essential output.
try:
    from ..tab_analysis.dialogs import export_field_csv, save_diagnostics
except Exception as _exc:  # pragma: no cover
    logger.warning("save_diagnostics unavailable: %s", _exc)

    def save_diagnostics(*_args, **_kwargs):
        return

    def export_field_csv(*_args, **_kwargs):
        return

__all__ = ["BatchAnalysisWorker"]


class BatchAnalysisWorker(QThread):
    progress = pyqtSignal(int, int, int)
    file_completed = pyqtSignal(int)
    group_completed = pyqtSignal(int, str)
    batch_completed = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    PROGRESS_MAP = {
        2000: 5, 2001: 10, 2002: 20, 2003: 30, 2004: 40,
        2005: 50, 2006: 60, 2007: 70, 2008: 80, 2009: 85,
        2010: 90, 2011: 100,
    }

    def __init__(
        self,
        index_checked_files: list[tuple[int, FileInfo]],
        config: AnalysisConfig,
        lag_frame: int,
        base_path: str,
        correction_mode: str,
        r2_threshold: float,
        pre_configured_corrector: DriftCorrector | None = None,
        export_corrected: bool = False,
    ):
        super().__init__()
        self.index_checked_files = index_checked_files
        self.config = config
        self.lag_frame = lag_frame
        self.base_path = base_path
        self.correction_mode = correction_mode
        self.r2_threshold = r2_threshold
        self.pre_configured_corrector = pre_configured_corrector
        self.export_corrected = export_corrected
        self.spot_files: list[FileInfo] = []
        self.all_results: list[FileInfo] = []
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def run(self):
        from .output import TabBatchAnalysisOutput

        try:
            total_files = len(self.index_checked_files)
            output_manager = TabBatchAnalysisOutput(self.base_path)
            output_dir = output_manager.create_output_directory()

            file_groups = {}
            file_to_idx = {}
            for idx, file_info in self.index_checked_files:
                group_num = file_info.get('group', 1)
                file_groups.setdefault(group_num, []).append(file_info)
                file_to_idx[id(file_info)] = idx

            n_groups = len(file_groups)
            self.log_message.emit(
                f"Batch started: {n_groups} groups, {total_files} total files"
            )

            processed_files = 0
            total_success = 0
            total_failed = 0
            total_skipped = 0

            for group_num in sorted(file_groups.keys()):
                if self._cancelled:
                    break
                group_files = file_groups[group_num]
                group_results = []
                group_success = 0
                group_failed = 0
                group_skipped = 0
                self.log_message.emit(
                    f"[Group {group_num}] started: {len(group_files)} files"
                )

                for file_info in group_files:
                    if self._cancelled:
                        break
                    idx = file_to_idx[id(file_info)]
                    try:
                        csv_path = file_info['path']

                        def file_progress_callback(
                            status_code: int,
                            _idx: int = idx,
                            _processed: int = processed_files,
                        ):
                            file_progress = self.PROGRESS_MAP.get(status_code, 0)
                            overall = int((_processed + file_progress / 100) / total_files * 100)
                            self.progress.emit(_idx, file_progress, overall)

                        processor = AnalysisProcessor(
                            config=self.config,
                            correction_mode=self.correction_mode,
                            progress_callback=file_progress_callback,
                            pre_configured_corrector=self.pre_configured_corrector,
                        )
                        results = processor.run_full_analysis(csv_path, self.lag_frame)


                        if self.pre_configured_corrector is not None:
                            diag_dir = os.path.join(
                                output_dir,
                                "velocity_fields",
                                file_info.get("folder", "unknown"),
                            )
                            stem = Path(csv_path).stem
                            try:
                                save_diagnostics(
                                    self.pre_configured_corrector,
                                    diag_dir,
                                    stem,
                                )
                            except Exception:
                                logger.exception(
                                    "save_diagnostics failed for %s", stem
                                )

                            if getattr(
                                self.pre_configured_corrector,
                                "export_field_enabled",
                                False,
                            ):
                                try:
                                    field_path = output_manager.ensure_field_csv_path(
                                        file_info.get("folder", "unknown"),
                                        file_info["name"],
                                    )
                                    export_field_csv(
                                        self.pre_configured_corrector,
                                        field_path,
                                    )
                                except Exception:
                                    logger.exception(
                                        "Field CSV export failed for %s", stem
                                    )

                        valid_diameters_by_id = filter_by_r2(
                            results.diameters_by_id,
                            results.msd_r2_by_id,
                            self.r2_threshold,
                        )
                        results.diameters_by_id = valid_diameters_by_id

                        valid_diffusion_by_id = filter_by_r2(
                            results.D_by_id,
                            results.msd_r2_by_id,
                            self.r2_threshold,
                        )
                        results.D_by_id = valid_diffusion_by_id
                        results.r2_threshold = float(self.r2_threshold)

                        stats_dict = StatisticsUtils.from_results(results)
                        diameters = np.array([d for d in results.diameters_by_id.values() if d > 0])

                        if diameters.size > 0:
                            result_data = {
                                'file': file_info['name'],
                                'path': csv_path,
                                'folder': file_info.get('folder', 'unknown'),
                                'diameters': diameters,
                                'group': group_num,
                                'stats': stats_dict,
                            }
                            if (
                                self.export_corrected
                                and results.df is not None
                                and "X_diff_corr" in results.df.columns
                                and "Y_diff_corr" in results.df.columns
                            ):
                                result_data['corrected_df'] = results.df
                                result_data['pixel_size'] = self.config.pixel_size
                            self.all_results.append(result_data)
                            group_results.append(result_data)
                            output_manager.save_individual_result(result_data)
                            self.file_completed.emit(idx)
                            self.log_message.emit(f"done: {file_info['name']}")
                            group_success += 1
                        else:
                            self.log_message.emit(f"skipped: {file_info['name']} (no diameter data)")
                            group_skipped += 1

                    except Exception:
                        logger.exception(
                            "Batch file failed: %s", file_info.get('name'),
                        )
                        self.progress.emit(idx, 0, int((processed_files / total_files) * 100))
                        group_failed += 1

                    processed_files += 1

                merge_note = ""
                if group_results:
                    merge_success = output_manager.save_merged_group_with_report(
                        group_num, group_results, log_callback=self.log_message.emit,
                    )
                    if merge_success:
                        merge_note = " (merge + report done)"
                        self.group_completed.emit(
                            group_num, f"Group {group_num} completed with {len(group_results)} files",
                        )
                    else:
                        merge_note = " (merge failed)"

                summary = f"{group_success} success, {group_failed} failed"
                if group_skipped:
                    summary += f", {group_skipped} skipped"
                self.log_message.emit(
                    f"[Group {group_num}] done: {summary}{merge_note}"
                )

                total_success += group_success
                total_failed += group_failed
                total_skipped += group_skipped

            output_manager.save_batch_log(
                self.index_checked_files, self.all_results,
                self.lag_frame, self.correction_mode, self.r2_threshold,
                config=self.config,
                export_corrected=self.export_corrected,
                pre_configured_corrector=self.pre_configured_corrector,
            )

            done_summary = f"{total_success} success, {total_failed} failed"
            if total_skipped:
                done_summary += f", {total_skipped} skipped"
            self.log_message.emit(
                f"Done {total_files} files: {done_summary}"
            )

            self.batch_completed.emit(True, f"Batch analysis completed\nResults saved to: {output_dir}")

        except Exception as e:
            error_msg = f"Batch analysis error: {type(e).__name__}: {e}"
            logger.exception(error_msg)
            self.log_message.emit(error_msg)
            self.log_message.emit(traceback.format_exc())
            self.batch_completed.emit(False, error_msg)
