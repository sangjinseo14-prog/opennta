import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..common.report_templates import (
    format_individual_diameter_summary,
    get_analysis_report_template,
)
from ..common.statistics import StatisticsUtils
from .types import FileInfo

if TYPE_CHECKING:
    from ...analysis.drift_corrector import DriftCorrector
    from ...analysis.types import AnalysisConfig


class TabBatchAnalysisOutput:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.output_dir = None
        self.individual_dir = None
        self.merged_dir = None
        # Top-level per-feature folders for checkbox-driven CSV outputs;
        # populated lazily by the writer that needs them so they only appear
        # in the run directory when the user actually opted in.
        self.corrected_tracks_dir: str | None = None
        self.field_csv_dir: str | None = None

    def create_output_directory(self) -> str:
        date_str = datetime.now().strftime('%Y%m%d')
        num = 1

        while True:
            dir_name = f"size_{date_str}_{num}"
            output_path = os.path.join(self.base_path, dir_name)
            if not os.path.exists(output_path):
                break
            num += 1

        os.makedirs(output_path)
        individual_dir = os.path.join(output_path, "individual_results")
        merged_dir = os.path.join(output_path, "merged_results")
        os.makedirs(individual_dir)
        os.makedirs(merged_dir)

        self.output_dir = output_path
        self.individual_dir = individual_dir
        self.merged_dir = merged_dir
        self.corrected_tracks_dir = os.path.join(output_path, "corrected_tracks")
        self.field_csv_dir = os.path.join(output_path, "field_csv")

        return output_path

    def ensure_corrected_track_path(self, folder_name: str, file_name: str) -> str:
        # Per-feature top-level folder keeps checkbox outputs from cluttering
        # individual_results/, which only holds the always-on _diameter.csv +
        # _report.html.
        subdir = os.path.join(self.corrected_tracks_dir, folder_name)
        os.makedirs(subdir, exist_ok=True)
        if file_name.endswith("_spots.csv"):
            out_name = file_name.replace("_spots.csv", "_corrected.csv")
        else:
            out_name = Path(file_name).stem + "_corrected.csv"
        return os.path.join(subdir, out_name)

    def ensure_field_csv_path(self, folder_name: str, file_name: str) -> str:
        subdir = os.path.join(self.field_csv_dir, folder_name)
        os.makedirs(subdir, exist_ok=True)
        if file_name.endswith("_spots.csv"):
            out_name = file_name.replace("_spots.csv", "_field.csv")
        else:
            out_name = Path(file_name).stem + "_field.csv"
        return os.path.join(subdir, out_name)

    def save_individual_result(self, result_data: dict[str, Any]) -> str:
        folder_name = result_data.get('folder', 'unknown')
        result_subdir = os.path.join(self.individual_dir, folder_name)
        os.makedirs(result_subdir, exist_ok=True)

        csv_name = result_data['file'].replace('_spots.csv', '_diameter.csv')
        csv_path = os.path.join(result_subdir, csv_name)

        df = pd.DataFrame({'Estimated_Diameter_nm': result_data['diameters']})
        df.to_csv(csv_path, index=False, header=False)


        corrected_df = result_data.get('corrected_df')
        if corrected_df is not None:
            try:
                from ...analysis.corrected_track_exporter import write_corrected_track_csv

                corrected_path = self.ensure_corrected_track_path(
                    folder_name, result_data['file'],
                )
                write_corrected_track_csv(
                    corrected_df,
                    result_data['path'],
                    corrected_path,
                    result_data['pixel_size'],
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Corrected-track CSV export failed for %s", result_data.get('file')
                )


        stats = result_data.get('stats')
        if stats:
            report_name = result_data['file'].replace('_spots.csv', '_report.html')
            report_path = os.path.join(result_subdir, report_name)
            html = get_analysis_report_template(
                stats,
                title="Statistical Report",
                file_path=result_data.get('path'),
            )
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html)

        return csv_path

    def save_merged_group_with_report(self, group_num: int, group_data: list[dict[str, Any]],
                                      log_callback=None) -> bool:
        try:
            series_list = []
            file_list = []

            for data in group_data:
                diam = np.asarray(data['diameters'], dtype=float)
                stem = Path(data['file']).stem.replace('_spots', '')
                col_name = f"{data['folder']}/{stem}"
                s = pd.Series(diam, name=col_name)
                series_list.append(s)
                file_list.append(col_name)

            # Side-by-side concat (axis=1) is intentional — files have different
            # particle counts and we want each file in its own column with NaN
            # padding rather than a stacked long-format frame.
            merge_df = pd.concat(series_list, axis=1)
            merge_name = f"merged_group_{group_num}_diameter.csv"
            merge_path = os.path.join(self.merged_dir, merge_name)
            merge_df.to_csv(merge_path, index=False)

            meta_name = f"merged_group_{group_num}_files.txt"
            meta_path = os.path.join(self.merged_dir, meta_name)
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                for i, col in enumerate(file_list, 1):
                    f.write(f"  Col {i}: {col}\n")
                f.write("\nCounts per column:\n")
                for i, s in enumerate(series_list, 1):
                    f.write(f"  Col {i}: {s.notna().sum()} particles\n")

            merged_diameters = []
            for s in series_list:
                merged_diameters.extend(s.dropna().tolist())

            self._create_merged_group_report(group_num, group_data, merged_diameters)

            return True

        except Exception as e:
            if log_callback:
                log_callback(f"Error creating merge for Group {group_num}: {str(e)}")
            return False

    def _create_merged_group_report(self, group_num: int, group_data: list[dict[str, Any]],
                                    merged_diameters: list[float]):
        merged_stats = self._calculate_merged_statistics(merged_diameters)

        merged_paths = [data.get('path') for data in group_data if data.get('path')]

        per_file_html = self._build_per_file_summary_html(group_data)

        merged_html = get_analysis_report_template(
            merged_stats,
            title=f"Batch Analysis Report - Group {group_num}<br>"
                  f"<span style='font-size:10pt;color:#7f8c8d;'>"
                  f"Merged Statistics: {len(group_data)} files, {len(merged_diameters)} particles"
                  f"</span>",
            file_path=merged_paths,
            extra_html=per_file_html,
        )

        report_name = f"merged_group_{group_num}_report.html"
        report_path = os.path.join(self.merged_dir, report_name)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(merged_html)

    @staticmethod
    def _build_per_file_summary_html(group_data: list[dict[str, Any]]) -> str:
        # Renders a compact per-file diameter summary appended under the
        # merged statistics, in the same input order as the group.
        sections: list[str] = []
        for data in group_data:
            stats = data.get('stats') or {}
            folder = data.get('folder', '')
            file_name = data.get('file', '')
            label = f"{folder}/{file_name}" if folder else file_name
            html = format_individual_diameter_summary(label, stats)
            if html:
                sections.append(html)

        if not sections:
            return ""

        return (
            '<h2 style="margin-top:18px;">Individual file summaries</h2>'
            + ''.join(sections)
        )

    def _calculate_merged_statistics(self, diameters_array: list[float]) -> dict[str, Any]:
        diameters = np.array(diameters_array)

        stats_dict = {
            'diameter': StatisticsUtils._calculate_distribution_stats(
                diameters, kind="diameter"
            )
        }

        return stats_dict

    def save_batch_log(
        self,
        index_checked_files: list[tuple[int, FileInfo]],
        all_results: list[FileInfo],
        lag_frame: int,
        correction_mode: str,
        r2_threshold: float,
        config: "AnalysisConfig | None" = None,
        export_corrected: bool | None = None,
        pre_configured_corrector: "DriftCorrector | None" = None,
    ):
        log_content = []
        log_content.append(f"Batch Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_content.append("=" * 60)
        log_content.append(f"Output Directory: {self.output_dir}")
        log_content.append(f"Total files processed: {len(index_checked_files)}")
        log_content.append("")

        if config is not None:
            log_content.append("[Acquisition Config]")
            log_content.extend(self._format_acquisition_config(config))
            log_content.append("")

        log_content.append("[Analysis Settings]")
        log_content.append(f"Lag: {lag_frame} frames")
        log_content.append(f"Correction Mode: {correction_mode}")
        log_content.append(f"MSD R^2 threshold: {r2_threshold}")
        if export_corrected is not None:
            log_content.append(f"Export corrected CSV: {export_corrected}")
        log_content.append("")

        corrector_params = self._extract_corrector_params(pre_configured_corrector)
        if corrector_params:
            log_content.append("[Drift Corrector Parameters]")
            log_content.extend(corrector_params)
            log_content.append("")

        groups = {}
        for result in all_results:
            group = result['group']
            if group not in groups:
                groups[group] = []
            groups[group].append(result)

        log_content.append("Results by Group:")
        for group_num in sorted(groups.keys()):
            log_content.append(f"\nGroup {group_num}:")
            for result in groups[group_num]:
                log_content.append(f"{result['file']}")
                log_content.append(f"{len(result['diameters'])}")

        log_path = os.path.join(self.output_dir, "batch_analysis_log.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))

    @staticmethod
    def _format_acquisition_config(config: "AnalysisConfig") -> list[str]:
        # Mirrors the Config dialog field order/units so the log reads like the
        # dialog the user filled in, plus the two derived values that downstream
        # analysis actually consumes. config.eta is canonical Pa·s; convert back
        # to the mPa·s the dialog uses so the printed unit matches the value.
        from ...analysis import units

        eta_mpas = units.from_canonical("eta", config.eta, "mPa·s")
        rows: list[tuple[str, Any, str]] = [
            ("Sensor pixel size", config.sensor_size, "µm"),
            ("Lens magnification", config.magnification, "X"),
            ("Frame per second", config.fps, ""),
            ("Temperature", config.temp, "K"),
            ("Viscosity", eta_mpas, "mPa·s"),
            ("Pixel size", config.pixel_size, "µm/px"),
            ("dt", config.dt, "s"),
        ]
        return [
            f"{label}: {value}{(' ' + unit) if unit else ''}"
            for label, value, unit in rows
        ]

    @staticmethod
    def _extract_corrector_params(
        corrector: "DriftCorrector | None",
    ) -> list[str]:
        if corrector is None:
            return []
        try:
            params = corrector.get_parameters()
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to read corrector parameters for batch log",
            )
            return []
        if not params:
            return []

        lines: list[str] = []
        for label, entry in params.items():
            if isinstance(entry, tuple) and len(entry) == 2:
                value, unit = entry
                lines.append(f"{label}: {value}{(' ' + unit) if unit else ''}")
            else:
                lines.append(f"{label}: {entry}")
        return lines

