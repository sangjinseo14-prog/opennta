import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox

from ...analysis.corrected_track_exporter import write_corrected_track_csv
from ..common.path_selector import PathSelector
from ..common.report_templates import get_analysis_report_template

logger = logging.getLogger(__name__)


class TabAnalysisHelper:

    def __init__(self, parent, browse_dir: str, export_dir: str):
        self.parent = parent
        self._csv_path: str | None = None
        self._browse_dir = browse_dir
        self._export_dir = export_dir
        self._desktop_dir = export_dir

    def select_csv_file(self):
        path, new_dir = PathSelector.select_csv_file(
            self.parent,
            self._browse_dir,
            "Select CSV File for Analysis"
        )

        if path:
            self._browse_dir = new_dir
            self._csv_path = path
            self.parent.analysis_lineEdit_browse.setText(path)


        return path

    def export_diameter_csv(self, results: Any):
        try:
            if not results or results.diameters_by_id is None:
                QMessageBox.warning(self.parent, "Warning", "No diameter data to export.")
                return

            path, new_dir = PathSelector.save_csv_file(
                self.parent,
                self._export_dir,
                "diameter_export.csv",
                "Export Diameter Data"
            )

            if not path:
                return

            self._export_dir = new_dir

            diameters = np.array([d for d in results.diameters_by_id.values() if d > 0])

            df_export = pd.DataFrame({"Estimated_Diameter_nm": diameters})
            df_export.to_csv(path, index=False, header=False)

            logger.info("Analysis CSV exported: %s (rows=%d)", path, len(df_export))
            QMessageBox.information(self.parent, "Export Complete", f"CSV saved to:\n{path}")

        except Exception as e:
            logger.exception("Analysis CSV export failed")
            QMessageBox.critical(self.parent, "Export Error", str(e))

    def export_report_html(self, stats_dict: dict | None):
        try:
            if not stats_dict:
                QMessageBox.warning(self.parent, "Warning", "No report data to export.")
                return

            path, new_dir = PathSelector.save_html_file(
                self.parent,
                self._export_dir,
                "analysis_report.html",
                "Export Analysis Report"
            )

            if not path:
                return

            self._export_dir = new_dir

            html = get_analysis_report_template(
                stats_dict,
                title="Statistical Report",
                file_path=self._csv_path,
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info("Analysis HTML report exported: %s", path)
            QMessageBox.information(self.parent, "Export Complete", f"HTML saved to:\n{path}")

        except Exception as e:
            logger.exception("Analysis HTML export failed")
            QMessageBox.critical(self.parent, "Export Error", str(e))

    def export_corrected_track(self, df: pd.DataFrame, original_csv_path: str, pixel_size: float):
        if df is None or df.empty:
            QMessageBox.warning(self.parent, "Warning", "No corrected track data to export.")
            return

        if "X_diff_corr" not in df.columns or "Y_diff_corr" not in df.columns:
            QMessageBox.warning(self.parent, "Warning", "Drift correction has not been applied.")
            return

        stem, ext = os.path.splitext(os.path.basename(original_csv_path))
        output_path = os.path.join(self._desktop_dir, f"{stem}_corrected{ext}")

        rows = write_corrected_track_csv(df, original_csv_path, output_path, pixel_size)
        if rows is None:
            QMessageBox.warning(self.parent, "Warning", "Unsupported CSV format.")
            return
        logger.info("Analysis corrected-track CSV exported: %s (rows=%d)", output_path, rows)

    def get_csv_path(self) -> str | None:
        return self._csv_path
