import atexit
import logging
import os
import tempfile
from typing import Any

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QMessageBox,
    QTabWidget,
    QWidget,
)

from ..common.report_templates import get_analysis_report_template
from .adaptor import (
    drift_correction_ui_label_to_mode,
    get_drift_correction_ui_labels,
    get_size_distribution_ui_labels,
    size_distribution_ui_label_to_mode,
)

logger = logging.getLogger(__name__)

# QWebEngineView's setHtml() truncates payloads at ~2 MB; spill larger reports
# to a temp .html file and load() them instead.
INLINE_HTML_BYTES_LIMIT = 1_900_000

DEFAULT_LAG_FRAME = 2


def _update_progress_bar(bar, progress: int | None = None, msg: str | None = None) -> None:
    if bar is None:
        return
    if progress is None:
        progress = bar.value()
    progress = max(0, min(100, int(progress)))
    bar.setValue(progress)
    bar.setFormat(f"{msg} %p%" if msg is not None else "%p%")


def _reset_progress_bar(bar) -> None:
    if bar is not None:
        bar.setValue(0)
        bar.setFormat("%p%")


def _set_widgets_enabled(parent, widget_names: list, enabled: bool) -> None:
    for name in widget_names:
        w = getattr(parent, name, None)
        if w is None:
            w = parent.findChild(QWidget, name)
        if w is not None:
            w.setEnabled(enabled)


def _get_lag_frame(line_edit) -> int:
    try:
        return int(line_edit.text().strip())
    except (ValueError, TypeError, AttributeError):
        return DEFAULT_LAG_FRAME


def _set_lag_seconds(lag_frame_widget, lag_sec_widget, fps: float) -> None:
    try:
        lag_frames = int(lag_frame_widget.text())
        lag_sec_widget.setText(str(round(lag_frames / fps, 3)))
    except (ValueError, TypeError, AttributeError):
        pass


class TabAnalysisUi:

    def __init__(self, parent):
        self.parent = parent

        self._report_tmp_files: list[str] = []
        atexit.register(self._cleanup_report_tmp_files)

        self._populate_comboboxes()

    def _populate_comboboxes(self) -> None:
        combo = self.parent.analysis_comboBox_distributionMode
        combo.clear()
        combo.addItems(get_size_distribution_ui_labels())

        combo = self.parent.analysis_comboBox_correctionMode
        combo.clear()
        combo.addItems(get_drift_correction_ui_labels())

    def _cleanup_report_tmp_files(self):
        remaining = []
        for path in self._report_tmp_files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                remaining.append(path)
        self._report_tmp_files = remaining

    def display_matplotlib_figure(self, graphics_view, fig, dpi=80):
        if fig is None:
            logger.warning("No figure to display.")
            return

        vp = graphics_view.viewport()
        w, h = vp.width(), vp.height()
        if w <= 0 or h <= 0:
            w = graphics_view.width()
            h = graphics_view.height()

        SUPERSAMPLE = 2
        fig.set_size_inches(max(1, w / dpi), max(1, h / dpi))
        fig.set_dpi(dpi * SUPERSAMPLE)

        for ax in fig.axes:
            for ln in ax.lines:
                ln.set_antialiased(True)
            for coll in ax.collections:
                try:
                    coll.set_antialiaseds(True)
                except Exception:
                    pass
            for p in ax.patches:
                p.set_antialiased(True)
            for t in ax.texts:
                t.set_antialiased(True)

        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background: transparent;")
        canvas.draw()

        width, height = canvas.get_width_height()
        buf = canvas.buffer_rgba()
        qimage = QImage(buf, width, height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)

        scene = QGraphicsScene()
        item = scene.addPixmap(pixmap)
        item.setTransformationMode(Qt.SmoothTransformation)

        graphics_view.setScene(scene)
        graphics_view.setRenderHints(
            QPainter.Antialiasing
            | QPainter.TextAntialiasing
            | QPainter.SmoothPixmapTransform
            | QPainter.HighQualityAntialiasing
        )
        graphics_view.fitInView(item, Qt.KeepAspectRatio)

    def _force_render_all_tabs(self, tab_widget: QTabWidget) -> None:
        def render_tabs_recursively(widget: QTabWidget) -> None:
            current = widget.currentIndex()
            for i in range(widget.count()):
                widget.setCurrentIndex(i)
                QApplication.processEvents()

                page = widget.widget(i)

                for child in page.findChildren(QTabWidget):
                    render_tabs_recursively(child)

            widget.setCurrentIndex(current)

        if tab_widget is not None:
            tab_widget.setUpdatesEnabled(False)
            render_tabs_recursively(tab_widget)
            tab_widget.setUpdatesEnabled(True)

    def update_results_display(self, stats_dict: dict[str, Any], html_report: str = None,
                               file_path: str | None = None):
        if  html_report is None:
            html_report = get_analysis_report_template(
                stats_dict,
                title="Statistical Report",
                file_path=file_path,
            )

        if hasattr(self.parent, 'analysis_textBrowser_report'):
            self._set_report_html(self.parent.analysis_textBrowser_report, html_report)

        if hasattr(self.parent, 'analysis_textBrowser_summary'):
            results_lines = []

            if 'diameter' in stats_dict:
                diameter_stats = stats_dict['diameter']
                results_lines.append(f"Sample Count: {len(diameter_stats['data'])}")
                results_lines.append(f"Mean: {diameter_stats['mean']:.1f} nm")
                results_lines.append(f"Median: {diameter_stats['median']:.1f} nm")
                results_lines.append(f"Mode: {diameter_stats['mode']:.1f} nm" if diameter_stats['mode'] is not None else "Mode: N/A")
                results_lines.append(f"Std Dev: {diameter_stats['std']:.1f} nm")

                if diameter_stats['peaks'] and len(diameter_stats['peaks']) > 0:
                    results_lines.append("")
                    for i, peak in enumerate(diameter_stats['peaks'][:5], 1):
                        results_lines.append(f"  Peak {i}: {peak['mode']:.3f} nm (count: {peak['count']:.0f})")

            results_text = "\n".join(results_lines)
            self.parent.analysis_textBrowser_summary.setPlainText(results_text)

    def show_warning(self, message: str):
        QMessageBox.warning(self.parent, "Warning", message)

    def show_error(self, message: str):
        QMessageBox.critical(self.parent, "Error", message)

    def show_insufficient_tracks_warning(self, insufficient_tracks):
        violated = len(insufficient_tracks)
        preview = ", ".join([f"{pid}({n})" for pid, n in insufficient_tracks[:10]])
        more = violated - min(10, violated)
        detail = preview + (f", +{more} more" if more > 0 else "")

        QMessageBox.warning(
            self.parent,
            "MSD Fitting Warning",
            f"minimum points (2) for tracks.\n"
            f"Examples: {detail}"
        )

    def update_progress_bar(self, progress: int | None = None, msg: str | None = None):
        _update_progress_bar(self.parent.main_progressBar_progress, progress=progress, msg=msg)

    def reset_progress_bar(self):
        _reset_progress_bar(self.parent.main_progressBar_progress)


    def set_processing_ui_enabled(self, enabled: bool):
        widget_names = [
            "analysis_pushButton_browse",
            "analysis_pushButton_runAnalysis",
            "analysis_pushButton_replot",
            "analysis_pushButton_exportSize",
            "analysis_pushButton_exportReport",
            "pushButton_Config",
            "analysis_pushButton_config",
            "analysis_lineEdit_lagFrame",
            "analysis_lineEdit_r2Threshold",
            "analysis_lineEdit_minD", "analysis_lineEdit_maxD",
            "analysis_lineEdit_binNum", "analysis_lineEdit_maxC", "analysis_lineEdit_tickC",
            "analysis_lineEdit_specialTick",
            "analysis_checkBox_csv",
            "analysis_checkBox_logarithmPlot",
            "analysis_comboBox_correctionMode",
            "analysis_comboBox_distributionMode",
        ]
        _set_widgets_enabled(self.parent, widget_names, enabled)

    def read_lag_frame_and_refresh_seconds(self, lag_frame_widget, lag_sec_widget, fps: float):
        lag_frame = _get_lag_frame(lag_frame_widget)
        _set_lag_seconds(lag_frame_widget, lag_sec_widget, fps)
        return lag_frame

    def get_correction_mode(self) -> str:
        return drift_correction_ui_label_to_mode(
            self.parent.analysis_comboBox_correctionMode.currentText()
        )

    def get_distribution_mode(self) -> str:
        return size_distribution_ui_label_to_mode(
            self.parent.analysis_comboBox_distributionMode.currentText()
        )

    def parse_custom_axis_ticks(self) -> list[float]:
        try:
            text = self.parent.analysis_lineEdit_specialTick.text().strip()
            if not text:
                return []

            ticks = []
            for part in text.split(","):
                part = part.strip()
                if part:
                    ticks.append(float(part))
            return ticks
        except (ValueError, AttributeError):
            return []

    def _set_report_html(self, view, html: str):

        self._cleanup_report_tmp_files()

        if len(html.encode('utf-8')) < INLINE_HTML_BYTES_LIMIT:
            view.setHtml(html, QUrl.fromLocalFile(os.getcwd() + os.sep))
            return

        fd, tmp_path = tempfile.mkstemp(suffix='.html')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

        self._report_tmp_files.append(tmp_path)
        view.load(QUrl.fromLocalFile(tmp_path))
