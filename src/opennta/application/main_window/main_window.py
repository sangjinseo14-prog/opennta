"""OpenNtaMainWindow — the top-level Qt window that owns each tab."""

from __future__ import annotations

import logging

import psutil
from PyQt5.QtCore import QPoint, QSize, Qt, QTimer
from PyQt5.QtGui import QCursor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QToolTip,
)

from ... import __version__
from ..common import get_app_font
from ..config import ConfigDialog, ConfigManager
from ..tab_analysis import TabAnalysis
from ..tab_batch_analysis import TabBatchAnalysis
from ..tab_tracking import TabTracking
from ._ui_builder import _UIBuilderMixin
from .branding import generate_logo, make_fit_to_view_icon, resolve_logo_path

logger = logging.getLogger(__name__)

_PROCESS = psutil.Process()

EXTRA_QSS = """
    QToolTip {
        color: #8caaed;
        padding: 6px;
        font-size: 13px;
    }
    QLineEdit[readOnly="false"] {
        background-color: #304356 !important;
        color: #FFFFFF !important;
    }
    QProgressBar {
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: rgb(52, 145, 220);
    }
"""


class OpenNtaMainWindow(_UIBuilderMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self._apply_app_stylesheet()
        self._apply_app_font()
        self._build_ui()
        self._apply_window_icon()

        self.config_manager = ConfigManager(parent=self)

        self._tabs_initialized = False
        self._initial_size = None
        self._initial_captured = False

        self.analysis_pushButton_config.clicked.connect(self._open_config)
        self.batch_pushButton_config.clicked.connect(self._open_config)
        self.main_pushButton_resize.clicked.connect(self._restore_initial_window_size)
        self.main_pushButton_resize.setIcon(make_fit_to_view_icon(22))
        self.main_pushButton_resize.setIconSize(QSize(22, 22))

        self.analysis_tab = TabAnalysis(parent=self)
        self.track_tab = TabTracking(parent=self)
        self.batch_tab = TabBatchAnalysis(parent=self)

        self._start_memory_monitor(interval_ms=1000)

    def closeEvent(self, event):
        # Stop any running worker thread before teardown so Qt never destroys a
        # QThread that is still executing (crash/segfault on exit). Each tab
        # cancels its worker (killing any Fiji subprocess) and joins it.
        for tab in (self.analysis_tab, self.track_tab, self.batch_tab):
            shutdown = getattr(tab, "shutdown_active_worker", None)
            if shutdown is None:
                continue
            try:
                shutdown()
            except Exception:
                logger.exception("Worker shutdown failed during close")
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)

        if not self._tabs_initialized:
            self._initialize_all_tabs()
            self._tabs_initialized = True

        if not self._initial_captured:
            self._initial_captured = True
            QTimer.singleShot(0, self._capture_initial_size)

        views = [self.tracking_graphicsView_logo, self.analysis_graphicsView_logo,
                 self.batch_graphicsView_logo]
        self._setup_logo_graphics_views(views)

    def _initialize_all_tabs(self):
        def activate_tabs(widget):
            if isinstance(widget, QTabWidget):
                current = widget.currentIndex()
                for i in range(widget.count()):
                    widget.setCurrentIndex(i)
                    QApplication.processEvents()
                    activate_tabs(widget.widget(i))
                widget.setCurrentIndex(current)

            for child in widget.findChildren(QTabWidget):
                activate_tabs(child)

        activate_tabs(self)

    def _capture_initial_size(self):
        if self.isMaximized() or self.isFullScreen():
            self._initial_size = self.normalGeometry().size()
        else:
            self._initial_size = self.size()

    def _restore_initial_window_size(self):
        if self._initial_size is None:
            self.resize(self.minimumSizeHint())
            return

        if self.isMaximized() or self.isFullScreen():
            self.showNormal()
        self.resize(self._initial_size)

    def _apply_app_stylesheet(self):
        import qdarkstyle

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5() + EXTRA_QSS)

    def _apply_app_font(self):
        app = QApplication.instance()
        if app is None:
            return
        app.setFont(get_app_font())

    def _apply_window_icon(self):
        logo_path = resolve_logo_path()
        if not logo_path.exists():
            return
        icon = QIcon(str(logo_path))
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _open_config(self):
        try:
            dialog = ConfigDialog(self.config_manager, parent=self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Setting dialog error:\n{e}")

    def _setup_logo_graphics_views(self, views: list[QGraphicsView], logo_height: int = 50):
        logo_image = generate_logo(view_height=logo_height)
        if logo_image is None:
            return

        pixmap = QPixmap.fromImage(logo_image)

        for view in views:
            scene = QGraphicsScene(view)
            scene.addPixmap(pixmap)

            view.setScene(scene)
            view.setFrameShape(QGraphicsView.NoFrame)
            view.setStyleSheet("background: transparent; border: none;")
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            view.setAlignment(Qt.AlignCenter)
            view.setToolTip(f"opennta v{__version__}")

    def _start_memory_monitor(self, interval_ms: int = 1000) -> QTimer:
        timer = QTimer(self)
        timer.timeout.connect(self._update_memory_status)
        timer.start(interval_ms)
        return timer

    def _update_memory_status(self):
        try:
            mem_used = _PROCESS.memory_info().rss / 1024 / 1024
            mem_info = psutil.virtual_memory()
            mem_total = mem_info.total / 1024 / 1024
            mem_avail = mem_info.available / 1024 / 1024
            mem_other = mem_total - mem_avail - mem_used

            summary = f"This App: {mem_used:.1f} MB / Free: {mem_avail:.1f} MB"
            tooltip = f"Other apps: {mem_other:.1f} MB\nTotal memory: {mem_total:.1f} MB"

            if hasattr(self, "main_lineEdit_memoryStatus"):
                self.main_lineEdit_memoryStatus.setText(summary)
                widget = self.main_lineEdit_memoryStatus
                if widget.underMouse():
                    pos = QCursor.pos() - QPoint(-20, 50)
                    QToolTip.showText(pos, tooltip, widget)
        except BaseException:
            pass
