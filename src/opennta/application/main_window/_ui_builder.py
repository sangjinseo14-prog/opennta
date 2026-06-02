"""Hand-coded UI builder for OpenNtaMainWindow.

Replaces the pyuic5-generated ``Ui_MainWindow``. Each top-level tab is built
by its own mixin (``_ui_tab_tracking``, ``_ui_tab_analysis``,
``_ui_tab_batch``) and the status row by ``_ui_status_bar``. ``_UIBuilderMixin``
multiply-inherits from all four, provides the orchestrator ``_build_ui`` and
shared widget helpers used across tabs.

All fonts are pixel-sized at construction time so the UI renders identically
on macOS (logical DPI 72) and Windows (logical DPI 96).
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTabWidget,
    QWidget,
)

from ._ui_constants import FS_LARGE, FS_SMALL
from ._ui_status_bar import _StatusBarBuilder
from ._ui_tab_analysis import _AnalysisTabBuilder
from ._ui_tab_batch import _BatchTabBuilder
from ._ui_tab_tracking import _TrackingTabBuilder

# Window geometry (matches the .ui).
WINDOW_INIT_W = 1096
WINDOW_INIT_H = 713
WINDOW_MIN_W = 1095
WINDOW_MIN_H = 712
TAB_MIN_W = 1095
TAB_MIN_H = 677

# Outer grid stretches shared by every tab body (logo row : content row).
TAB_BODY_LEFT_STRETCH = 4
TAB_BODY_RIGHT_STRETCH = 8
TAB_BODY_TOP_ROW_STRETCH = 1
TAB_BODY_CONTENT_ROW_STRETCH = 10

# Top-of-tab row (browse | run/FIJI/config | logo) column stretches.
TOP_ROW_BROWSE_STRETCH = 12
TOP_ROW_ACTION_STRETCH = 7
TOP_ROW_LOGO_STRETCH = 7

class _UIBuilderMixin(
    _StatusBarBuilder,
    _TrackingTabBuilder,
    _AnalysisTabBuilder,
    _BatchTabBuilder,
):
    """Builds the OpenNtaMainWindow widget tree in code."""

    def _build_ui(self) -> None:
        self.setObjectName("MainWindow")
        self.setWindowTitle("OpenNTA")
        self.resize(WINDOW_INIT_W, WINDOW_INIT_H)
        self.setMinimumSize(WINDOW_MIN_W, WINDOW_MIN_H)

        central = QWidget(self)
        central.setObjectName("centralwidget")
        self.setCentralWidget(central)
        self.centralwidget = central

        root = QGridLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(3)

        self.tabWidget = QTabWidget(central)
        self.tabWidget.setObjectName("tabWidget")
        self.tabWidget.setMinimumSize(TAB_MIN_W, TAB_MIN_H)
        self.tabWidget.setMouseTracking(False)
        self.tabWidget.setAcceptDrops(False)
        self.tabWidget.setElideMode(Qt.ElideNone)
        self.tabWidget.setTabShape(QTabWidget.Rounded)

        self.tab_tracking = self._build_tracking_tab()
        self.tab_analysis = self._build_analysis_tab()
        self.tab_batch = self._build_batch_tab()
        self.tabWidget.addTab(self.tab_tracking, "Tracking")
        self.tabWidget.addTab(self.tab_analysis, "Analysis")
        self.tabWidget.addTab(self.tab_batch, "Batch")
        self.tabWidget.setCurrentIndex(0)

        root.addWidget(self.tabWidget, 0, 0, 1, 1)
        root.addLayout(self._build_status_row(central), 1, 0, 1, 1)

    # ---- shared widget helpers ---------------------------------------

    def _register(self, name: str, widget):
        """Bind ``widget`` as both ``self.<name>`` and ``widget.objectName``.

        External code reaches into the main window via attribute access AND
        the ``_set_widgets_enabled`` fallback uses ``findChild(QWidget, name)``,
        so both contracts must hold simultaneously.
        """
        widget.setObjectName(name)
        setattr(self, name, widget)
        return widget

    @staticmethod
    def _pixel_font(size_px: int, *, bold: bool = False) -> QFont:
        font = QFont()
        font.setPixelSize(size_px)
        if bold:
            font.setBold(True)
            font.setWeight(75)
        return font

    def _make_label(
        self,
        text: str,
        *,
        size_px: int = FS_SMALL,
        bold: bool = False,
        alignment: Qt.Alignment | None = None,
        text_format: Qt.TextFormat | None = None,
        parent: QWidget | None = None,
    ) -> QLabel:
        label = QLabel(text, parent)
        label.setFont(self._pixel_font(size_px, bold=bold))
        if alignment is not None:
            label.setAlignment(alignment)
        if text_format is not None:
            label.setTextFormat(text_format)
        return label

    def _make_param_line_edit(
        self,
        *,
        default: str = "",
        read_only: bool = False,
        parent: QWidget | None = None,
    ) -> QLineEdit:
        le = QLineEdit(parent)
        le.setFont(self._pixel_font(FS_SMALL))
        # NB: setReadOnly BEFORE the [readOnly] property selector in EXTRA_QSS
        # is evaluated at first paint.
        le.setReadOnly(read_only)
        if default:
            le.setText(default)
        return le

    @staticmethod
    def _expanding_vertical_spacer(width: int = 20, height: int = 40) -> QSpacerItem:
        return QSpacerItem(width, height, QSizePolicy.Minimum, QSizePolicy.Expanding)

    @staticmethod
    def _expanding_horizontal_spacer(width: int = 40, height: int = 20) -> QSpacerItem:
        return QSpacerItem(width, height, QSizePolicy.Expanding, QSizePolicy.Minimum)

    def _build_browse_row_layout(
        self,
        *,
        button_name: str,
        line_edit_name: str,
        line_edit_read_only: bool = True,
        parent: QWidget | None = None,
    ) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.addItem(self._expanding_vertical_spacer(), 0, 0, 1, 1)
        grid.addItem(self._expanding_vertical_spacer(), 2, 0, 1, 1)

        line_edit = QLineEdit(parent)
        line_edit.setReadOnly(line_edit_read_only)
        line_edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._register(line_edit_name, line_edit)
        grid.addWidget(line_edit, 1, 1, 1, 1)

        button = QPushButton("Browse", parent)
        button.setFont(self._pixel_font(FS_LARGE))
        button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self._register(button_name, button)
        grid.addWidget(button, 1, 0, 1, 1)

        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 8)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 2)
        grid.setRowStretch(2, 1)
        return grid

    def _build_logo_view_layout(
        self,
        *,
        view_name: str,
        transformation_anchor: bool = False,
        parent: QWidget | None = None,
    ) -> QHBoxLayout:
        """The logo column of each tab's top row."""
        from PyQt5.QtWidgets import QFrame, QGraphicsView

        row = QHBoxLayout()
        view = QGraphicsView(parent)
        view.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        view.setFrameShape(QFrame.NoFrame)
        view.setLineWidth(0)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if transformation_anchor:
            view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self._register(view_name, view)
        row.addWidget(view)
        row.setStretch(0, 15)
        return row
