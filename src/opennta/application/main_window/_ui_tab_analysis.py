"""Builds the Analysis tab of OpenNtaMainWindow.

Layout:

    [ browse row | run-cell | logo ]                     (top row)
    [ Settings + Histogram + Summary + Export buttons    | nested
       (left frame)                                      | results QTabWidget
                                                         |  (right frame) ]

The Analysis tab is the largest and most parameter-dense, so it is the most
sensitive to stretch / spacing regressions. Every stretch from the legacy
``.ui`` is named here as an explicit constant.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ._ui_constants import FS_LARGE, FS_MEDIUM, FS_SMALL

# Outer grid (logo row + content row).
_OUTER_COL_LEFT_STRETCH = 4
_OUTER_COL_RIGHT_STRETCH = 8
_OUTER_ROW_TOP_STRETCH = 1
_OUTER_ROW_BODY_STRETCH = 10

# Top row column stretches (browse | run-cell | logo).
_TOP_BROWSE_STRETCH = 12
_TOP_RUN_STRETCH = 7
_TOP_LOGO_STRETCH = 7

# Run-button cell column / row stretches.
_RUN_COL_BUTTON_STRETCH = 2
_RUN_COL_FILL_STRETCH = 5
_RUN_ROW_TOP_STRETCH = 1
_RUN_ROW_MID_STRETCH = 2
_RUN_ROW_BOTTOM_STRETCH = 1

# Left-frame vertical stretches (config row, Setting, Histogram, Summary, spacer, exports).
_LEFT_CONFIG_STRETCH = 2
_LEFT_SETTING_STRETCH = 18
_LEFT_HISTOGRAM_STRETCH = 21
_LEFT_SUMMARY_STRETCH = 12
_LEFT_GAP_STRETCH = 1
_LEFT_EXPORTS_STRETCH = 2

# Setting / Histogram column stretches (label | edit | unit).
_FORM_COL_LABEL_STRETCH = 6
_FORM_COL_FIELD_STRETCH = 4
_FORM_COL_UNIT_STRETCH = 2

# Inline stylesheets baked into the .ui (kept verbatim, pt converted to px).
_RUN_BUTTON_QSS = "background-color: rgb(52, 145, 220);"
_EXPORT_SIZE_BUTTON_QSS = (
    "background-color: rgb(255,255,255);\n"
    "color: rgb(0, 0, 0);\n"
    "font-size: 12px;"
)
_R2_THRESHOLD_HTML = (
    "<html><head/><body><p>&nbsp;&nbsp;&nbsp;&nbsp;"
    "R<span style=\" vertical-align:super;\">2</span>"
    "&nbsp;Threshold&nbsp;</p>\n</body></html>"
)


class _AnalysisTabBuilder:
    """Mixin: ``_build_analysis_tab()`` returns the populated tab body."""

    def _build_analysis_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tab_analysis")

        grid = QGridLayout(tab)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)

        # Row 0: browse | run | logo.
        # Row 1: settings frame | results tabwidget.
        grid.addLayout(self._analysis_build_top_row(tab), 0, 0, 1, 2)
        grid.addWidget(self._analysis_build_left_frame(tab), 1, 0, 1, 1)
        grid.addWidget(self._analysis_build_right_frame(tab), 1, 1, 1, 1)

        grid.setColumnStretch(0, _OUTER_COL_LEFT_STRETCH)
        grid.setColumnStretch(1, _OUTER_COL_RIGHT_STRETCH)
        grid.setRowStretch(0, _OUTER_ROW_TOP_STRETCH)
        grid.setRowStretch(1, _OUTER_ROW_BODY_STRETCH)
        return tab

    # ----- top row (browse | run | logo) -----------------------------

    def _analysis_build_top_row(self, parent: QWidget) -> QGridLayout:
        row = QGridLayout()
        row.setHorizontalSpacing(6)

        # VTop horizontal layout: browse, run button, logo.
        row.addLayout(
            self._build_browse_row_layout(
                button_name="analysis_pushButton_browse",
                line_edit_name="analysis_lineEdit_browse",
                line_edit_read_only=True,
                parent=parent,
            ),
            0, 0, 1, 1,
        )
        row.addLayout(self._analysis_build_run_cell(parent), 0, 1, 1, 1)
        row.addLayout(
            self._build_logo_view_layout(
                view_name="analysis_graphicsView_logo",
                parent=parent,
            ),
            0, 2, 1, 1,
        )

        row.setColumnStretch(0, _TOP_BROWSE_STRETCH)
        row.setColumnStretch(1, _TOP_RUN_STRETCH)
        row.setColumnStretch(2, _TOP_LOGO_STRETCH)
        return row

    def _analysis_build_run_cell(self, parent: QWidget) -> QGridLayout:
        grid = QGridLayout()
        grid.addItem(self._expanding_vertical_spacer(), 0, 0, 1, 1)
        grid.addItem(self._expanding_vertical_spacer(), 2, 0, 1, 1)
        grid.addItem(self._expanding_horizontal_spacer(), 1, 1, 1, 1)

        run = QPushButton("Analysis", parent)
        run.setFont(self._pixel_font(FS_LARGE))
        run.setStyleSheet(_RUN_BUTTON_QSS)
        run.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._register("analysis_pushButton_runAnalysis", run)
        grid.addWidget(run, 1, 0, 1, 1)

        grid.setColumnStretch(0, _RUN_COL_BUTTON_STRETCH)
        grid.setColumnStretch(1, _RUN_COL_FILL_STRETCH)
        grid.setRowStretch(0, _RUN_ROW_TOP_STRETCH)
        grid.setRowStretch(1, _RUN_ROW_MID_STRETCH)
        grid.setRowStretch(2, _RUN_ROW_BOTTOM_STRETCH)
        return grid

    # ----- left frame (settings + histogram + summary + exports) -----

    def _analysis_build_left_frame(self, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.Box)
        frame.setObjectName("analysis_left_frame")

        col = QVBoxLayout(frame)
        col.setContentsMargins(5, 5, 5, 5)
        col.setSpacing(6)

        # HLeft vertical layout: config, settings, histogram, summary, exports.
        col.addLayout(self._analysis_build_config_row(frame))
        col.addLayout(self._analysis_build_setting_grid_row(frame))
        col.addLayout(self._analysis_build_histogram_grid_row(frame))
        col.addLayout(self._analysis_build_summary_row(frame))
        col.addItem(self._expanding_vertical_spacer())
        col.addLayout(self._analysis_build_export_row(frame))

        col.setStretch(0, _LEFT_CONFIG_STRETCH)
        col.setStretch(1, _LEFT_SETTING_STRETCH)
        col.setStretch(2, _LEFT_HISTOGRAM_STRETCH)
        col.setStretch(3, _LEFT_SUMMARY_STRETCH)
        col.setStretch(4, _LEFT_GAP_STRETCH)
        col.setStretch(5, _LEFT_EXPORTS_STRETCH)
        return frame

    def _analysis_build_config_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        config_btn = QPushButton("Config", parent)
        config_btn.setFont(self._pixel_font(FS_SMALL))
        config_btn.setMouseTracking(False)
        config_btn.setAcceptDrops(False)
        self._register("analysis_pushButton_config", config_btn)
        row.addWidget(config_btn)
        row.addItem(self._expanding_horizontal_spacer())
        row.setStretch(0, 5)
        row.setStretch(1, 7)
        return row

    def _analysis_build_setting_grid_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        # Row 0: Setting header.
        grid.addWidget(self._make_label("Setting", bold=True, parent=parent), 0, 0, 1, 1)

        # Row 1: Lag label | lag_frame edit | "frame" unit.
        grid.addWidget(self._make_label("    Lag", parent=parent), 1, 0, 1, 1)
        lag_frame = self._make_param_line_edit(default="2", parent=parent)
        lag_frame.setMouseTracking(False)
        lag_frame.setAcceptDrops(False)
        self._register("analysis_lineEdit_lagFrame", lag_frame)
        grid.addWidget(lag_frame, 1, 1, 1, 1)
        grid.addWidget(self._make_label("frame", parent=parent), 1, 2, 1, 1)

        # Row 2: Lag label | lag_sec edit | "sec" unit.
        grid.addWidget(self._make_label("    Lag", parent=parent), 2, 0, 1, 1)
        lag_sec = self._make_param_line_edit(read_only=True, parent=parent)
        lag_sec.setEnabled(True)
        lag_sec.setMouseTracking(False)
        lag_sec.setAcceptDrops(False)
        self._register("analysis_lineEdit_lagSec", lag_sec)
        grid.addWidget(lag_sec, 2, 1, 1, 1)
        grid.addWidget(self._make_label("sec", parent=parent), 2, 2, 1, 1)

        # Row 3: R^2 threshold label | r2 edit.
        r2_label = self._make_label("", parent=parent, text_format=Qt.RichText)
        r2_label.setText(_R2_THRESHOLD_HTML)
        r2_label.setFont(self._pixel_font(FS_SMALL))
        grid.addWidget(r2_label, 3, 0, 1, 1)
        r2_le = self._make_param_line_edit(default="0", parent=parent)
        r2_le.setEnabled(True)
        r2_le.setMouseTracking(False)
        r2_le.setAcceptDrops(False)
        self._register("analysis_lineEdit_r2Threshold", r2_le)
        grid.addWidget(r2_le, 3, 1, 1, 1)

        # Row 4: Correction mode label | combo box | csv checkbox.
        grid.addWidget(
            self._make_label("    Correction mode",
                             alignment=Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter,
                             parent=parent),
            4, 0, 1, 1,
        )
        correction = QComboBox(parent)
        correction.setFont(self._pixel_font(FS_SMALL))
        correction.setMinimumContentsLength(8)
        self._register("analysis_comboBox_correctionMode", correction)
        grid.addWidget(correction, 4, 1, 1, 1)
        csv_box = QCheckBox("csv", parent)
        csv_box.setFont(self._pixel_font(FS_SMALL))
        self._register("analysis_checkBox_csv", csv_box)
        grid.addWidget(csv_box, 4, 2, 1, 1)

        # Row 5: Distribution mode label | combo box.
        grid.addWidget(self._make_label("    Distribution mode", parent=parent), 5, 0, 1, 1)
        distribution = QComboBox(parent)
        distribution.setFont(self._pixel_font(FS_SMALL))
        distribution.setMinimumContentsLength(8)
        self._register("analysis_comboBox_distributionMode", distribution)
        grid.addWidget(distribution, 5, 1, 1, 1)

        grid.setColumnStretch(0, _FORM_COL_LABEL_STRETCH)
        grid.setColumnStretch(1, _FORM_COL_FIELD_STRETCH)
        grid.setColumnStretch(2, _FORM_COL_UNIT_STRETCH)

        row.addLayout(grid)
        row.setStretch(0, 5)
        return row

    def _analysis_build_histogram_grid_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(0)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        # Row 0: Histogram header | replot button | Log checkbox.
        grid.addWidget(self._make_label("Histogram", bold=True, parent=parent), 0, 0, 1, 1)
        replot = QPushButton("plot", parent)
        replot.setFont(self._pixel_font(FS_SMALL))
        replot.setMouseTracking(False)
        replot.setAcceptDrops(False)
        replot.setLayoutDirection(Qt.LeftToRight)
        self._register("analysis_pushButton_replot", replot)
        grid.addWidget(replot, 0, 1, 1, 1)
        log_box = QCheckBox("Log", parent)
        log_box.setFont(self._pixel_font(FS_SMALL))
        self._register("analysis_checkBox_logarithmPlot", log_box)
        grid.addWidget(log_box, 0, 2, 1, 1)

        # Row 1: Size min label | min_d edit | "nm" unit.
        size_min_label = self._make_label("    Size min", parent=parent)
        size_min_label.setMouseTracking(True)
        size_min_label.setTextFormat(Qt.AutoText)
        grid.addWidget(size_min_label, 1, 0, 1, 1)
        min_d = self._make_param_line_edit(default="10", parent=parent)
        min_d.setMouseTracking(False)
        min_d.setAcceptDrops(False)
        self._register("analysis_lineEdit_minD", min_d)
        grid.addWidget(min_d, 1, 1, 1, 1)
        grid.addWidget(self._make_label("nm", parent=parent), 1, 2, 1, 1)

        # Row 2: Size max label | max_d edit | "nm" unit.
        grid.addWidget(self._make_label("    Size max", parent=parent), 2, 0, 1, 1)
        max_d = self._make_param_line_edit(default="1000", parent=parent)
        max_d.setMouseTracking(False)
        max_d.setAcceptDrops(False)
        self._register("analysis_lineEdit_maxD", max_d)
        grid.addWidget(max_d, 2, 1, 1, 1)
        grid.addWidget(self._make_label("nm", parent=parent), 2, 2, 1, 1)

        # Row 3: Size bin number label | bin_num edit.
        grid.addWidget(self._make_label("    Size bin number", parent=parent), 3, 0, 1, 1)
        bin_num = self._make_param_line_edit(default="100", parent=parent)
        bin_num.setMouseTracking(False)
        bin_num.setAcceptDrops(False)
        self._register("analysis_lineEdit_binNum", bin_num)
        grid.addWidget(bin_num, 3, 1, 1, 1)

        # Row 4: Count max label | max_c edit.
        grid.addWidget(self._make_label("    Count max", parent=parent), 4, 0, 1, 1)
        max_c = self._make_param_line_edit(default="200", parent=parent)
        max_c.setMouseTracking(False)
        max_c.setAcceptDrops(False)
        self._register("analysis_lineEdit_maxC", max_c)
        grid.addWidget(max_c, 4, 1, 1, 1)

        # Row 5: Count tick label | tick_c edit.
        grid.addWidget(self._make_label("    Count tick", parent=parent), 5, 0, 1, 1)
        tick_c = self._make_param_line_edit(default="20", parent=parent)
        tick_c.setMouseTracking(False)
        tick_c.setAcceptDrops(False)
        self._register("analysis_lineEdit_tickC", tick_c)
        grid.addWidget(tick_c, 5, 1, 1, 1)

        # Row 6: Special tick label | special_tick edit | "nm" unit.
        grid.addWidget(self._make_label("    Special tick", parent=parent), 6, 0, 1, 1)
        special_tick = self._make_param_line_edit(parent=parent)
        special_tick.setMouseTracking(False)
        special_tick.setAcceptDrops(False)
        self._register("analysis_lineEdit_specialTick", special_tick)
        grid.addWidget(special_tick, 6, 1, 1, 1)
        grid.addWidget(self._make_label("nm", parent=parent), 6, 2, 1, 1)

        grid.setColumnStretch(0, _FORM_COL_LABEL_STRETCH)
        grid.setColumnStretch(1, _FORM_COL_FIELD_STRETCH)
        grid.setColumnStretch(2, _FORM_COL_UNIT_STRETCH)

        row.addLayout(grid)
        row.setStretch(0, 5)
        return row

    def _analysis_build_summary_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(0)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)

        # "Summary" header.
        summary_label = self._make_label(
            "Summary",
            size_px=FS_MEDIUM,
            bold=True,
            alignment=Qt.AlignBottom | Qt.AlignLeading | Qt.AlignLeft,
            parent=parent,
        )
        grid.addWidget(summary_label, 0, 0, 1, 1)

        summary = QTextBrowser(parent)
        # 11px renders the appended setPlainText content at a compact size,
        # matching tracking_textBrowser_logs.
        summary.setFont(self._pixel_font(11))
        self._register("analysis_textBrowser_summary", summary)
        grid.addWidget(summary, 1, 0, 1, 1)

        row.addLayout(grid)
        row.setStretch(0, 5)
        return row

    def _analysis_build_export_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()

        export_size = QPushButton("Export size", parent)
        export_size.setMouseTracking(False)
        export_size.setAcceptDrops(False)
        export_size.setAutoFillBackground(False)
        export_size.setStyleSheet(_EXPORT_SIZE_BUTTON_QSS)
        self._register("analysis_pushButton_exportSize", export_size)
        row.addWidget(export_size)

        row.addItem(self._expanding_horizontal_spacer())

        export_report = QPushButton("Export report", parent)
        export_report.setFont(self._pixel_font(FS_SMALL))
        self._register("analysis_pushButton_exportReport", export_report)
        row.addWidget(export_report)

        row.setStretch(0, 2)
        row.setStretch(1, 1)
        row.setStretch(2, 2)
        return row

    # ----- right frame (nested results tab widget) -------------------

    def _analysis_build_right_frame(self, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName("analysis_results_frame")

        grid = QGridLayout(frame)
        grid.addWidget(self._analysis_build_results_tabwidget(frame), 0, 0, 1, 1)
        return frame

    def _analysis_build_results_tabwidget(self, parent: QWidget) -> QTabWidget:
        tabw = QTabWidget(parent)
        tabw.setFont(self._pixel_font(FS_SMALL))
        tabw.setStyleSheet("")
        tabw.setTabShape(QTabWidget.Rounded)
        tabw.setTabBarAutoHide(False)
        self._register("analysis_tabWidget_results", tabw)

        tabw.addTab(self._analysis_make_graphics_tab(
            tab_name="tab_Analysis_Drift",
            view_name="analysis_graphicsView_drift",
        ), "Drift")
        tabw.addTab(self._analysis_make_graphics_tab(
            tab_name="tab_Analysis_Brownian",
            view_name="analysis_graphicsView_brownian",
        ), "Brownian")
        tabw.addTab(self._analysis_make_graphics_tab(
            tab_name="tab_Analysis_MSD",
            view_name="analysis_graphicsView_MSD",
        ), "MSD")
        tabw.addTab(self._analysis_make_graphics_tab(
            tab_name="tab_Analysis_Size",
            view_name="analysis_graphicsView_size",
        ), "Size")
        tabw.addTab(self._analysis_make_report_tab(), "Report")

        tabw.setCurrentIndex(0)
        return tabw

    def _analysis_make_graphics_tab(self, *, tab_name: str, view_name: str) -> QWidget:
        tab = QWidget()
        tab.setObjectName(tab_name)
        tab.setEnabled(True)
        setattr(self, tab_name, tab)

        grid = QGridLayout(tab)
        view = QGraphicsView(tab)
        view.setFocusPolicy(Qt.TabFocus)
        self._register(view_name, view)
        grid.addWidget(view, 0, 0, 1, 1)
        return tab

    def _analysis_make_report_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tab_Report")
        self.tab_Report = tab

        grid = QGridLayout(tab)
        report = QWebEngineView(tab)
        self._register("analysis_textBrowser_report", report)
        grid.addWidget(report, 0, 0, 1, 1)
        return tab
