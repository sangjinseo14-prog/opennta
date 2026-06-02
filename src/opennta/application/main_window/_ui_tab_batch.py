"""Builds the Batch tab of OpenNtaMainWindow.

Layout:

    [ browse row | run-cell | logo ]                        (top row)
    [ tree + config + setting grid (left frame) | table view (right frame) ]
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QTableView,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from ._ui_constants import FS_LARGE, FS_SMALL

_OUTER_COL_LEFT_STRETCH = 4
_OUTER_COL_RIGHT_STRETCH = 8
_OUTER_ROW_TOP_STRETCH = 1
_OUTER_ROW_BODY_STRETCH = 10

# Top row column stretches (browse | run-cell | logo).
_TOP_BROWSE_STRETCH = 12
_TOP_RUN_STRETCH = 7
_TOP_LOGO_STRETCH = 7

# Run-cell stretches.
_RUN_COL_BUTTON_STRETCH = 2
_RUN_COL_FILL_STRETCH = 5
_RUN_ROW_TOP_STRETCH = 1
_RUN_ROW_MID_STRETCH = 2
_RUN_ROW_BOTTOM_STRETCH = 1

# Left-frame vertical stretches (tree, spacer, config-row, setting grid, spacer).
_LEFT_TREE_STRETCH = 30
_LEFT_GAP_TREE_STRETCH = 2
_LEFT_CONFIG_STRETCH = 2
_LEFT_SETTING_STRETCH = 18
_LEFT_GAP_BOTTOM_STRETCH = 4

# Setting form column stretches.
_FORM_COL_LABEL_STRETCH = 6
_FORM_COL_FIELD_STRETCH = 4
_FORM_COL_UNIT_STRETCH = 2

_RUN_BUTTON_QSS = "background-color: rgb(52, 145, 220);"
_R2_THRESHOLD_HTML = (
    "<html><head/><body><p>&nbsp;&nbsp;&nbsp;&nbsp;"
    "R<span style=\" vertical-align:super;\">2</span>"
    "&nbsp;Threshold&nbsp;</p>\n</body></html>"
)


class _BatchTabBuilder:
    """Mixin: ``_build_batch_tab()`` returns the populated tab body."""

    def _build_batch_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tab_batch")

        grid = QGridLayout(tab)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)

        # Row 0: browse | run | logo.
        # Row 1: tree & settings | results table.
        grid.addLayout(self._batch_build_top_row(tab), 0, 0, 1, 2)
        grid.addWidget(self._batch_build_left_frame(tab), 1, 0, 1, 1)
        grid.addWidget(self._batch_build_right_frame(tab), 1, 1, 1, 1)

        grid.setColumnStretch(0, _OUTER_COL_LEFT_STRETCH)
        grid.setColumnStretch(1, _OUTER_COL_RIGHT_STRETCH)
        grid.setRowStretch(0, _OUTER_ROW_TOP_STRETCH)
        grid.setRowStretch(1, _OUTER_ROW_BODY_STRETCH)
        return tab

    def _batch_build_top_row(self, parent: QWidget) -> QGridLayout:
        row = QGridLayout()
        row.setHorizontalSpacing(6)

        # VTop horizontal layout: browse, run button, logo.
        row.addLayout(
            self._build_browse_row_layout(
                button_name="batch_pushButton_browse",
                line_edit_name="batch_lineEdit_browse",
                line_edit_read_only=True,
                parent=parent,
            ),
            0, 0, 1, 1,
        )
        row.addLayout(self._batch_build_run_cell(parent), 0, 1, 1, 1)
        row.addLayout(
            self._build_logo_view_layout(
                view_name="batch_graphicsView_logo",
                transformation_anchor=True,
                parent=parent,
            ),
            0, 2, 1, 1,
        )

        row.setColumnStretch(0, _TOP_BROWSE_STRETCH)
        row.setColumnStretch(1, _TOP_RUN_STRETCH)
        row.setColumnStretch(2, _TOP_LOGO_STRETCH)
        return row

    def _batch_build_run_cell(self, parent: QWidget) -> QGridLayout:
        grid = QGridLayout()
        grid.addItem(self._expanding_vertical_spacer(width=5), 0, 0, 1, 1)
        grid.addItem(self._expanding_vertical_spacer(width=5), 2, 0, 1, 1)
        grid.addItem(self._expanding_horizontal_spacer(), 1, 1, 1, 1)

        run = QPushButton("Analysis", parent)
        run.setFont(self._pixel_font(FS_LARGE))
        run.setStyleSheet(_RUN_BUTTON_QSS)
        run.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._register("batch_pushButton_runAnalysis", run)
        grid.addWidget(run, 1, 0, 1, 1)

        grid.setColumnStretch(0, _RUN_COL_BUTTON_STRETCH)
        grid.setColumnStretch(1, _RUN_COL_FILL_STRETCH)
        grid.setRowStretch(0, _RUN_ROW_TOP_STRETCH)
        grid.setRowStretch(1, _RUN_ROW_MID_STRETCH)
        grid.setRowStretch(2, _RUN_ROW_BOTTOM_STRETCH)
        return grid

    def _batch_build_left_frame(self, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName("batch_left_frame")

        col = QVBoxLayout(frame)
        col.setContentsMargins(5, 5, 5, 5)
        col.setSpacing(6)

        # HLeft vertical layout: tree, config, settings.
        tree = QTreeWidget(frame)
        tree.headerItem().setText(0, "Browse result folders")
        self._register("batch_treeWidget_folders", tree)
        col.addWidget(tree)

        col.addItem(self._expanding_vertical_spacer())
        col.addLayout(self._batch_build_config_row(frame))
        col.addLayout(self._batch_build_setting_grid(frame))
        col.addItem(self._expanding_vertical_spacer())

        col.setStretch(0, _LEFT_TREE_STRETCH)
        col.setStretch(1, _LEFT_GAP_TREE_STRETCH)
        col.setStretch(2, _LEFT_CONFIG_STRETCH)
        col.setStretch(3, _LEFT_SETTING_STRETCH)
        col.setStretch(4, _LEFT_GAP_BOTTOM_STRETCH)
        return frame

    def _batch_build_config_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        config_btn = QPushButton("Config", parent)
        config_btn.setFont(self._pixel_font(FS_SMALL))
        self._register("batch_pushButton_config", config_btn)
        row.addWidget(config_btn)
        row.addItem(self._expanding_horizontal_spacer())
        row.setStretch(0, 5)
        row.setStretch(1, 7)
        return row

    def _batch_build_setting_grid(self, parent: QWidget) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        # Row 0: Setting header.
        grid.addWidget(self._make_label("Setting", parent=parent), 0, 0, 1, 1)

        # Row 1: Lag label | lag_frame edit | "frame" unit.
        grid.addWidget(self._make_label("    Lag", parent=parent), 1, 0, 1, 1)
        lag_frame = self._make_param_line_edit(default="2", parent=parent)
        self._register("batch_lineEdit_lagFrame", lag_frame)
        grid.addWidget(lag_frame, 1, 1, 1, 1)
        grid.addWidget(self._make_label("frame", parent=parent), 1, 2, 1, 1)

        # Row 2: Lag label | lag_sec edit | "sec" unit.
        grid.addWidget(self._make_label("    Lag", parent=parent), 2, 0, 1, 1)
        lag_sec = self._make_param_line_edit(read_only=True, parent=parent)
        self._register("batch_lineEdit_lagSec", lag_sec)
        grid.addWidget(lag_sec, 2, 1, 1, 1)
        grid.addWidget(self._make_label("sec", parent=parent), 2, 2, 1, 1)

        # Row 3: R^2 threshold label | r2 edit.
        r2_label = self._make_label("", parent=parent, text_format=Qt.RichText)
        r2_label.setText(_R2_THRESHOLD_HTML)
        r2_label.setFont(self._pixel_font(FS_SMALL))
        grid.addWidget(r2_label, 3, 0, 1, 1)
        r2_le = self._make_param_line_edit(default="0", parent=parent)
        self._register("batch_lineEdit_r2Threshold", r2_le)
        grid.addWidget(r2_le, 3, 1, 1, 1)

        # Row 4: Correction mode label | combo box.
        grid.addWidget(
            self._make_label("    Correction mode",
                             alignment=Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter,
                             parent=parent),
            4, 0, 1, 1,
        )
        correction = QComboBox(parent)
        correction.setEnabled(True)
        correction.setMinimumContentsLength(8)
        self._register("batch_comboBox_correctionMode", correction)
        grid.addWidget(correction, 4, 1, 1, 1)
        csv_box = QCheckBox("csv", parent)
        csv_box.setFont(self._pixel_font(FS_SMALL))
        self._register("batch_checkBox_csv", csv_box)
        grid.addWidget(csv_box, 4, 2, 1, 1)

        # Row 5: Distribution mode label | combo box.
        grid.addWidget(self._make_label("    Distribution mode", parent=parent), 5, 0, 1, 1)
        distribution = QComboBox(parent)
        distribution.setMinimumContentsLength(8)
        self._register("batch_comboBox_distributionMode", distribution)
        grid.addWidget(distribution, 5, 1, 1, 1)

        grid.setColumnStretch(0, _FORM_COL_LABEL_STRETCH)
        grid.setColumnStretch(1, _FORM_COL_FIELD_STRETCH)
        grid.setColumnStretch(2, _FORM_COL_UNIT_STRETCH)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)
        grid.setRowStretch(3, 1)
        grid.setRowStretch(4, 1)
        grid.setRowStretch(5, 1)
        return grid

    def _batch_build_right_frame(self, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName("batch_right_frame")

        grid = QGridLayout(frame)
        table = QTableView(frame)
        table.setProperty("showDropIndicator", True)
        self._register("batch_tableView_progress", table)
        grid.addWidget(table, 0, 0, 1, 1)
        return frame
