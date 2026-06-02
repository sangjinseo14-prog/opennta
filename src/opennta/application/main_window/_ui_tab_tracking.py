"""Builds the Tracking tab of OpenNtaMainWindow.

Layout:

    [ browse row | FIJI / spacer | logo ]   (top row, grid_top)
    [ left frame                 | logs frame ]   (content row)

Stretches and spacings mirror the legacy `.ui` source so the resulting widget
geometry is visually indistinguishable.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QTextBrowser,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from ._ui_constants import FS_LARGE

# Outer grid (logo row + content row).
_OUTER_ROW_TOP_STRETCH = 1
_OUTER_ROW_BODY_STRETCH = 10
_OUTER_COL_LEFT_STRETCH = 4
_OUTER_COL_RIGHT_STRETCH = 8

# Top row stretches (browse | FIJI cell | logo).
_TOP_BROWSE_STRETCH = 12
_TOP_FIJI_STRETCH = 7
_TOP_LOGO_STRETCH = 7

# FIJI sub-grid (button on the left, expanding column on the right).
_FIJI_COL_BUTTON_STRETCH = 1
_FIJI_COL_FILL_STRETCH = 5
_FIJI_ROW_TOP_STRETCH = 1
_FIJI_ROW_MID_STRETCH = 2
_FIJI_ROW_BOTTOM_STRETCH = 1

# Left column vertical stretches (tree, spacer, method group, spacer,
# browse-row, spacer, run-button).
_LEFT_TREE_STRETCH = 34
_LEFT_GAP1_STRETCH = 3
_LEFT_METHOD_STRETCH = 9
_LEFT_GAP2_STRETCH = 3
_LEFT_SAVE_STRETCH = 2
_LEFT_GAP3_STRETCH = 3
_LEFT_RUN_STRETCH = 2

# Inline stylesheet baked into the .ui for the red Run button.
_RUN_BUTTON_QSS = (
    "background-color: rgb(252, 40, 41);\n"
    "color: rgb(255, 255, 255);"
)


class _TrackingTabBuilder:
    """Mixin: ``_build_tracking_tab()`` returns the populated tab body."""

    def _build_tracking_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("tab_tracking")

        grid = QGridLayout(tab)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)

        # Row 0: browse | FIJI | logo.
        # Row 1: tree & method & save & run | logs.
        grid.addLayout(self._tracking_build_top_row(tab), 0, 0, 1, 2)
        grid.addWidget(self._tracking_build_left_frame(tab), 1, 0, 1, 1)
        grid.addWidget(self._tracking_build_logs_frame(tab), 1, 1, 1, 1)

        grid.setColumnStretch(0, _OUTER_COL_LEFT_STRETCH)
        grid.setColumnStretch(1, _OUTER_COL_RIGHT_STRETCH)
        grid.setRowStretch(0, _OUTER_ROW_TOP_STRETCH)
        grid.setRowStretch(1, _OUTER_ROW_BODY_STRETCH)
        return tab

    # ----- top row (browse | FIJI | logo) ----------------------------

    def _tracking_build_top_row(self, parent: QWidget) -> QGridLayout:
        row = QGridLayout()
        row.setHorizontalSpacing(6)

        # VTop horizontal layout: browse, FIJI launcher, logo.
        row.addLayout(
            self._build_browse_row_layout(
                button_name="tracking_pushButton_browse",
                line_edit_name="tracking_lineEdit_browse",
                line_edit_read_only=True,
                parent=parent,
            ),
            0, 0, 1, 1,
        )
        row.addLayout(self._tracking_build_fiji_cell(parent), 0, 1, 1, 1)
        row.addLayout(
            self._build_logo_view_layout(
                view_name="tracking_graphicsView_logo",
                transformation_anchor=True,
                parent=parent,
            ),
            0, 2, 1, 1,
        )

        row.setColumnStretch(0, _TOP_BROWSE_STRETCH)
        row.setColumnStretch(1, _TOP_FIJI_STRETCH)
        row.setColumnStretch(2, _TOP_LOGO_STRETCH)
        return row

    def _tracking_build_fiji_cell(self, parent: QWidget) -> QGridLayout:
        grid = QGridLayout()
        grid.addItem(self._expanding_vertical_spacer(), 0, 0, 1, 1)
        grid.addItem(self._expanding_vertical_spacer(), 2, 0, 1, 1)
        grid.addItem(self._expanding_horizontal_spacer(), 1, 1, 1, 1)

        fiji = QPushButton("Find Fiji", parent)
        fiji.setFont(self._pixel_font(FS_LARGE))
        fiji.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._register("tracking_pushButton_FIJI", fiji)
        grid.addWidget(fiji, 1, 0, 1, 1)

        grid.setColumnStretch(0, _FIJI_COL_BUTTON_STRETCH)
        grid.setColumnStretch(1, _FIJI_COL_FILL_STRETCH)
        grid.setRowStretch(0, _FIJI_ROW_TOP_STRETCH)
        grid.setRowStretch(1, _FIJI_ROW_MID_STRETCH)
        grid.setRowStretch(2, _FIJI_ROW_BOTTOM_STRETCH)
        return grid

    # ----- left column (tree + method group + save row + run button) -

    def _tracking_build_left_frame(self, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.Box)
        frame.setObjectName("tracking_left_frame")

        col = QVBoxLayout(frame)
        col.setContentsMargins(5, 5, 5, 5)
        col.setSpacing(5)

        # HLeft vertical layout: tree, method, save, run button.
        tree = QTreeWidget(frame)
        tree.headerItem().setText(0, "Browse image folders")
        self._register("tracking_treeWidget_folders", tree)
        col.addWidget(tree)

        col.addItem(QSpacerItem(20, 18, QSizePolicy.Minimum, QSizePolicy.Fixed))
        col.addWidget(self._tracking_build_method_group(frame))
        col.addItem(self._expanding_vertical_spacer())
        col.addLayout(self._tracking_build_save_row(frame))
        col.addItem(self._expanding_vertical_spacer())
        col.addWidget(self._tracking_build_run_button(frame))

        col.setStretch(0, _LEFT_TREE_STRETCH)
        col.setStretch(1, _LEFT_GAP1_STRETCH)
        col.setStretch(2, _LEFT_METHOD_STRETCH)
        col.setStretch(3, _LEFT_GAP2_STRETCH)
        col.setStretch(4, _LEFT_SAVE_STRETCH)
        col.setStretch(5, _LEFT_GAP3_STRETCH)
        col.setStretch(6, _LEFT_RUN_STRETCH)
        return frame

    def _tracking_build_method_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Tracking method", parent)
        self._register("groupBox_Track_Method", box)

        outer = QVBoxLayout(box)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Radio row.
        radio_row = QHBoxLayout()
        combined = QRadioButton("Combined", box)
        combined.setChecked(True)
        self._register("tracking_radioButton_combined", combined)
        radio_row.addWidget(combined)

        split = QRadioButton("Split", box)
        self._register("tracking_radioButton_split", split)
        radio_row.addWidget(split)
        radio_row.addItem(self._expanding_horizontal_spacer())

        radio_row.setStretch(0, 3)
        radio_row.setStretch(1, 3)
        radio_row.setStretch(2, 5)
        outer.addLayout(radio_row)

        # Method | Detector | Linker combos.
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)

        method_combo = QComboBox(box)
        method_combo.setMinimumContentsLength(8)
        self._register("tracking_comboBox_method", method_combo)
        grid.addWidget(self._make_label("Method", parent=box), 0, 0, 1, 1)
        grid.addWidget(method_combo, 0, 1, 1, 1)

        detector_combo = QComboBox(box)
        detector_combo.setMinimumContentsLength(8)
        self._register("tracking_comboBox_detector", detector_combo)
        grid.addWidget(self._make_label("Detector", parent=box), 1, 0, 1, 1)
        grid.addWidget(detector_combo, 1, 1, 1, 1)

        linker_combo = QComboBox(box)
        linker_combo.setMinimumContentsLength(8)
        self._register("tracking_comboBox_linker", linker_combo)
        grid.addWidget(self._make_label("Linker", parent=box), 2, 0, 1, 1)
        grid.addWidget(linker_combo, 2, 1, 1, 1)

        outer.addLayout(grid)
        return box

    def _tracking_build_save_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSizeConstraint(QHBoxLayout.SetDefaultConstraint)

        save_btn = QPushButton("Save to...", parent)
        save_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._register("tracking_pushButton_save", save_btn)
        row.addWidget(save_btn)

        save_le = QLineEdit(parent)
        save_le.setText("")
        save_le.setReadOnly(False)
        self._register("tracking_lineEdit_save", save_le)
        row.addWidget(save_le)

        row.setStretch(0, 1)
        row.setStretch(1, 4)
        return row

    def _tracking_build_run_button(self, parent: QWidget) -> QPushButton:
        btn = QPushButton("Run tracking", parent)
        btn.setEnabled(True)
        btn.setStyleSheet(_RUN_BUTTON_QSS)
        self._register("tracking_pushButton_runTracking", btn)
        return btn

    # ----- right column (logs frame) ---------------------------------

    def _tracking_build_logs_frame(self, parent: QWidget) -> QFrame:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setObjectName("tracking_logs_frame")

        grid = QGridLayout(frame)
        logs = QTextBrowser(frame)
        # 11px renders log output at a compact, legible size; controls
        # tab_tracking/ui.py's append(message) plain-text rendering.
        logs.setFont(self._pixel_font(11))
        self._register("tracking_textBrowser_logs", logs)
        grid.addWidget(logs, 0, 0, 1, 1)
        return frame
