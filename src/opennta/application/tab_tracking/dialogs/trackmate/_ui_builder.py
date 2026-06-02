from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from opennta.tracking.trackmate.fitting_models import MODEL_CLASSES


class _UIBuilderMixin:

    _WINDOW_TITLE = "Configure TrackMate"
    _WINDOW_INIT_W = 1240
    _WINDOW_INIT_H = 760
    _LEFT_W = 340
    _FIELD_FIXED_W = 113
    _COMBO_POPUP_W = 226  # 2x field width so the full sample/model name is readable
    _BUTTON_HEIGHT = 22
    _PRIMARY_BUTTON_HEIGHT = 26
    _ACCENT = "#1c74ff"
    _ACCENT_MUTED = "#3b68c7"

    _LOG_VIEW_MAX_HEIGHT = 140

    _LABEL_COL_STRETCH = 6
    _FIELD_COL_STRETCH = 4

    def _build_ui(self) -> None:
        self.setWindowTitle(self._WINDOW_TITLE)
        self.setModal(True)
        self.resize(self._WINDOW_INIT_W, self._WINDOW_INIT_H)

        left = self._build_left_column()
        right = self._build_right_column()
        separator = self._build_vertical_separator()

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(left, 0)
        outer.addWidget(separator, 0)
        outer.addWidget(right, 1)

        self._apply_styles()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QGroupBox {
                margin-top: 16px;
                padding-top: 10px;
                font-weight: normal;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0 4px;
                font-weight: normal;
            }
            QLabel, QCheckBox, QRadioButton {
                font-weight: normal;
            }
            """
        )

    def _build_left_column(self) -> QWidget:
        left = QWidget(self)
        left.setFixedWidth(self._LEFT_W)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        left_layout.addWidget(self._build_sample_group(left))
        left_layout.addWidget(self._build_normalization_group(left))
        left_layout.addWidget(self._build_detection_group(left))
        left_layout.addWidget(self._build_tracker_group(left))
        left_layout.addWidget(self._build_preview_button(left))
        left_layout.addStretch(1)
        left_layout.addLayout(self._build_action_row(left))
        return left

    def _build_right_column(self) -> QWidget:
        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_image = self.figure.add_subplot(1, 2, 1)
        self.ax_hist = self.figure.add_subplot(1, 2, 2)

        self.check_show_overlay = QCheckBox("Show spot overlay", right)
        self.check_show_overlay.setChecked(True)
        self.check_show_overlay.toggled.connect(self._on_display_toggle)
        self.check_show_smoothed = QCheckBox("Smoothed histogram (±5 neighbors)", right)
        self.check_show_smoothed.setChecked(True)
        self.check_show_smoothed.toggled.connect(self._on_display_toggle)
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(12)
        toggle_row.addWidget(self.check_show_overlay)
        toggle_row.addWidget(self.check_show_smoothed)
        toggle_row.addStretch(1)

        self.status_label = QLabel("", right)
        self.status_label.setWordWrap(True)

        self.log_view = QTextBrowser(right)
        self.log_view.setMaximumHeight(self._LOG_VIEW_MAX_HEIGHT)

        right_layout.addWidget(self.canvas, 1)
        right_layout.addLayout(toggle_row)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.log_view)
        return right

    def _build_vertical_separator(self) -> QFrame:
        separator = QFrame(self)
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: rgba(255,255,255,0.08);")
        return separator

    def _make_grid_layout(self, parent: QWidget) -> QGridLayout:
        grid = QGridLayout(parent)
        grid.setContentsMargins(10, 14, 10, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, self._LABEL_COL_STRETCH)
        grid.setColumnStretch(1, self._FIELD_COL_STRETCH)
        return grid

    def _add_grid_row(self, grid: QGridLayout, label_text: str, widget: QWidget) -> None:
        row = grid.rowCount()
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(label, row, 0)
        grid.addWidget(widget, row, 1, Qt.AlignRight | Qt.AlignVCenter)

    def _build_sample_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Preview sample", parent)
        grid = self._make_grid_layout(box)

        self.combo_sample = QComboBox(box)
        for item in self._folder_items:
            self.combo_sample.addItem(self._format_sample_key(item))
        self._apply_uniform_combo_width(self.combo_sample)

        self.spin_frame_index = QSpinBox(box)
        self.spin_frame_index.setRange(0, 99_999)
        self.spin_frame_index.setValue(0)
        self.spin_frame_index.setFixedWidth(self._FIELD_FIXED_W)

        self._add_grid_row(grid, "Sample", self.combo_sample)
        self._add_grid_row(grid, "Frame (0-based)", self.spin_frame_index)
        return box

    def _build_normalization_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Normalization", parent)
        grid = self._make_grid_layout(box)

        self.spin_normalization_percentile = QDoubleSpinBox()
        self.spin_normalization_percentile.setRange(0.0, 100.0)
        self.spin_normalization_percentile.setDecimals(1)
        self.spin_normalization_percentile.setSingleStep(0.1)
        self.spin_normalization_percentile.setFixedWidth(self._FIELD_FIXED_W)

        self._add_grid_row(grid, "K (percentile)", self.spin_normalization_percentile)
        return box

    def _build_detection_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("LoG detector", parent)
        grid = self._make_grid_layout(box)

        self.spin_particle_radius = QDoubleSpinBox()
        self.spin_particle_radius.setRange(0.1, 100.0)
        self.spin_particle_radius.setDecimals(2)
        self.spin_particle_radius.setSingleStep(0.5)

        self.spin_fit_fraction = QDoubleSpinBox()
        self.spin_fit_fraction.setRange(0.01, 1.0)
        self.spin_fit_fraction.setDecimals(3)
        self.spin_fit_fraction.setSingleStep(0.01)

        self.spin_significance_alpha = QDoubleSpinBox()
        self.spin_significance_alpha.setRange(0.001, 1.0)
        self.spin_significance_alpha.setDecimals(3)
        self.spin_significance_alpha.setSingleStep(0.001)

        self.combo_quality_model = QComboBox(box)
        for name in MODEL_CLASSES.keys():
            self.combo_quality_model.addItem(name)
        self._apply_uniform_combo_width(self.combo_quality_model)

        for spin in (
            self.spin_particle_radius,
            self.spin_fit_fraction,
            self.spin_significance_alpha,
        ):
            spin.setFixedWidth(self._FIELD_FIXED_W)

        self._add_grid_row(grid, "Particle size", self.spin_particle_radius)
        self._add_grid_row(grid, "FRAC", self.spin_fit_fraction)
        self._add_grid_row(grid, "ALPHA", self.spin_significance_alpha)
        self._add_grid_row(grid, "Quality fit model", self.combo_quality_model)
        return box

    def _build_tracker_group(self, parent: QWidget) -> QGroupBox:
        box = QGroupBox("Linker", parent)
        grid = self._make_grid_layout(box)

        self.spin_linking_max_distance = QSpinBox()
        self.spin_linking_max_distance.setRange(0, 10_000)
        self.spin_gap_closing_max_distance = QSpinBox()
        self.spin_gap_closing_max_distance.setRange(0, 10_000)
        self.spin_max_frame_gap = QSpinBox()
        self.spin_max_frame_gap.setRange(0, 10_000)
        self.spin_min_track_frames = QSpinBox()
        self.spin_min_track_frames.setRange(0, 10_000_000)
        self.spin_border_margin_pixels = QSpinBox()
        self.spin_border_margin_pixels.setRange(0, 10_000)

        for spin in (
            self.spin_linking_max_distance,
            self.spin_gap_closing_max_distance,
            self.spin_max_frame_gap,
            self.spin_min_track_frames,
            self.spin_border_margin_pixels,
        ):
            spin.setFixedWidth(self._FIELD_FIXED_W)

        self._add_grid_row(grid, "Linking max distance", self.spin_linking_max_distance)
        self._add_grid_row(grid, "Gap closing max distance", self.spin_gap_closing_max_distance)
        self._add_grid_row(grid, "Max frame gap", self.spin_max_frame_gap)
        self._add_grid_row(grid, "Min track frames", self.spin_min_track_frames)
        self._add_grid_row(grid, "Border margin", self.spin_border_margin_pixels)
        return box

    def _build_preview_button(self, parent: QWidget) -> QPushButton:
        self.button_preview = self._make_action_button(
            "Preview using sample",
            self._on_preview,
            accent=self._ACCENT_MUTED,
            parent=parent,
        )
        return self.button_preview

    def _build_action_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.button_run = self._make_action_button(
            "Run tracking",
            self.accept,
            accent=self._ACCENT,
            primary=True,
            parent=parent,
        )
        self.button_run.setDefault(True)
        self.button_cancel = self._make_action_button(
            "Cancel",
            self.reject,
            parent=parent,
        )
        self.button_cancel.setFixedHeight(self._PRIMARY_BUTTON_HEIGHT)
        row.addWidget(self.button_run, 2)
        row.addWidget(self.button_cancel, 1)
        return row

    def _make_action_button(
        self,
        text: str,
        on_click,
        *,
        accent: str | None = None,
        primary: bool = False,
        parent: QWidget | None = None,
    ) -> QPushButton:
        button = QPushButton(text, parent)
        button.setFixedHeight(self._PRIMARY_BUTTON_HEIGHT if primary else self._BUTTON_HEIGHT)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(on_click)
        if accent is not None:
            lighter = self._shift_hex(accent, 0.12, lighten=True)
            darker = self._shift_hex(accent, 0.12, lighten=False)
            button.setStyleSheet(
                f"QPushButton {{ "
                f"background-color: {accent}; color: white; border: none; "
                f"border-radius: 4px; padding: 0 14px; font-weight: normal; "
                f"}} "
                f"QPushButton:hover {{ background-color: {lighter}; }} "
                f"QPushButton:pressed {{ background-color: {darker}; }} "
                f"QPushButton:disabled {{ background-color: #444; color: #888; }}"
            )
        else:
            button.setStyleSheet(
                "QPushButton { font-weight: normal; padding: 0 14px; border-radius: 4px; }"
            )
        return button

    def _apply_uniform_combo_width(self, combo: QComboBox) -> None:
        combo.setFixedWidth(self._FIELD_FIXED_W)
        combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        combo.view().setMinimumWidth(self._COMBO_POPUP_W)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    @classmethod
    def _shift_hex(cls, hex_color: str, amount: float, *, lighten: bool) -> str:
        r, g, b = cls._hex_to_rgb(hex_color)
        if lighten:
            r = min(255, int(r + (255 - r) * amount))
            g = min(255, int(g + (255 - g) * amount))
            b = min(255, int(b + (255 - b) * amount))
        else:
            r = max(0, int(r * (1.0 - amount)))
            g = max(0, int(g * (1.0 - amount)))
            b = max(0, int(b * (1.0 - amount)))
        return f"#{r:02x}{g:02x}{b:02x}"
