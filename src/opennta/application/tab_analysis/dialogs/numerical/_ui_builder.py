from __future__ import annotations

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _plot_style as _ps
from ._kernel_preview import _KernelPreview
from ._plot_style import style_figure


class _UIBuilderMixin:

    _WINDOW_TITLE = "Configure Numerical Drift Field"
    _WINDOW_INIT_W = 1500
    _WINDOW_INIT_H = 750
    _WINDOW_MIN_H = 650

    _LEFT_MIN_W = 300
    _LEFT_MAX_W = 350
    _RIGHT_MIN_W = 1050
    _RIGHT_MAX_W = 1200

    _FIELD_HEIGHT = 28
    _BUTTON_HEIGHT = 20
    _PRIMARY_BUTTON_HEIGHT = 20

    _INTERP_DEFAULT_NODES = 128

    _ACCENT = "#1c74ff"
    _ACCENT_MUTED = "#3b68c7"

    _VECTOR_COLOR = _ps.VECTOR_COLOR

    def _build_ui(self) -> None:
        self.setWindowTitle(self._WINDOW_TITLE)
        self.setModal(True)
        self.resize(self._WINDOW_INIT_W, self._WINDOW_INIT_H)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        root.addWidget(self._build_left_panel(), stretch=0)
        root.addWidget(self._build_right_panel(), stretch=1)

        self._apply_styles()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-size: 11px;
            }
            QGroupBox {
                margin-top: 14px;
                padding-top: 8px;
                font-size: 13px;
                font-weight: normal;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0 4px;
                font-weight: normal;
            }
            QLabel {
                font-size: 11px;
            }
            QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton {
                font-size: 13px;
            }
            """
        )

    def _build_left_panel(self) -> QWidget:
        wrap = QWidget()
        wrap.setMinimumWidth(self._LEFT_MIN_W)
        wrap.setMaximumWidth(self._LEFT_MAX_W)

        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)

        params_box, _ = self._build_field_group()
        interp_box = self._build_interp_group()
        smooth_box = self._build_smoothing_group()

        col.addWidget(params_box)
        col.addWidget(smooth_box)
        col.addWidget(self._build_vectors_button())
        col.addWidget(interp_box)
        col.addStretch(1)

        col.addWidget(self._build_export_checkbox())
        col.addWidget(self._make_separator())
        col.addLayout(self._build_bottom_actions())
        return wrap

    def _build_right_panel(self) -> QWidget:
        wrap = QWidget()
        wrap.setMinimumWidth(self._RIGHT_MIN_W)
        wrap.setMaximumWidth(self._RIGHT_MAX_W)

        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self.fig = style_figure(Figure(figsize=(13.5, 8.0)))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(self._WINDOW_MIN_H)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)

        col.addWidget(self.toolbar)
        col.addWidget(self.canvas, stretch=1)
        return wrap

    def _build_field_group(self) -> tuple[QGroupBox, QFormLayout]:
        box = QGroupBox("Field parameters")

        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(10, 12, 10, 10)
        vbox.setSpacing(8)

        form_wrap = QWidget()
        form = QFormLayout(form_wrap)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.sb_nwin = self._make_int_spin(2, 200, 10)
        self.sb_xmin = self._make_double(-1e6, 1e6, 0.0)
        self.sb_xmax = self._make_double(-1e6, 1e6, 2304.0)
        self.sb_ymin = self._make_double(-1e6, 1e6, 0.0)
        self.sb_ymax = self._make_double(-1e6, 1e6, 2304.0)
        self.sb_k = self._make_double(0.1, 50.0, 4.0, step=0.5)

        form.addRow("n_windows", self.sb_nwin)
        form.addRow("x_min, x_max", self._horizontal_pair_widget(self.sb_xmin, self.sb_xmax))
        form.addRow("y_min, y_max", self._horizontal_pair_widget(self.sb_ymin, self.sb_ymax))
        form.addRow("outlier_k", self.sb_k)

        for sb in (self.sb_nwin, self.sb_xmin, self.sb_xmax, self.sb_ymin, self.sb_ymax):
            sb.valueChanged.connect(self._on_grid_param_changed)

        vbox.addWidget(form_wrap)
        vbox.addWidget(
            self._make_button("Compute field", self._compute_field, accent=self._ACCENT_MUTED)
        )
        vbox.addWidget(self._build_uv_stats_label())

        return box, form

    # Order shown left to right; max-min is the range (max - min).
    _UV_STATS_COLUMNS = ("max", "min", "max-min", "mean", "sd")

    # Fixed column widths (label + 5 data columns = 100%) so the empty dash
    # table and the populated one share one layout; values then drop into the
    # same columns instead of resizing them and sliding the header text.
    _UV_STATS_LABEL_COL_WIDTH = "15%"
    _UV_STATS_DATA_COL_WIDTH = "17%"

    # u, v are the x/y velocity components; subscripts match the plot labels.
    _UV_STATS_LABEL_U = "u<sub>x</sub>"
    _UV_STATS_LABEL_V = "u<sub>y</sub>"

    # Shown in the title in every state (placeholder and populated) so the unit
    # is always visible and the title length never changes when values arrive.
    _UV_STATS_UNIT = "µm/s"

    def _build_uv_stats_label(self) -> QLabel:
        lbl = QLabel()
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_uv_stats = lbl
        # Render the empty table now so the panel reserves the stats space up
        # front; "Compute field" then fills the cells without shifting layout.
        self._clear_uv_stats()
        return lbl

    def _clear_uv_stats(self) -> None:
        # Same table as the populated state -- same title (with unit) and same
        # row count -- but with placeholder cells, so the reserved height
        # matches and nothing jumps once values arrive.
        self.lbl_uv_stats.setText(
            self._format_uv_stats_html(None, None, self._UV_STATS_UNIT)
        )

    def _reserve_uv_stats_height(self) -> None:
        # Lock the stats box to the height of the placeholder table so that
        # "Compute field" only swaps dashes for values and never grows the box,
        # which would push the controls below it down. Measured at the panel's
        # minimum width (where text wraps the most) so the reservation still
        # holds when the window is dragged narrow. Call once after fonts are
        # set, while the placeholder is showing.
        height = self.lbl_uv_stats.heightForWidth(self._LEFT_MIN_W)
        if height > 0:
            self.lbl_uv_stats.setFixedHeight(height)

    def _set_uv_stats(self, u_stats, v_stats, unit: str = "") -> None:
        self.lbl_uv_stats.setText(self._format_uv_stats_html(u_stats, v_stats, unit))

    def _format_uv_stats_html(self, u_stats, v_stats, unit: str) -> str:
        title = f"{self._UV_STATS_LABEL_U}, {self._UV_STATS_LABEL_V} statistics"
        if unit:
            title += f" ({unit})"
        # font-weight:normal drops the default bold <th> face so the header
        # row matches the un-bolded row labels and title.
        header_cells = "".join(
            f"<th align='right' width='{self._UV_STATS_DATA_COL_WIDTH}' "
            f"style='font-weight:normal;'>{col}</th>"
            for col in self._UV_STATS_COLUMNS
        )
        header = (
            f"<tr><th align='left' width='{self._UV_STATS_LABEL_COL_WIDTH}' "
            f"style='font-weight:normal;'>&nbsp;</th>{header_cells}</tr>"
        )
        body = (
            self._uv_stats_row(self._UV_STATS_LABEL_U, u_stats)
            + self._uv_stats_row(self._UV_STATS_LABEL_V, v_stats)
        )
        # 11px matches the other left-panel labels; family is inherited.
        return (
            f"<div style='color:{_ps.FG_MUTED}; font-size:11px;'>{title}</div>"
            "<table width='100%' cellspacing='0' cellpadding='2' "
            "style='font-size:11px;'>"
            f"{header}{body}</table>"
        )

    @classmethod
    def _uv_stats_row(cls, name: str, stats) -> str:
        label = f"<td align='left' width='{cls._UV_STATS_LABEL_COL_WIDTH}'>{name}</td>"
        width = cls._UV_STATS_DATA_COL_WIDTH
        if stats is None:
            # One right-aligned dash per column (not a single centered span) so
            # the placeholder reserves the same width and height as the values
            # that later replace it. Rendered transparent so the table reads as
            # blank cells until "Compute field" fills them in.
            cells = "".join(
                f"<td align='right' width='{width}' style='color:transparent;'>"
                f"&mdash;</td>"
                for _ in cls._UV_STATS_COLUMNS
            )
            return f"<tr>{label}{cells}</tr>"
        values = (stats.max, stats.min, stats.max - stats.min, stats.mean, stats.std)
        cells = "".join(
            f"<td align='right' width='{width}'>{v:.4g}</td>" for v in values
        )
        return f"<tr>{label}{cells}</tr>"

    def _build_interp_group(self) -> QGroupBox:
        self._last_interp_nodes = self._INTERP_DEFAULT_NODES

        box = QGroupBox("Interpolation")
        box.setCheckable(True)
        box.setChecked(True)
        box.setCursor(Qt.PointingHandCursor)
        self.chk_interp = box

        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(10, 12, 10, 10)
        vbox.setSpacing(6)

        form_wrap = QWidget()
        form = QFormLayout(form_wrap)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.le_nodes = QLineEdit()
        self.le_nodes.setValidator(QIntValidator(2, 8192, self.le_nodes))
        self.le_nodes.setFixedHeight(self._FIELD_HEIGHT)
        self.le_nodes.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.le_nodes.setText(str(self._last_interp_nodes))

        form.addRow("nodes", self.le_nodes)
        vbox.addWidget(form_wrap)

        box.toggled.connect(self._on_interp_toggled)
        self.le_nodes.textEdited.connect(self._on_nodes_edited)

        return box

    def _on_grid_param_changed(self, *_args) -> None:
        self._sync_node_default()
        self._redraw_tracks_only()

    def _on_interp_toggled(self, checked: bool) -> None:
        # Checked: user picks the node resolution (seeded with the last value).
        # Unchecked: the field is disabled and tracks n_windows (the window
        # centers), so 10 windows shows 10 nodes.
        self.le_nodes.setEnabled(checked)
        if checked:
            self.le_nodes.setText(str(self._last_interp_nodes))
        else:
            self._sync_node_default()

    def _on_nodes_edited(self, text: str) -> None:
        try:
            self._last_interp_nodes = max(int(text), 2)
        except ValueError:
            pass

    def _sync_node_default(self) -> None:
        if not self.chk_interp.isChecked():
            self.le_nodes.setText(str(self.sb_nwin.value()))

    def _build_smoothing_group(self) -> QGroupBox:
        box = QGroupBox("CI-weighted smoothing")

        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(10, 12, 10, 10)
        vbox.setSpacing(8)

        form_wrap = QWidget()
        grid = QGridLayout(form_wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)

        self.sb_mincount = self._make_int_spin(1, 500, 1)
        self.sb_niter = self._make_int_spin(1, 50, 1)
        self.sb_ksize = self._make_int_spin(1, 15, 5, step=2)
        self.sb_sigma = self._make_double(0.01, 20.0, 1.0, step=0.05, decimals=2)

        def _add(row: int, col: int, text: str, widget: QWidget) -> None:
            label = QLabel(text)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(label, row, col * 2)
            grid.addWidget(widget, row, col * 2 + 1)

        _add(0, 0, "min_count", self.sb_mincount)
        _add(0, 1, "iterations", self.sb_niter)
        _add(1, 0, "kernel size (odd)", self.sb_ksize)
        _add(1, 1, "sigma (σ)", self.sb_sigma)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        vbox.addWidget(form_wrap)

        eqn_html = (
            "<html><div style=\""
            "text-align:center;"
            "font-family:'Times New Roman','Cambria Math','STIX Two Math',serif;"
            "font-size:13px;"
            "line-height:1.2;"
            "color:#f3f6fb;"
            "\">"
            "<span style=\"font-style:italic;\">w</span>(&#916;j, &#916;i)"
            " = <span style=\"font-style:italic;\">G</span>(&#916;j, &#916;i)"
            " &middot; 1 / (CI<sub>95</sub><sup>2</sup> + &#949;)"
            "<br>"
            "<span style=\"font-style:italic;\">G</span>(&#916;j, &#916;i)"
            " = exp&nbsp;[ - (&#916;j<sup>2</sup> + &#916;i<sup>2</sup>) / (2&#963;<sup>2</sup>) ]"
            "</div></html>"
        )

        self.kernel_preview = _KernelPreview()
        self.kernel_preview.setToolTip(eqn_html)
        self.kernel_preview.setStyleSheet(
            "QToolTip {"
            "  color: #f3f6fb;"
            "  background-color: #12161e;"
            "  border: 1px solid #6b7e9c;"
            "  padding: 6px 8px;"
            "}"
        )
        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.addStretch(1)
        preview_row.addWidget(self.kernel_preview)
        preview_row.addStretch(1)
        vbox.addLayout(preview_row)

        vbox.addWidget(
            self._make_button("Apply smoothing", self._apply_smoothing, accent=self._ACCENT_MUTED)
        )

        self.sb_ksize.valueChanged.connect(self._refresh_kernel_preview)
        self.sb_sigma.valueChanged.connect(self._refresh_kernel_preview)
        self._refresh_kernel_preview()

        return box

    def _refresh_kernel_preview(self) -> None:
        self.kernel_preview.set_kernel(self.sb_ksize.value(), self.sb_sigma.value())

    def _build_vectors_button(self) -> QPushButton:
        self.btn_vectors = QPushButton("Show field overlay")
        self.btn_vectors.setCheckable(True)
        self.btn_vectors.setCursor(Qt.PointingHandCursor)
        self.btn_vectors.setFixedHeight(self._BUTTON_HEIGHT)
        self.btn_vectors.setStyleSheet(
            f"QPushButton {{ background-color: {self._ACCENT_MUTED}; color: white; "
            f"border: none; border-radius: 4px; padding: 0 14px; }} "
            f"QPushButton:checked {{ background-color: {self._VECTOR_COLOR}; }} "
            f"QPushButton:disabled {{ background-color: #444; color: #888; }}"
        )
        self.btn_vectors.toggled.connect(self._toggle_vectors)
        return self.btn_vectors

    def _build_export_checkbox(self) -> QWidget:
        self.chk_export_csv = QCheckBox("Export velocity fields as .csv")
        self.chk_export_csv.setCursor(Qt.PointingHandCursor)
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.chk_export_csv)
        row.addStretch(1)
        return wrap

    def _build_bottom_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.btn_accept = self._make_button(
            "Correct using field",
            self._accept_field,
            accent=self._ACCENT,
            primary=True,
        )

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(self._PRIMARY_BUTTON_HEIGHT)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.reject)

        row.addWidget(self.btn_accept, stretch=2)
        row.addWidget(self.btn_cancel, stretch=1)
        return row

    def _make_double(self, lo, hi, val, step=1.0, decimals=1) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(decimals)
        sb.setSingleStep(step)
        sb.setValue(val)
        sb.setFixedHeight(self._FIELD_HEIGHT)
        sb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sb.setMinimumWidth(60)
        sb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return sb

    def _make_int_spin(self, lo, hi, val, step=1) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setValue(val)
        sb.setFixedHeight(self._FIELD_HEIGHT)
        sb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sb.setMinimumWidth(60)
        sb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return sb

    def _horizontal_pair_widget(self, a: QWidget, b: QWidget) -> QWidget:
        wrap = QWidget()
        wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(a, 1)
        row.addWidget(b, 1)
        return wrap

    def _make_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: rgba(255,255,255,0.08);")
        return line

    def _make_button(
        self,
        text: str,
        on_click,
        *,
        accent: str | None = None,
        primary: bool = False,
    ) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(self._PRIMARY_BUTTON_HEIGHT if primary else self._BUTTON_HEIGHT)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(on_click)

        if accent is not None:
            btn.setStyleSheet(
                f"QPushButton {{ "
                f"background-color: {accent}; color: white; border: none; "
                f"border-radius: 4px; padding: 0 14px; "
                f"}} "
                f"QPushButton:hover {{ background-color: {self._lighten(accent)}; }} "
                f"QPushButton:pressed {{ background-color: {self._darken(accent)}; }} "
                f"QPushButton:disabled {{ background-color: #444; color: #888; }}"
            )

        return btn

    @staticmethod
    def _lighten(hex_color: str, amount: float = 0.12) -> str:
        r, g, b = _UIBuilderMixin._hex_to_rgb(hex_color)
        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _darken(hex_color: str, amount: float = 0.12) -> str:
        r, g, b = _UIBuilderMixin._hex_to_rgb(hex_color)
        r = max(0, int(r * (1 - amount)))
        g = max(0, int(g * (1 - amount)))
        b = max(0, int(b * (1 - amount)))
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
