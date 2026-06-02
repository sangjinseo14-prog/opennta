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
    QFormLayout,
    QFrame,
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

_LABEL_STYLES = {
    "info": "QLabel { color: #c5cbd6; font-size: 13px; padding: 8px 0 0 0; }",
}

_DIALOG_STYLE = """
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
"""

_MODEL_DESCRIPTION = (
    "Pipeline:\n"
    "  1) OpenCV preprocessing\n"
    "     spots → Usp_n, Vsp_n, mask\n"
    "     (128×128 normalized maps)\n"
    "  2) U-Net inference\n"
    "     inputs → predicted u, v fields\n"
    "     speed = √(u² + v²)\n"
    "\n"
    "The predicted (u, v) field is applied\n"
    "to correct per-spot drift."
)

_WINDOW_TITLE = "Configure U-Net Drift Field"
_WINDOW_INIT_W = 1300
_WINDOW_INIT_H = 750
_WINDOW_MIN_H = 650

_LEFT_MIN_W = 300
_LEFT_MAX_W = 350
_PANEL_MIN_W = 300
_PANEL_MAX_W = 450

_BUTTON_HEIGHT = 20
_PRIMARY_BUTTON_HEIGHT = 30

_ACCENT = "#1c74ff"
_ACCENT_MUTED = "#3b68c7"


class _UIBuilderMixin:

    def _build_ui(self) -> None:
        self.setWindowTitle(_WINDOW_TITLE)
        self.setModal(True)
        self.resize(_WINDOW_INIT_W, _WINDOW_INIT_H)

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        root.addWidget(self._build_left_panel(), stretch=0)
        root.addWidget(self._build_panel("OpenCV preprocessing (inputs)", "inputs"), stretch=1)
        root.addWidget(self._build_panel("U-Net inference (outputs)", "outputs"), stretch=1)

        self.setStyleSheet(_DIALOG_STYLE)

    def _build_left_panel(self) -> QWidget:
        wrap = QWidget()
        wrap.setMinimumWidth(_LEFT_MIN_W)
        wrap.setMaximumWidth(_LEFT_MAX_W)

        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(14)

        info = QGroupBox("Model info")
        info_v = QVBoxLayout(info)
        info_v.setContentsMargins(14, 18, 14, 14)
        info_v.setSpacing(6)

        info_v.addWidget(QLabel("U-Net drift field predictor"))
        n_ids = self.df["ID"].nunique() if "ID" in self.df.columns else 0
        info_v.addWidget(QLabel(f"Tracks: {n_ids}"))
        info_v.addWidget(QLabel(f"Rows: {len(self.df)}"))
        info_v.addWidget(self._make_label(_MODEL_DESCRIPTION, style="info"))

        col.addWidget(info)

        col.addLayout(self._build_image_size_row())

        self.btn_preprocess = self._make_button(
            "1) Run preprocessing (OpenCV)", self._run_preprocess, accent=_ACCENT_MUTED,
        )
        col.addWidget(self.btn_preprocess)

        self.btn_inference = self._make_button(
            "2) Run inference (U-Net)", self._run_inference, accent=_ACCENT_MUTED,
        )
        self.btn_inference.setEnabled(False)
        col.addWidget(self.btn_inference)

        col.addWidget(self._build_overlay_group())

        self.chk_export_csv = QCheckBox("Export velocity fields as *.csv")
        self.chk_export_csv.setCursor(Qt.PointingHandCursor)
        col.addStretch(1)
        col.addWidget(self.chk_export_csv)

        col.addWidget(self._make_separator())
        col.addLayout(self._build_bottom_actions())
        return wrap

    def _build_overlay_group(self) -> QGroupBox:
        box = QGroupBox("Overlay")
        form = QFormLayout(box)
        form.setContentsMargins(14, 18, 14, 14)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.btn_vectors = QPushButton("Show field overlay")
        self.btn_vectors.setCheckable(True)
        self.btn_vectors.setCursor(Qt.PointingHandCursor)
        self.btn_vectors.setFixedHeight(_BUTTON_HEIGHT)
        self.btn_vectors.setEnabled(False)
        self.btn_vectors.setStyleSheet(
            f"QPushButton {{ background-color: {_ACCENT_MUTED}; color: white; "
            f"border: none; border-radius: 4px; padding: 0 14px; }} "
            f"QPushButton:checked {{ background-color: {_ps.VECTOR_COLOR}; }} "
            f"QPushButton:disabled {{ background-color: #444; color: #888; }}"
        )
        self.btn_vectors.toggled.connect(self._toggle_vectors)

        self.sb_vector_n = QSpinBox()
        self.sb_vector_n.setRange(_ps.VECTOR_N_MIN, _ps.VECTOR_N_MAX)
        self.sb_vector_n.setValue(_ps.VECTOR_N_DEFAULT)
        self.sb_vector_n.setFixedHeight(_BUTTON_HEIGHT)
        self.sb_vector_n.valueChanged.connect(self._on_vector_n_changed)

        form.addRow(self.btn_vectors)
        form.addRow("N (grid)", self.sb_vector_n)
        return box

    def _image_size_default(self) -> int:
        # Side length of the N×N source image the model was trained against.
        # Deferred import: pulling unet_bundle at module load would eagerly
        # import the heavy inference module at app startup.
        try:
            from opennta.analysis.unet_field.unet_bundle.config import ACTUAL_IMAGE_WIDTH_PX
            return int(ACTUAL_IMAGE_WIDTH_PX)
        except Exception:
            return 2304

    def _build_image_size_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.le_image_size = QLineEdit(str(self._image_size_default()))
        self.le_image_size.setValidator(QIntValidator(1, 100000, self.le_image_size))
        self.le_image_size.setFixedHeight(_BUTTON_HEIGHT)
        self.le_image_size.setToolTip(
            "Side length N (px) of the N×N source image the spots were detected in."
        )

        row.addWidget(QLabel("Image size N (px)"))
        row.addWidget(self.le_image_size, stretch=1)
        return row

    def _build_bottom_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.btn_accept = self._make_button(
            "Correct using field", self._accept_field, accent=_ACCENT, primary=True,
        )
        self.btn_accept.setEnabled(False)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(_PRIMARY_BUTTON_HEIGHT)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.reject)

        row.addWidget(self.btn_accept, stretch=2)
        row.addWidget(self.btn_cancel, stretch=1)
        return row

    def _build_panel(self, header_text: str, suffix: str) -> QWidget:
        wrap = QWidget()
        wrap.setMinimumWidth(_PANEL_MIN_W)
        wrap.setMaximumWidth(_PANEL_MAX_W)

        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        header = QLabel(header_text)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("QLabel { padding: 2px 0; }")

        fig = _ps.style_figure(Figure(figsize=(5.5, 11.0)))
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(_WINDOW_MIN_H)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar = NavigationToolbar(canvas, self)

        setattr(self, f"fig_{suffix}", fig)
        setattr(self, f"canvas_{suffix}", canvas)
        setattr(self, f"toolbar_{suffix}", toolbar)

        col.addWidget(header)
        col.addWidget(toolbar)
        col.addWidget(canvas, stretch=1)
        return wrap

    def _make_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: rgba(255,255,255,0.08);")
        return line

    def _make_label(self, text: str, *, style: str | None = None) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        if style is not None and style in _LABEL_STYLES:
            lbl.setStyleSheet(_LABEL_STYLES[style])
        return lbl

    def _make_button(self, text, on_click, *, accent=None, primary=False) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(_PRIMARY_BUTTON_HEIGHT if primary else _BUTTON_HEIGHT)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(on_click)
        if accent is not None:
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {accent}; color: white; border: none; "
                f"border-radius: 4px; padding: 0 14px; }} "
                f"QPushButton:disabled {{ background-color: #444; color: #888; }}"
            )
        return btn
