from __future__ import annotations

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


class _UIBuilderMixin:

    _WINDOW_TITLE = "Configure"
    _WINDOW_INIT_W = 390
    _WINDOW_INIT_H = 270
    _FONT_FAMILY = "SansSerif"

    _UPDATE_ACCENT = "rgb(17, 103, 255)"

    _FIELDS: tuple[tuple[str, str, str], ...] = (
        ("le_sensor_size", "Sensor pixel size", "µm"),
        ("le_magnification", "Lens magnification", "X"),
        ("le_fps", "Frame per second", ""),
        ("le_temp", "Temperature", "K"),
        ("le_eta", "Viscosity", "mPa·s"),
    )

    def _build_ui(self) -> None:
        self.setWindowTitle(self._WINDOW_TITLE)
        self.resize(self._WINDOW_INIT_W, self._WINDOW_INIT_H)
        self.setFont(QFont(self._FONT_FAMILY))

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(10)

        root.addLayout(self._build_field_grid())
        root.addStretch(1)
        root.addLayout(self._build_button_row())

    def _build_field_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        for row, (attr, label_text, unit_text) in enumerate(self._FIELDS):
            grid.addWidget(QLabel(label_text), row, 0)

            edit = QLineEdit()
            setattr(self, attr, edit)
            grid.addWidget(edit, row, 1)

            grid.addWidget(QLabel(unit_text), row, 2)

        grid.setColumnStretch(1, 1)
        return grid

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        self.btn_update = QPushButton("Update")
        self.btn_update.setStyleSheet(
            f"background-color: {self._UPDATE_ACCENT};"
        )

        for btn in (self.btn_load, self.btn_save, self.btn_update):
            row.addWidget(btn)

        return row

