"""Builds the bottom status row (memory readout + resize button + progress bar)."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QWidget,
)


class _StatusBarBuilder:
    """Mixin: ``_build_status_row()`` returns the bottom row layout."""

    def _build_status_row(self, parent: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(-1, 0, -1, -1)
        row.setSpacing(6)

        # VBot horizontal layout: memory status, resize button, progress.
        memory_status = QLineEdit(parent)
        memory_status.setMouseTracking(False)
        memory_status.setAutoFillBackground(False)
        # NB: setReadOnly BEFORE applying the stylesheet so the EXTRA_QSS
        # property-selector for editable line edits doesn't repaint over us.
        memory_status.setReadOnly(True)
        memory_status.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        memory_status.setFrame(False)
        self._register("main_lineEdit_memoryStatus", memory_status)
        row.addWidget(memory_status)

        row.addItem(self._expanding_horizontal_spacer())

        resize_btn = QPushButton(parent)
        resize_btn.setText("")
        self._register("main_pushButton_resize", resize_btn)
        row.addWidget(resize_btn)

        progress = QProgressBar(parent)
        progress.setValue(0)
        self._register("main_progressBar_progress", progress)
        row.addWidget(progress)

        row.setStretch(0, 8)
        row.setStretch(1, 8)
        row.setStretch(3, 8)
        return row
