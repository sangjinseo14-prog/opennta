from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget


class _KernelPreview(QWidget):

    _FIXED_W = 128
    _FIXED_H = 128
    _PAD = 6

    _COLOR_MIN = (20, 24, 36)
    _COLOR_MAX = (255, 140, 40)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._FIXED_W, self._FIXED_H)
        self._ksize = 5
        self._sigma = 1.0

    def set_kernel(self, ksize: int, sigma: float) -> None:
        # ksize is forced to odd, sigma forced positive.
        ksize = max(1, int(ksize))
        if ksize % 2 == 0:
            ksize += 1
        sigma = float(sigma)
        if sigma <= 0.0:
            sigma = 1e-6
        self._ksize = ksize
        self._sigma = sigma
        self.update()

    def _compute_kernel(self) -> np.ndarray:
        r = self._ksize // 2
        offsets = np.arange(-r, r + 1, dtype=float)
        dj, di = np.meshgrid(offsets, offsets, indexing="ij")
        return np.exp(-(dj * dj + di * di) / (2.0 * self._sigma * self._sigma))

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        p.fillRect(self.rect(), QColor("#161b24"))

        k = self._compute_kernel()
        ksize = self._ksize

        area_w = self._FIXED_W - 2 * self._PAD
        area_h = self._FIXED_H - 2 * self._PAD
        cell = max(4, min(area_w // ksize, area_h // ksize))
        total = cell * ksize
        x0 = (self._FIXED_W - total) // 2
        y0 = (self._FIXED_H - total) // 2

        vmin = float(k.min())
        vmax = float(k.max())
        rng = max(vmax - vmin, 1e-12)

        font_size = max(5, int(cell * 0.24))
        font = QFont("Arial", font_size)
        font.setBold(False)
        font.setHintingPreference(QFont.PreferFullHinting)
        p.setFont(font)

        cmin = self._COLOR_MIN
        cmax = self._COLOR_MAX

        for j in range(ksize):
            for i in range(ksize):
                val = float(k[j, i])
                t = (val - vmin) / rng

                r = int(cmin[0] + (cmax[0] - cmin[0]) * t)
                g = int(cmin[1] + (cmax[1] - cmin[1]) * t)
                b = int(cmin[2] + (cmax[2] - cmin[2]) * t)

                rect = QRect(x0 + i * cell, y0 + j * cell, cell, cell)
                p.fillRect(rect, QColor(r, g, b))

                p.setPen(QPen(QColor(85, 95, 110), 1))
                p.drawRect(rect)

                text = f"{val:.3f}" if val > 0.95 else f"{val:.2f}"
                p.setPen(QColor(255, 255, 255))
                p.drawText(rect.adjusted(0, 0, 0, -1), Qt.AlignCenter, text)

        p.setPen(QPen(QColor(70, 80, 95), 1))
        p.drawRect(0, 0, self._FIXED_W - 1, self._FIXED_H - 1)

        p.end()
