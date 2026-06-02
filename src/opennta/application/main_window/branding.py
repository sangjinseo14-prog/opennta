"""Qt rendering helpers for OpenNtaMainWindow (logo + fit-to-view icon)."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)


def resolve_logo_path() -> Path:
    if getattr(sys, "frozen", False):
        package_root = Path(sys._MEIPASS) / "opennta"
        return package_root / "application" / "main_window" / "logo.png"
    return Path(__file__).resolve().parent / "logo.png"


def generate_logo(view_height: int = 50) -> QImage | None:
    art_str = (
        "     ▄▀▀▀▄   █▀▀▄   █▀▀▀  █▒   █▒  █▒   █▒ ▀▀█▀▀   ▄█▒  \n"
        "    █▒  █▒  █▒  █▒ █▒    █▀█  █▒  █▀█  █▒   █▒   ▄▀ █▒  \n"
        "   █▒  █▒  █▀▀▀▀  █▀▀▀  █▒ █ █▒  █▒ █ █▒   █▒  ▄█▄▄▄█▒  \n"
        "  █▒  █▒  █▒     █▒    █▒  ██▒  █▒  ██▒   █▒  █▒   █▒   \n"
        "   ▀▀▀   ▀▒     ▀▀▀▀  ▀▒   ▀▒  ▀▒   ▀▒   ▀▒  ▀▒   ▀▒      "
    )

    lines = [line for line in art_str.splitlines() if line.strip()]
    if not lines:
        return None

    candidates = [
        "Consolas",
        "Menlo",
        "Monaco",
        "SF Mono",
        "DejaVu Sans Mono",
        "Courier New",
    ]
    available = set(QFontDatabase().families())
    families = [f for f in candidates if f in available]
    if not families:
        families = [QFontDatabase.systemFont(QFontDatabase.FixedFont).family()]

    font = QFont()
    font.setFamilies(families)
    font.setStyleHint(QFont.Monospace)
    font.setFixedPitch(True)
    font.setPixelSize(90)

    fm = QFontMetrics(font)
    line_height = fm.height()
    max_width = max(fm.horizontalAdvance(line) for line in lines)

    raw_height = line_height * len(lines)
    if raw_height <= 0 or max_width <= 0:
        return None

    scale = view_height / float(raw_height)

    img_width = max(1, int(max_width * scale))
    img_height = max(1, int(raw_height * scale))

    img = QImage(img_width, img_height, QImage.Format_ARGB32)
    img.fill(Qt.transparent)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    painter.scale(scale, scale)
    painter.setFont(font)
    painter.setPen(QColor("#909FB2"))

    y = fm.ascent()
    for line in lines:
        painter.drawText(0, y, line)
        y += line_height

    painter.end()
    return img


def make_fit_to_view_icon(size: int = 24, color: QColor = QColor("white")) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)

    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)

    pen = QPen(color, 2)
    p.setPen(pen)

    margin = int(size * 0.25)
    end = size - margin
    wing = int(size * 0.2)

    p.drawLine(margin, margin, margin - wing, margin)
    p.drawLine(margin, margin, margin, margin - wing)

    p.drawLine(end, margin, end + wing, margin)
    p.drawLine(end, margin, end, margin - wing)

    p.drawLine(margin, end, margin - wing, end)
    p.drawLine(margin, end, margin, end + wing)

    p.drawLine(end, end, end + wing, end)
    p.drawLine(end, end, end, end + wing)

    p.end()
    return QIcon(pix)
