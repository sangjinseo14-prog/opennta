"""Cross-platform default font selection for the Qt UI."""

from __future__ import annotations

import sys

from PyQt5.QtGui import QFont, QFontDatabase

_REFERENCE_DPI = 96


def _candidate_families() -> list[str]:
    if sys.platform.startswith("win"):
        return ["Sans Serif", "Segoe UI", "Tahoma", "Arial", "Verdana"]
    if sys.platform == "darwin":
        return ["Helvetica", "Helvetica Neue", "Arial", "Lucida Grande"]
    return ["Inter", "Ubuntu", "Noto Sans", "DejaVu Sans"]


def point_to_pixel(point_size: float) -> int:
    return max(1, round(point_size * _REFERENCE_DPI / 72))


def get_app_font(point_size: int = 10) -> QFont:
    candidates = _candidate_families()
    available = set(QFontDatabase().families())
    family = next((name for name in candidates if name in available), candidates[-1])

    font = QFont(family)
    font.setPixelSize(point_to_pixel(point_size))
    font.setStyleStrategy(QFont.PreferAntialias)
    font.setHintingPreference(QFont.PreferFullHinting)
    return font
