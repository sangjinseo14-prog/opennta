"""Shared UI constants for the main window builder.

Lives below the tab builders in the import graph so both ``_ui_builder`` and
the per-tab builder mixins can pull these values without forming a cycle.
"""
from __future__ import annotations

from ..common.fonts import point_to_pixel

# Pixel sizes equivalent to the legacy pt sizes used in the .ui under 96 DPI.
FS_SMALL = point_to_pixel(9)    # 12 px — labels, line edits, combo boxes, small buttons
FS_MEDIUM = point_to_pixel(10)  # 13 px — "Summary" header
FS_LARGE = point_to_pixel(11)   # 15 px — "Run …" / "Browse" / "Find Fiji" buttons, log/summary browsers
