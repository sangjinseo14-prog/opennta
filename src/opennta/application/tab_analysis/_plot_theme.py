"""Dark matplotlib rcParams for the Analysis result-tab figures.

Lives next to `output.py` so its global `plt.figure` / `plt.subplots`
calls pick up the qdarkstyle-matching palette without crossing into the
configurator-dialog subpackages.
"""
from __future__ import annotations

import matplotlib as mpl

# Palette aligned with qdarkstyle (PyQt5).
BG_FIGURE = "#19232D"
BG_AXES = "#1f2a36"
FG_TEXT = "#dde2eb"
FG_MUTED = "#9aa3b2"
GRID = "#2c3a4a"


_APPLIED = False


def apply_dark_plot_theme() -> None:
    # Safe to call repeatedly; the second call is a no-op.
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    mpl.rcParams.update({
        "figure.facecolor": BG_FIGURE,
        "figure.edgecolor": BG_FIGURE,
        "axes.facecolor": BG_AXES,
        "savefig.facecolor": BG_FIGURE,
        "savefig.edgecolor": BG_FIGURE,

        "text.color": FG_TEXT,
        "axes.labelcolor": FG_TEXT,
        "axes.titlecolor": FG_TEXT,
        "axes.edgecolor": FG_MUTED,
        "axes.linewidth": 0.8,
        "xtick.color": FG_MUTED,
        "ytick.color": FG_MUTED,
        "xtick.labelcolor": FG_TEXT,
        "ytick.labelcolor": FG_TEXT,

        "axes.grid": False,
        "grid.color": GRID,
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,

        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue",
            "Arial", "DejaVu Sans",
        ],
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "normal",
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        # Render math text in the same sans family, not the default Computer
        # Modern italics, so $V_x$ etc. visually match the surrounding labels.
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.default": "regular",

        # Do not set figure.dpi here: FigureCanvasQTAgg already honors
        # devicePixelRatioF() for live canvases. savefig.dpi is bumped because
        # disk-saved diagnostic PNGs do not go through the Qt canvas.
        "savefig.dpi": 200,
        "savefig.bbox": "tight",

        "lines.antialiased": True,
        "patch.antialiased": True,

        # Disable FreeType hinting for text: matplotlib's default snaps
        # horizontal glyphs to the pixel grid, which makes x-axis labels and
        # titles look coarser than the rotated y-axis label (rotated text
        # bypasses hinting). With hinting off every orientation renders
        # through the same unhinted path.
        "text.hinting": "none",
    })
