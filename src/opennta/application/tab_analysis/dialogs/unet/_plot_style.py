"""Plot constants, dark theme, and helpers for the U-Net field configurator
and the matching saved-PNG diagnostic."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.figure import Figure

# Palette aligned with qdarkstyle (PyQt5).
BG_FIGURE = "#19232D"   # qdarkstyle window background
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

# Vector-overlay knobs for the dialog's live preview and saved diagnostics.
VECTOR_COLOR = "#e53935"
VECTOR_N_DEFAULT = 10
VECTOR_N_MIN = 2
VECTOR_N_MAX = 40

# Model output is in m/s; the dialog displays in µm/s.
UNET_DISPLAY_SCALE = 1e6

# Per-panel layout for the three-stack figures used in the dialog.
INPUT_SPECS = (
    ("Usp_n (OpenCV)", "coolwarm", ""),
    ("Vsp_n (OpenCV)", "coolwarm", ""),
    ("mask (OpenCV)",  "gray",     ""),
)
OUTPUT_SPECS = (
    ("U predicted",     "coolwarm", "µm/s"),
    ("V predicted",     "coolwarm", "µm/s"),
    ("Field predicted", "viridis",  "µm/s"),
)

# GridSpec margins/spacing for the three-stack figures.
FIG_LAYOUT = dict(left=0.12, right=0.90, top=0.93, bottom=0.06, hspace=0.40)

# Manual colorbar geometry (pixels).
CBAR_W_PX = 18
CBAR_H_PX = 140
CBAR_GAP_PX = 18

# Tick locations for 128×128 grids (filtered against axis range at draw time).
AXIS_TICKS = (0, 32, 64, 96, 128)


def style_figure(fig: Figure) -> Figure:
    # rcParams already cover figures created after the package-wide dark
    # theme is applied, but this is cheap insurance for figures whose
    # constructor was passed an explicit facecolor.
    fig.set_facecolor(BG_FIGURE)
    fig.patch.set_facecolor(BG_FIGURE)
    return fig
