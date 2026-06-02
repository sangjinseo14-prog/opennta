"""Plot constants, dark theme, and helpers for the Numerical-field
configurator and the matching saved-PNG diagnostic."""
from __future__ import annotations

from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

# Palette aligned with qdarkstyle (PyQt5) + the app accent #1c74ff.
BG_FIGURE = "#19232D"   # qdarkstyle window background
BG_AXES = "#1f2a36"     # slightly lighter so panels stand out from the figure
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


def style_figure(fig: Figure) -> Figure:
    # rcParams already cover figures created after apply_dark_plot_theme(),
    # but this is cheap insurance for figures whose constructor was passed
    # an explicit facecolor.
    fig.set_facecolor(BG_FIGURE)
    fig.patch.set_facecolor(BG_FIGURE)
    return fig

# Titles use MathText so subscripts render in both Qt-embedded and Agg figures.
PLOT_SPECS = (
    ("Tracks on grid", None, ""),
    ("Mean velocity on grid", "viridis", "µm/s"),
    (r"CI$_{95}$", "magma", "µm/s"),
    ("Smoothed velocity-field", "viridis", "µm/s"),
)

TRACKS_INDEX = 0
MEAN_INDEX = 1
CI95_INDEX = 2
SMOOTHED_INDEX = 3

FS_TITLE = 13
FS_LABEL = 10
FS_TICK = 9
FS_CBAR_LABEL = 9
FS_CBAR_TICK = 8

TRACK_CMAP = "viridis"
TRACK_LW = 0.5
TRACK_ALPHA = 0.8
TRACK_TIME_LABEL = "Normalized track time"

VECTOR_COLOR = "#e53935"
VECTOR_KW: dict[str, Any] = {
    "color": VECTOR_COLOR,
    "angles": "xy",
    "scale_units": "xy",
    "scale": 1.0,
    "width": 0.006,
    "headwidth": 3.2,
    "headlength": 4.0,
    "headaxislength": 3.6,
    "pivot": "middle",
    "zorder": 5,
}


def style_panel_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel("X", fontsize=FS_LABEL)
    ax.set_ylabel("Y", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)


def style_colorbar(cb, unit: str = "") -> None:
    if unit:
        cb.set_label(unit, fontsize=FS_CBAR_LABEL)
    cb.ax.tick_params(labelsize=FS_CBAR_TICK)
    ot = cb.ax.yaxis.get_offset_text()
    ot.set_fontsize(FS_CBAR_TICK)
    ot.set_horizontalalignment("left")
    ot.set_verticalalignment("bottom")
    ot.set_position((0.0, 1.04))


def draw_tracks_on_axes(
    ax,
    df,
    n_windows: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
):
    # Returns a ScalarMappable for fig.colorbar, or None if there were no
    # tracks to draw.
    n = max(int(n_windows), 2)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    tick_step = max(1, n // 10)
    ticks = np.arange(0, n, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    for k in range(n + 1):
        ax.axvline(k - 0.5, color=GRID, linewidth=0.4, alpha=0.7, zorder=0)
        ax.axhline(k - 0.5, color=GRID, linewidth=0.4, alpha=0.7, zorder=0)

    x_min, x_max = x_range
    y_min, y_max = y_range
    ww = (x_max - x_min) / n
    wh = (y_max - y_min) / n

    if ww <= 0 or wh <= 0 or df is None or df.empty:
        ax.text(
            0.5, 0.5, "(no tracks)",
            transform=ax.transAxes,
            ha="center", va="center",
            color=FG_MUTED, fontsize=FS_LABEL,
        )
        return None

    xs = (df["X"].to_numpy(dtype=float) - x_min) / ww - 0.5
    ys = (df["Y"].to_numpy(dtype=float) - y_min) / wh - 0.5
    ids = df["ID"].to_numpy()

    if ids.size < 2:
        return None

    id_changes = np.where(np.diff(ids) != 0)[0] + 1
    id_starts = np.concatenate([[0], id_changes])
    id_ends = np.concatenate([id_changes, [ids.size]])

    cmap = mpl.colormaps[TRACK_CMAP]

    seg_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    for s, e in zip(id_starts, id_ends, strict=False):
        if e - s < 2:
            continue
        pts = np.column_stack([xs[s:e], ys[s:e]])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        m = segs.shape[0]
        t = (np.arange(m, dtype=float) + 0.5) / m
        seg_chunks.append(segs)
        col_chunks.append(cmap(t))

    if not seg_chunks:
        return None

    segments = np.concatenate(seg_chunks, axis=0)
    colors = np.concatenate(col_chunks, axis=0)

    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=TRACK_LW,
        alpha=TRACK_ALPHA,
        zorder=2,
    )
    ax.add_collection(lc)

    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    return sm


def draw_quiver_overlay(ax, u, v):
    # Arrows are normalized so the longest spans ~90% of a grid cell, so the
    # overlay scales with the field magnitude. Returns None if degenerate.
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if u.ndim != 2 or u.shape != v.shape:
        return None

    h, w = u.shape
    if h < 1 or w < 1:
        return None

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    mag = np.hypot(u, v)
    finite = np.isfinite(mag)
    if not finite.any():
        return None
    max_mag = float(mag[finite].max())
    if not np.isfinite(max_mag) or max_mag <= 0.0:
        return None

    scale_factor = 0.9 / max_mag
    U_plot = np.where(finite, u * scale_factor, 0.0)
    V_plot = np.where(finite, v * scale_factor, 0.0)

    return ax.quiver(X, Y, U_plot, V_plot, **VECTOR_KW)
