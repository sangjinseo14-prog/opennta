"""Plot constants and helpers for the TrackMate configurator dialog."""
from __future__ import annotations

# Palette aligned with qdarkstyle (PyQt5) + the app accent #1c74ff.
HIST_BAR = "#1f77b4"
HIST_SMOOTHED = "#ffd23f"
HIST_FIT = "#1c74ff"
HIST_THRESHOLD = "#ff3030"
HIST_CUTOFF = "#ffffff"
SPOT_OVERLAY = "#ff3030"

FS_LEGEND = 8

IMAGE_VMIN_PCT = 1.0
IMAGE_VMAX_PCT = 99.0

HIST_BINS = 200
HIST_BAR_ALPHA = 0.4
HIST_SMOOTHED_LW = 1.3
HIST_FIT_LW = 1.5
HIST_THRESHOLD_LW = 1.2
HIST_CUTOFF_LW = 1.0

SPOT_MARKER_SIZE = 14
SPOT_MARKER_LW = 1.0

TITLE_IMAGE_DEFAULT = "Selected frame + detected spots"
TITLE_HISTOGRAM = "Quality histogram"
LABEL_HIST_X = "Quality"
LABEL_HIST_Y = "Density"


def style_image_axis(ax, title: str = TITLE_IMAGE_DEFAULT) -> None:
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)


def style_histogram_axis(ax) -> None:
    ax.set_title(TITLE_HISTOGRAM)
    ax.set_xlabel(LABEL_HIST_X)
    ax.set_ylabel(LABEL_HIST_Y)
    ax.set_box_aspect(1)
