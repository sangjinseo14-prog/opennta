"""Callback types are re-exported from opennta.common.progress for back-compat."""

from __future__ import annotations

from ..common.progress import (
    LogCallback,
    ProgressCallback,
    ProgressEmitter,
    noop_log_callback,
    noop_progress_callback,
)


class TrackingStatus:
    STARTED = 1000
    PROCESSING_TIFF = 1001
    SAVING_IMAGES = 1002
    THRESHOLD_DETECTION_START = 1003
    THRESHOLD_DETECTION_COMPLETE = 1004
    CALCULATING_THRESHOLD = 1005
    QUALITY_LOADED = 1006
    THRESHOLD_CALCULATED = 1007
    TRACKING_START = 1008
    FULL_TRACKMATE_START = 1009
    FULL_TRACKMATE_COMPLETED = 1010
    TRACKING_COMPLETE = 1011
    ITEM_COMPLETE = 1012
    ALL_COMPLETE = 1013

    ERROR_THRESHOLD_DETECTION = 1900
    ERROR_THRESHOLD_CALCULATION = 1901
    ERROR_TRACKING = 1902
    ERROR_GENERAL = 1999


__all__ = [
    "ProgressCallback",
    "LogCallback",
    "TrackingStatus",
    "noop_progress_callback",
    "noop_log_callback",
    "ProgressEmitter",
]
