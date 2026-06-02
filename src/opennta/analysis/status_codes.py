"""Analysis-module progress codes (2000 series)."""

from __future__ import annotations

from ..common.progress import (
    LogCallback,
    ProgressCallback,
    ProgressEmitter,
    noop_log_callback,
    noop_progress_callback,
)


class AnalysisStatus:
    STARTED = 2000
    LOADING_DATA = 2001
    DATA_LOADED = 2002
    CALCULATING_DRIFT = 2003
    DRIFT_CALCULATED = 2004
    CORRECTING_DRIFT = 2005
    DRIFT_CORRECTED = 2006
    CALCULATING_MSD = 2007
    MSD_CALCULATED = 2008
    ESTIMATING_DIFFUSION = 2009
    DIFFUSION_ESTIMATED = 2010
    COMPLETE = 2011

    ERROR_LOADING = 2900
    ERROR_DRIFT = 2901
    ERROR_MSD = 2902
    ERROR_DIFFUSION = 2903
    ERROR_GENERAL = 2999


__all__ = [
    "ProgressCallback",
    "LogCallback",
    "AnalysisStatus",
    "noop_progress_callback",
    "noop_log_callback",
    "ProgressEmitter",
]
