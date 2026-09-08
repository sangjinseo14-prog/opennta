from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import AnalysisConfig


class DiffusionEstimator:

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def estimate(
        self,
        msd_by_id: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]],
        lag_frame: int
    ) -> tuple[
        dict[int, float],
        dict[int, float],
        list[tuple[int, int]],
        list[tuple[int, float]],
    ]:
        # insufficient: tracks too short to fit (pid, n_points).
        # nonpositive_d: tracks whose OLS slope gave D <= 0 (pid, D); kept as a
        # separate category rather than dropped, since a non-positive slope is
        # common for slow/large or noise-dominated tracks and silently folding
        # them away biases the recovered size distribution toward smaller sizes.
        D_by_id: dict[int, float] = {}
        diameters_by_id: dict[int, float] = {}
        insufficient: list[tuple[int, int]] = []
        nonpositive_d: list[tuple[int, float]] = []

        for pid, (lags, msds) in msd_by_id.items():
            use_n = min(lag_frame, len(lags))

            if use_n < 2:
                insufficient.append((pid, use_n))
                continue

            t = np.asarray(
                [self.config.effective_lag_time(int(lag)) for lag in lags[:use_n]]
            )
            y = msds[:use_n]
            slope, _ = np.polyfit(t, y, 1)
            D = slope / 4
            if D > 0:
                D_by_id[pid] = D
            else:
                nonpositive_d.append((pid, float(D)))

        if len(D_by_id) == 0:
            diameters_by_id = {}
        else:
            diameters_by_id = {
                pid: (self.config.KB * self.config.temp) /
                     (3 * np.pi * self.config.eta * (D * 1e-12)) * 1e9
                for pid, D in D_by_id.items()
            }

        return D_by_id, diameters_by_id, insufficient, nonpositive_d
