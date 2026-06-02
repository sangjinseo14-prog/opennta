from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .types import AnalysisConfig


class MSDCalculator:

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def calculate(self, df: pd.DataFrame, lag_frame) -> dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]]:

        msd_all: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]] = {}

        for pid, group in df.groupby("ID", sort=False):
            coords = group.sort_values("FRAME")[["X_diff_corr", "Y_diff_corr"]].to_numpy()
            N = len(coords)
            lags: list[int] = []
            msds: list[float] = []

            for lag in range(1, min(5 * lag_frame + 1, N)):
                disp = coords[lag:] - coords[:-lag]
                msds.append(np.mean(np.sum(disp ** 2, axis=1)))
                lags.append(lag)

            if msds:
                msd_all[pid] = (np.array(lags), np.array(msds))

        return msd_all

    def calculate_r2_per_track(
        self,
        msd_by_id: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]],
        lag_frame: int,
        extra_points: int = 2,
        min_points: int = 3,
    ) -> dict[int, float]:

        r2_dict: dict[int, float] = {}
        dt = self.config.dt
        target_n = lag_frame + extra_points

        for pid, (lags, msds) in msd_by_id.items():
            if len(lags) < min_points:
                r2_dict[pid] = float("nan")
                continue

            use_n = min(target_n, len(lags))
            if use_n < min_points:
                r2_dict[pid] = float("nan")
                continue

            t = lags[:use_n] * dt
            y = msds[:use_n]

            if np.allclose(y, y[0]):
                r2_dict[pid] = float("nan")
                continue

            a, b = np.polyfit(t, y, 1)
            y_pred = a * t + b

            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))

            if ss_tot <= 0:
                r2 = 1.0
            else:
                r2 = 1.0 - ss_res / ss_tot

            r2_dict[pid] = float(r2) if np.isfinite(r2) else float("nan")

        return r2_dict
