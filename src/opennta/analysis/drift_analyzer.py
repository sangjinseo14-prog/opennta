from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .types import AnalysisConfig


class DriftAnalyzer:

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def calculate(
        self,
        df: pd.DataFrame,
        x_col: str = "X_diff",
        y_col: str = "Y_diff",
    ) -> tuple[float, float, NDArray[np.floating], dict[int, tuple[float, float]]]:

        if df.empty:
            return 0.0, 0.0, np.empty((0, 2)), {}

        sorted_df = df.sort_values(["ID", "FRAME"])
        grouped = sorted_df.groupby("ID", sort=False)

        first = grouped[[x_col, y_col, "FRAME"]].first()
        last = grouped[[x_col, y_col, "FRAME"]].last()
        sizes = grouped.size()

        dt_seconds = (last["FRAME"] - first["FRAME"]) * self.config.dt
        valid = (dt_seconds > 0) & (sizes >= 2)

        vx = (last.loc[valid, x_col] - first.loc[valid, x_col]) / dt_seconds[valid]
        vy = (last.loc[valid, y_col] - first.loc[valid, y_col]) / dt_seconds[valid]

        vx_arr = vx.to_numpy()
        vy_arr = vy.to_numpy()
        speeds_array = np.column_stack([vx_arr, vy_arr]) if vx_arr.size else np.empty((0, 2))
        drift_by_id = {int(pid): (vx_arr[i], vy_arr[i]) for i, pid in enumerate(vx.index)}

        if speeds_array.size == 0:
            return 0.0, 0.0, np.empty((0, 2)), {}

        mean_vx, mean_vy = np.mean(speeds_array, axis=0)
        return mean_vx, mean_vy, speeds_array, drift_by_id
