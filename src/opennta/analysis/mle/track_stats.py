"""Per-track (z_k, n_k) statistics used by the MLE-based distributors.

z_k: mean squared displacement at lag = 1 frame, in m^2.
n_k: number of 1-step samples averaged into z_k.

DataPreprocessor drops the first row per ID after diff(), so the per-ID
groupby size is N_raw - 1. MSDCalculator then takes another diff over the
drift-corrected cumulative displacements, losing one more sample. The
actual count averaged into msds[0] is therefore N_raw - 2, i.e.
n_k = groupby_size - 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class TrackStats:
    # pids[i], z_m2[i], n_steps[i] refer to the same track. z_m2 is in m^2
    # (SI) to match the Stokes-Einstein formulation.
    pids: NDArray[np.int64]
    z_m2: NDArray[np.float64]
    n_steps: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.pids.size)


def extract_track_stats(
    msd_by_id: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]],
    df: pd.DataFrame,
) -> TrackStats:
    # Tracks without a lag=1 entry or without any steps in df are dropped.
    if not msd_by_id or df is None or df.empty:
        return TrackStats(
            pids=np.empty(0, dtype=np.int64),
            z_m2=np.empty(0, dtype=np.float64),
            n_steps=np.empty(0, dtype=np.int64),
        )

    # See module docstring: n_k = steps_per_id - 1 to account for the second
    # diff() inside MSDCalculator.
    steps_per_id = df.groupby("ID").size()

    pids: list[int] = []
    z_list: list[float] = []
    n_list: list[int] = []

    for pid, (lags, msds) in msd_by_id.items():
        if len(lags) == 0 or int(lags[0]) != 1:
            continue
        n = int(steps_per_id.get(pid, 0)) - 1
        if n < 2:
            continue
        z_um2 = float(msds[0])
        if not np.isfinite(z_um2) or z_um2 <= 0:
            continue
        pids.append(int(pid))
        z_list.append(z_um2 * 1e-12)  # um^2 -> m^2
        n_list.append(n)

    return TrackStats(
        pids=np.asarray(pids, dtype=np.int64),
        z_m2=np.asarray(z_list, dtype=np.float64),
        n_steps=np.asarray(n_list, dtype=np.int64),
    )


__all__ = ["TrackStats", "extract_track_stats"]
