"""Unit tests for ``opennta.analysis.mle.track_stats.extract_track_stats``."""
from __future__ import annotations

import numpy as np
import pandas as pd

from opennta.analysis.mle.track_stats import extract_track_stats


def _df_with_pids(*, pids_with_5_frames=(1, 2, 5, 99), pids_with_2_frames=(4,)):
    rows = []
    for pid in pids_with_5_frames:
        for frame in range(5):
            rows.append((pid, frame, frame, frame))
    for pid in pids_with_2_frames:
        for frame in range(2):
            rows.append((pid, frame, frame, frame))
    return pd.DataFrame(rows, columns=["ID", "X", "Y", "FRAME"])


def test_filters_invalid_lags():
    # 5 frames per ID -> after diff() + cumsum, groupby size = 4
    # n_k = groupby_size - 1 = 3
    # so a track with all-positive lag=1 MSD makes it through.
    # ID 99 has lag[0] != 1 and must be dropped.
    # ID 5 has lag[0] == 1 but msd <= 0 and must be dropped.
    # ID 4 has lag[0] == 1 but only 2 rows in df -> n_k = 1 < 2.
    df = _df_with_pids()
    msd_by_id = {
        1: (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])),
        2: (np.array([1, 2, 3]), np.array([2.0, 4.0, 6.0])),
        99: (np.array([2, 3]), np.array([1.0, 2.0])),
        5: (np.array([1, 2, 3]), np.array([-1.0, 1.0, 2.0])),
        4: (np.array([1, 2]), np.array([1.0, 2.0])),
    }
    stats = extract_track_stats(msd_by_id, df)
    assert set(stats.pids.tolist()) == {1, 2}
    # z_m2 should be msds[0] in m^2.
    assert np.allclose(stats.z_m2, np.array([1.0, 2.0]) * 1e-12)


def test_n_steps_equals_rows_minus_one_and_filters_below_two():
    # n_k = (rows per ID) - 1, accounting for the second diff() in MSDCalculator;
    # a track is kept only when n_k >= 2 (i.e. >= 3 rows in the displacement df).
    rows_per_pid = {10: 3, 11: 4, 12: 6, 13: 2}  # expected n_k: 2, 3, 5, and 1(drop)
    rows = []
    for pid, k in rows_per_pid.items():
        for frame in range(k):
            rows.append((pid, frame, frame, frame))
    df = pd.DataFrame(rows, columns=["ID", "X", "Y", "FRAME"])
    msd_by_id = {pid: (np.array([1, 2]), np.array([1.0, 2.0])) for pid in rows_per_pid}

    stats = extract_track_stats(msd_by_id, df)

    by_pid = dict(zip(stats.pids.tolist(), stats.n_steps.tolist()))
    assert by_pid == {10: 2, 11: 3, 12: 5}  # 13 dropped (n_k = 1 < 2)


def test_empty_inputs():
    empty_df = pd.DataFrame({"ID": [], "X": []})
    assert len(extract_track_stats({}, empty_df)) == 0
    assert len(extract_track_stats({1: (np.array([1]), np.array([1.0]))}, empty_df)) == 0
