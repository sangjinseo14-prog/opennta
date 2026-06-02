from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FLIP_Y,
    MIN_CONSEC_FRAMES,
    MIN_ROWS_AFTER_FILTER,
    OUTLIER_ACTION,
    PIXEL_SIZE_X_METERS,
    PIXEL_SIZE_Y_METERS,
    POSITION_ALIGN_MODE,
    SIM_META_PATH,
    STEP_MAD_K,
    SWAP_XY,
    UV_MAD_K,
    VELOCITY_UNIT_MODE,
)
from .coordinate_transforms import (
    build_physical_position_transform,
    build_position_transform_from_data,
    transform_positions_to_reference,
)
from .io import load_actual_spots_raw, load_simulation_meta_csv

_REQUIRED_SPOT_COLUMNS = ("particle_id", "x", "y", "frame_idx")


def centered_rolling_mean(a, n=9):
    n = max(int(n), 1)
    return pd.Series(a).rolling(window=n, min_periods=1, center=True).mean().to_numpy()


def median_and_mad_sigma(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return med, sigma


def _mad_threshold(values, k):
    # Returns the (med + k*sigma) cut-off, or None when not finite.
    med, sigma = median_and_mad_sigma(values)
    thr = med + float(k) * max(sigma, 1e-12)
    return thr if np.isfinite(thr) else None


def group_consecutive_frame_indices(frame_array):
    fr = np.asarray(frame_array, dtype=int)
    if fr.size == 0:
        return []
    runs = []
    start = 0
    for i in range(1, len(fr)):
        if fr[i] != fr[i - 1] + 1:
            runs.append(np.arange(start, i))
            start = i
    runs.append(np.arange(start, len(fr)))
    return runs


def iter_consecutive_frame_groups(g, col_fr="frame_idx", min_len=2):
    fr = g[col_fr].astype(int).to_numpy()
    runs = group_consecutive_frame_indices(fr)
    valid_runs = []
    for run_idx, run in enumerate(runs):
        if len(run) >= int(min_len):
            valid_runs.append((run_idx, g.iloc[run].copy()))
    return valid_runs


def central_difference_velocity(ma_x_px, ma_y_px, frames, dt_obs):
    ma_x_px = np.asarray(ma_x_px, dtype=float)
    ma_y_px = np.asarray(ma_y_px, dtype=float)
    frames = np.asarray(frames, dtype=int)
    if len(frames) < 2:
        return np.full(len(frames), np.nan), np.full(len(frames), np.nan)

    vx_scale = float(PIXEL_SIZE_X_METERS) if VELOCITY_UNIT_MODE == "pixel_size" else 1.0
    vy_scale = float(PIXEL_SIZE_Y_METERS) if VELOCITY_UNIT_MODE == "pixel_size" else 1.0

    if SWAP_XY:
        x_src = ma_y_px.copy()
        y_src = ma_x_px.copy()
    else:
        x_src = ma_x_px.copy()
        y_src = ma_y_px.copy()
    y_sign = -1.0 if FLIP_Y else 1.0

    def vel_at(i0, i1):
        dt = (frames[i1] - frames[i0]) * float(dt_obs)
        if dt <= 0:
            return np.nan, np.nan
        u = (x_src[i1] - x_src[i0]) * vx_scale / dt
        v = y_sign * (y_src[i1] - y_src[i0]) * vy_scale / dt
        return u, v

    u = np.full(len(frames), np.nan, dtype=float)
    v = np.full(len(frames), np.nan, dtype=float)
    u[0], v[0] = vel_at(0, 1)
    for i in range(1, len(frames) - 1):
        dt = (frames[i + 1] - frames[i - 1]) * float(dt_obs)
        if dt > 0:
            u[i] = (x_src[i + 1] - x_src[i - 1]) * vx_scale / dt
            v[i] = y_sign * (y_src[i + 1] - y_src[i - 1]) * vy_scale / dt
    u[-1], v[-1] = vel_at(len(frames) - 2, len(frames) - 1)
    return u, v


def _normalize_spots_df(df: pd.DataFrame) -> pd.DataFrame:
    # Accept either inference-style columns (particle_id, x, y, frame_idx)
    # or opennta-style columns (ID, X, Y, FRAME + dX_um etc). Output is the
    # 4-column raw_df equivalent to load_actual_spots_raw()'s output; extras
    # like dX_um and QUALITY are dropped.
    if df is None:
        raise ValueError("spots_df is None")

    d = df.copy()

    rename_map = {}
    if "ID" in d.columns and "particle_id" not in d.columns:
        rename_map["ID"] = "particle_id"
    if "X" in d.columns and "x" not in d.columns:
        rename_map["X"] = "x"
    if "Y" in d.columns and "y" not in d.columns:
        rename_map["Y"] = "y"
    if "FRAME" in d.columns and "frame_idx" not in d.columns:
        rename_map["FRAME"] = "frame_idx"
    if rename_map:
        d = d.rename(columns=rename_map)

    missing = [c for c in _REQUIRED_SPOT_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(
            f"spots_df is missing required columns: {missing}. "
            f"Expected either {_REQUIRED_SPOT_COLUMNS} "
            f"or opennta-style (ID, X, Y, FRAME)."
        )

    d = d[list(_REQUIRED_SPOT_COLUMNS)].copy()
    for c in _REQUIRED_SPOT_COLUMNS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=list(_REQUIRED_SPOT_COLUMNS)).copy()
    d["particle_id"] = d["particle_id"].astype(np.int64)
    d["frame_idx"] = d["frame_idx"].astype(np.int64)
    d["x"] = d["x"].astype(np.float32)
    d["y"] = d["y"].astype(np.float32)
    d = d.drop_duplicates(subset=["particle_id", "frame_idx"])
    return d.sort_values(["particle_id", "frame_idx"]).reset_index(drop=True)


def preprocess_actual_spots_to_flow(
    xs_ref,
    ys_ref,
    ma_window,
    dt_obs,
    csv_path=None,
    spots_df=None,
    image_width_px=None,
    image_height_px=None,
    min_consec_frames=MIN_CONSEC_FRAMES,
    min_rows_after_filter=MIN_ROWS_AFTER_FILTER,
    step_mad_k=STEP_MAD_K,
    uv_mad_k=UV_MAD_K,
    outlier_action=OUTLIER_ACTION,
):
    if spots_df is not None:
        raw_df = _normalize_spots_df(spots_df)
        source_tag = "<dataframe>"
    elif csv_path is not None:
        raw_df = load_actual_spots_raw(Path(csv_path))
        source_tag = str(csv_path)
    else:
        raise ValueError(
            "preprocess_actual_spots_to_flow requires either spots_df or csv_path"
        )

    sim_meta = load_simulation_meta_csv(SIM_META_PATH)
    if POSITION_ALIGN_MODE == "physical_meta":
        tform = build_physical_position_transform(
            xs_ref, ys_ref, sim_meta,
            image_width_px=image_width_px, image_height_px=image_height_px,
        )
    else:
        tform = build_position_transform_from_data(raw_df, xs_ref, ys_ref)
    particle_cache = {}
    eligible_pids = []
    next_pid = 0
    n_total_runs = 0
    n_kept_runs = 0
    n_dropped_short_runs = 0

    for pid, g in raw_df.groupby("particle_id"):
        g = g[["frame_idx", "x", "y"]].sort_values("frame_idx").copy()
        all_runs = group_consecutive_frame_indices(g["frame_idx"].to_numpy(dtype=np.int64))
        n_total_runs += len(all_runs)
        valid_runs = iter_consecutive_frame_groups(g, col_fr="frame_idx", min_len=min_consec_frames)
        n_dropped_short_runs += max(0, len(all_runs) - len(valid_runs))

        for run_local_idx, g_run in valid_runs:
            frames = g_run["frame_idx"].to_numpy(dtype=np.int64)
            raw_x_px = g_run["x"].to_numpy(dtype=np.float32)
            raw_y_px = g_run["y"].to_numpy(dtype=np.float32)
            ma_x_px = centered_rolling_mean(raw_x_px, n=ma_window).astype(np.float32)
            ma_y_px = centered_rolling_mean(raw_y_px, n=ma_window).astype(np.float32)
            raw_x_pos, raw_y_pos = transform_positions_to_reference(raw_x_px, raw_y_px, tform)
            ma_x_pos, ma_y_pos = transform_positions_to_reference(ma_x_px, ma_y_px, tform)
            u, v = central_difference_velocity(ma_x_px, ma_y_px, frames, dt_obs)

            run_pid = int(next_pid)
            next_pid += 1
            dfp = pd.DataFrame(
                {
                    "particle_id": np.full(len(frames), run_pid, dtype=np.int64),
                    "source_particle_id": np.full(len(frames), int(pid), dtype=np.int64),
                    "run_local_idx": np.full(len(frames), int(run_local_idx), dtype=np.int64),
                    "frame_idx": frames.astype(np.int64),
                    "raw_x_px": raw_x_px.astype(np.float32),
                    "raw_y_px": raw_y_px.astype(np.float32),
                    "ma_x_px": ma_x_px.astype(np.float32),
                    "ma_y_px": ma_y_px.astype(np.float32),
                    "raw_x": raw_x_pos.astype(np.float32),
                    "raw_y": raw_y_pos.astype(np.float32),
                    "ma_x": ma_x_pos.astype(np.float32),
                    "ma_y": ma_y_pos.astype(np.float32),
                    "u": np.asarray(u, dtype=np.float32),
                    "v": np.asarray(v, dtype=np.float32),
                }
            )
            dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna().copy()
            if len(dfp) < min_rows_after_filter:
                continue

            if step_mad_k is not None:
                dx = np.diff(dfp["ma_x"].to_numpy(dtype=np.float32))
                dy = np.diff(dfp["ma_y"].to_numpy(dtype=np.float32))
                step = np.hypot(dx, dy)
                step_thr = _mad_threshold(step, step_mad_k)
                if step.size > 0 and step_thr is not None:
                    keep = np.ones(len(dfp), dtype=bool)
                    keep[np.where(step > step_thr)[0] + 1] = False
                    dfp = dfp.loc[keep].copy()
                if len(dfp) < min_rows_after_filter:
                    continue

            if uv_mad_k is not None:
                u_arr = dfp["u"].to_numpy(dtype=np.float32)
                v_arr = dfp["v"].to_numpy(dtype=np.float32)
                speed = np.hypot(u_arr, v_arr)
                sp_thr = _mad_threshold(speed, uv_mad_k)
                if sp_thr is not None:
                    if outlier_action == "drop":
                        dfp = dfp.loc[speed <= sp_thr].copy()
                    elif outlier_action == "clip":
                        scale = np.minimum(1.0, sp_thr / np.maximum(speed, 1e-12))
                        dfp["u"] = u_arr * scale
                        dfp["v"] = v_arr * scale
                if len(dfp) < min_rows_after_filter:
                    continue

            particle_cache[run_pid] = dfp
            eligible_pids.append(run_pid)
            n_kept_runs += 1

    if not eligible_pids:
        raise ValueError(f"No eligible particles left after preprocess: {source_tag}")

    flow_df = pd.concat([particle_cache[pid] for pid in eligible_pids], axis=0, ignore_index=True)
    flow_df = flow_df.sort_values(["particle_id", "frame_idx"]).reset_index(drop=True)

    serializable_tform = {
        k: float(v) if isinstance(v, int | float | np.integer | np.floating) else v
        for k, v in tform.items()
    }

    meta = {
        "csv_path": source_tag,
        "position_transform": serializable_tform,
        "simulation_meta": sim_meta,
        "n_raw_rows": int(len(raw_df)),
        "n_raw_particles": int(raw_df["particle_id"].nunique()),
        "n_flow_rows": int(len(flow_df)),
        "n_flow_particles": int(flow_df["particle_id"].nunique()),
        "n_source_particles_used": int(flow_df["source_particle_id"].nunique()),
        "n_total_runs": int(n_total_runs),
        "n_kept_runs": int(n_kept_runs),
        "n_dropped_short_runs": int(n_dropped_short_runs),
        "frame_min": int(flow_df["frame_idx"].min()),
        "frame_max": int(flow_df["frame_idx"].max()),
    }
    return flow_df, meta
