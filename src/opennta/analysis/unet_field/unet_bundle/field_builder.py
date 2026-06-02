"""Build U-Net input samples and post-process predictions into field outputs."""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .config import (
    COARSE_SIGMA,
    FLIP_Y,
    FRAME_STRIDE,
    INPUT_MODE,
    LINE_THICKNESS,
    MAX_POINTS_PER_FILE,
    N_PARTICLES_TARGET,
    PIXEL_SIZE_X_METERS,
    PIXEL_SIZE_Y_METERS,
    SWAP_XY,
)
from .coordinate_transforms import inverse_transform_positions_from_reference


def convert_vector_sc_to_ac(u_sc, v_sc):
    u_ac = np.asarray(u_sc, dtype=np.float32).copy()
    v_ac = np.asarray(v_sc, dtype=np.float32).copy()
    if SWAP_XY:
        u_ac, v_ac = v_ac.copy(), u_ac.copy()
    if FLIP_Y:
        v_ac = -v_ac
    return u_ac.astype(np.float32), v_ac.astype(np.float32)


def normalize_uv(U, V, norm):
    Un = (U - norm["u_mean"]) / norm["u_std"]
    Vn = (V - norm["v_mean"]) / norm["v_std"]
    return Un.astype(np.float32), Vn.astype(np.float32)


def denormalize_uv(Un, Vn, norm):
    U = Un * norm["u_std"] + norm["u_mean"]
    V = Vn * norm["v_std"] + norm["v_mean"]
    return U.astype(np.float32), V.astype(np.float32)


def normalize_uv_sparse_masked(U, V, mask, norm):
    Un = np.where(mask > 0, (U - norm["u_mean"]) / norm["u_std"], 0.0)
    Vn = np.where(mask > 0, (V - norm["v_mean"]) / norm["v_std"], 0.0)
    return Un.astype(np.float32), Vn.astype(np.float32)


def _xy_to_nearest_grid_index(qx, qy, xs, ys):
    ix = np.abs(xs - qx).argmin()
    iy = np.abs(ys - qy).argmin()
    return int(ix), int(iy)


def line_rasterize_segments_to_grid(df, xs, ys, thickness=LINE_THICKNESS):
    H = len(ys)
    W = len(xs)
    Uacc = np.zeros((H, W), dtype=np.float32)
    Vacc = np.zeros((H, W), dtype=np.float32)
    Cacc = np.zeros((H, W), dtype=np.float32)

    for _, g in df.groupby("particle_id"):
        g = g.sort_values("frame_idx")
        qx = g["ma_x"].to_numpy(dtype=np.float32)
        qy = g["ma_y"].to_numpy(dtype=np.float32)
        qu = g["u"].to_numpy(dtype=np.float32)
        qv = g["v"].to_numpy(dtype=np.float32)
        fr = g["frame_idx"].to_numpy(dtype=np.int32)
        if len(g) < 2:
            continue
        for k in range(len(g) - 1):
            if fr[k + 1] != fr[k] + 1:
                continue
            x0_idx, y0_idx = _xy_to_nearest_grid_index(qx[k], qy[k], xs, ys)
            x1_idx, y1_idx = _xy_to_nearest_grid_index(qx[k + 1], qy[k + 1], xs, ys)
            u_seg = float(0.5 * (qu[k] + qu[k + 1]))
            v_seg = float(0.5 * (qv[k] + qv[k + 1]))
            line_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.line(line_mask, (x0_idx, y0_idx), (x1_idx, y1_idx), color=1, thickness=thickness, lineType=cv2.LINE_8)
            idx = line_mask > 0
            if np.any(idx):
                Uacc[idx] += u_seg
                Vacc[idx] += v_seg
                Cacc[idx] += 1.0

    mask = (Cacc > 0).astype(np.float32)
    Usp = np.where(mask > 0, Uacc / np.maximum(Cacc, 1e-8), 0.0).astype(np.float32)
    Vsp = np.where(mask > 0, Vacc / np.maximum(Cacc, 1e-8), 0.0).astype(np.float32)
    count_norm = np.log1p(Cacc)
    count_norm = count_norm / (count_norm.max() + 1e-8)
    return Usp, Vsp, mask.astype(np.float32), count_norm.astype(np.float32)


def build_dual_coordinate_prediction(pred_u_sc, pred_v_sc, xs_ref, ys_ref, tform):
    pred_u_sc = np.asarray(pred_u_sc, dtype=np.float32)
    pred_v_sc = np.asarray(pred_v_sc, dtype=np.float32)
    pred_u_ac, pred_v_ac = convert_vector_sc_to_ac(pred_u_sc, pred_v_sc)

    xs_sc = np.asarray(xs_ref, dtype=np.float32)
    ys_sc = np.asarray(ys_ref, dtype=np.float32)
    xs_ac_px, _ = inverse_transform_positions_from_reference(xs_sc, np.full_like(xs_sc, ys_sc.min()), tform)
    _, ys_ac_px = inverse_transform_positions_from_reference(np.full_like(ys_sc, xs_sc.min()), ys_sc, tform)
    xs_ac = xs_ac_px * float(PIXEL_SIZE_X_METERS)
    ys_ac = ys_ac_px * float(PIXEL_SIZE_Y_METERS)

    return {
        "pred_u_sc": pred_u_sc.astype(np.float32),
        "pred_v_sc": pred_v_sc.astype(np.float32),
        "pred_speed_sc": np.sqrt(pred_u_sc ** 2 + pred_v_sc ** 2).astype(np.float32),
        "pred_u_ac": pred_u_ac.astype(np.float32),
        "pred_v_ac": pred_v_ac.astype(np.float32),
        "pred_speed_ac": np.sqrt(pred_u_ac ** 2 + pred_v_ac ** 2).astype(np.float32),
        "xs_sc": xs_sc.astype(np.float32),
        "ys_sc": ys_sc.astype(np.float32),
        "xs_ac": xs_ac.astype(np.float32),
        "ys_ac": ys_ac.astype(np.float32),
        "xs_ac_px": xs_ac_px.astype(np.float32),
        "ys_ac_px": ys_ac_px.astype(np.float32),
    }


def _make_masked_gaussian_coarse(Usp, Vsp, mask, sigma=COARSE_SIGMA):
    mask = mask.astype(np.float32)
    Um = Usp * mask
    Vm = Vsp * mask
    denom = gaussian_filter(mask, sigma=sigma)
    denom = np.maximum(denom, 1e-6)
    Ucoarse = gaussian_filter(Um, sigma=sigma) / denom
    Vcoarse = gaussian_filter(Vm, sigma=sigma) / denom
    return Ucoarse.astype(np.float32), Vcoarse.astype(np.float32)


def build_actual_sample_from_flow_df(flow_df, xs, ys, norm, seed=40):
    df = flow_df.copy()
    if FRAME_STRIDE is not None and FRAME_STRIDE > 1:
        df = df[(df["frame_idx"] % int(FRAME_STRIDE)) == 0].copy()
    if MAX_POINTS_PER_FILE is not None and len(df) > int(MAX_POINTS_PER_FILE):
        df = df.sample(n=int(MAX_POINTS_PER_FILE), random_state=seed).copy()
    if N_PARTICLES_TARGET is not None:
        pids = df["particle_id"].unique()
        if len(pids) > int(N_PARTICLES_TARGET):
            rng = np.random.default_rng(seed)
            chosen = rng.choice(pids, size=int(N_PARTICLES_TARGET), replace=False)
            df = df[df["particle_id"].isin(chosen)].copy()

    Usp, Vsp, mask, count_norm = line_rasterize_segments_to_grid(df, xs, ys, thickness=LINE_THICKNESS)
    Ucoarse, Vcoarse = _make_masked_gaussian_coarse(Usp, Vsp, mask)
    Usp_n, Vsp_n = normalize_uv_sparse_masked(Usp, Vsp, mask, norm)
    if INPUT_MODE == "3ch":
        X = np.stack([Usp_n, Vsp_n, mask], axis=-1).astype(np.float32)
    else:
        Ucoarse_n, Vcoarse_n = normalize_uv(Ucoarse, Vcoarse, norm)
        X = np.stack([Usp_n, Vsp_n, mask, Ucoarse_n, Vcoarse_n], axis=-1).astype(np.float32)
    return {
        "X": X,
        "Usp": Usp,
        "Vsp": Vsp,
        "mask": mask,
        "count_norm": count_norm,
        "Ucoarse": Ucoarse,
        "Vcoarse": Vcoarse,
        "flow_df_used": df,
        "traj_xy": df[["ma_x", "ma_y"]].to_numpy(dtype=np.float32),
    }
