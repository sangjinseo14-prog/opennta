import numpy as np

from .config import (
    ACTUAL_IMAGE_HEIGHT_PX,
    ACTUAL_IMAGE_WIDTH_PX,
    CLIP_POS_TO_REFERENCE_DOMAIN,
    FLIP_Y,
    MANUAL_X_OFFSET,
    MANUAL_X_SCALE,
    MANUAL_Y_OFFSET,
    MANUAL_Y_SCALE,
    PIXEL_SIZE_X_METERS,
    PIXEL_SIZE_Y_METERS,
    POSITION_ALIGN_MODE,
    SWAP_XY,
)


def build_position_transform_from_data(raw_df, xs_ref, ys_ref):
    x0 = float(raw_df["x"].min())
    x1 = float(raw_df["x"].max())
    y0 = float(raw_df["y"].min())
    y1 = float(raw_df["y"].max())
    xr0 = float(np.min(xs_ref))
    xr1 = float(np.max(xs_ref))
    yr0 = float(np.min(ys_ref))
    yr1 = float(np.max(ys_ref))
    sx = (xr1 - xr0) / max(x1 - x0, 1e-12)
    sy = (yr1 - yr0) / max(y1 - y0, 1e-12)
    return {
        "mode": "fit_to_reference_extent",
        "raw_x_min": x0,
        "raw_x_max": x1,
        "raw_y_min": y0,
        "raw_y_max": y1,
        "ref_x_min": xr0,
        "ref_x_max": xr1,
        "ref_y_min": yr0,
        "ref_y_max": yr1,
        "sx_pos": sx,
        "sy_pos": sy,
    }


def build_physical_position_transform(xs_ref, ys_ref, meta, image_width_px=None, image_height_px=None):
    ref_x_min = float(np.min(xs_ref))
    ref_x_max = float(np.max(xs_ref))
    ref_y_min = float(np.min(ys_ref))
    ref_y_max = float(np.max(ys_ref))

    pixel_size_x_m = float(PIXEL_SIZE_X_METERS)
    pixel_size_y_m = float(PIXEL_SIZE_Y_METERS)
    raw_width_px = int(ACTUAL_IMAGE_WIDTH_PX if image_width_px is None else image_width_px)
    raw_height_px = int(ACTUAL_IMAGE_HEIGHT_PX if image_height_px is None else image_height_px)

    actual_width_m = float((raw_width_px - 1) * pixel_size_x_m)
    actual_height_m = float((raw_height_px - 1) * pixel_size_y_m)
    ref_width_m = ref_x_max - ref_x_min
    ref_height_m = ref_y_max - ref_y_min

    return {
        "mode": "physical_meta",
        "raw_x_min": 0.0,
        "raw_x_max": float(raw_width_px - 1),
        "raw_y_min": 0.0,
        "raw_y_max": float(raw_height_px - 1),
        "ref_x_min": ref_x_min,
        "ref_x_max": ref_x_max,
        "ref_y_min": ref_y_min,
        "ref_y_max": ref_y_max,
        "sx_pos": ref_width_m / max(actual_width_m, 1e-12),
        "sy_pos": ref_height_m / max(actual_height_m, 1e-12),
        "pixel_size_x_m": pixel_size_x_m,
        "pixel_size_y_m": pixel_size_y_m,
        "actual_width_px": raw_width_px,
        "actual_height_px": raw_height_px,
        "actual_width_m": actual_width_m,
        "actual_height_m": actual_height_m,
        "simulation_L_m": float(meta.get("L", ref_width_m)),
        "simulation_dt_s": float(meta.get("dt", 0.0)),
        "simulation_save_every": int(meta.get("saveEvery", 1)),
    }


def transform_positions_to_reference(x_px, y_px, tform):
    x = np.asarray(x_px, dtype=np.float32)
    y = np.asarray(y_px, dtype=np.float32)
    if SWAP_XY:
        x, y = y.copy(), x.copy()

    if tform.get("mode") == "physical_meta":
        x = tform["ref_x_min"] + x * tform["pixel_size_x_m"] * tform["sx_pos"]
        if FLIP_Y:
            y = tform["ref_y_min"] + (tform["actual_height_px"] - 1 - y) * tform["pixel_size_y_m"] * tform["sy_pos"]
        else:
            y = tform["ref_y_min"] + y * tform["pixel_size_y_m"] * tform["sy_pos"]
    elif POSITION_ALIGN_MODE == "fit_to_reference_extent":
        x = (x - tform["raw_x_min"]) * tform["sx_pos"] + tform["ref_x_min"]
        if FLIP_Y:
            y = (tform["raw_y_max"] - y) * tform["sy_pos"] + tform["ref_y_min"]
        else:
            y = (y - tform["raw_y_min"]) * tform["sy_pos"] + tform["ref_y_min"]
    else:
        x = x * float(MANUAL_X_SCALE) + float(MANUAL_X_OFFSET)
        y = y * float(MANUAL_Y_SCALE) + float(MANUAL_Y_OFFSET)
        if FLIP_Y:
            y = -y

    if CLIP_POS_TO_REFERENCE_DOMAIN:
        x = np.clip(x, tform["ref_x_min"], tform["ref_x_max"])
        y = np.clip(y, tform["ref_y_min"], tform["ref_y_max"])

    return x.astype(np.float32), y.astype(np.float32)


def inverse_transform_positions_from_reference(x_ref, y_ref, tform):
    x_ref = np.asarray(x_ref, dtype=np.float32)
    y_ref = np.asarray(y_ref, dtype=np.float32)

    if tform.get("mode") == "physical_meta":
        x = (x_ref - tform["ref_x_min"]) / max(tform["pixel_size_x_m"] * tform["sx_pos"], 1e-12)
        if FLIP_Y:
            y = (tform["actual_height_px"] - 1) - (y_ref - tform["ref_y_min"]) / max(tform["pixel_size_y_m"] * tform["sy_pos"], 1e-12)
        else:
            y = (y_ref - tform["ref_y_min"]) / max(tform["pixel_size_y_m"] * tform["sy_pos"], 1e-12)
    elif POSITION_ALIGN_MODE == "fit_to_reference_extent":
        x = (x_ref - tform["ref_x_min"]) / max(tform["sx_pos"], 1e-12) + tform["raw_x_min"]
        if FLIP_Y:
            y = tform["raw_y_max"] - (y_ref - tform["ref_y_min"]) / max(tform["sy_pos"], 1e-12)
        else:
            y = (y_ref - tform["ref_y_min"]) / max(tform["sy_pos"], 1e-12) + tform["raw_y_min"]
    else:
        x = (x_ref - float(MANUAL_X_OFFSET)) / max(float(MANUAL_X_SCALE), 1e-12)
        if FLIP_Y:
            y = -(y_ref - float(MANUAL_Y_OFFSET)) / max(float(MANUAL_Y_SCALE), 1e-12)
        else:
            y = (y_ref - float(MANUAL_Y_OFFSET)) / max(float(MANUAL_Y_SCALE), 1e-12)

    if SWAP_XY:
        x, y = y.copy(), x.copy()

    return x.astype(np.float32), y.astype(np.float32)
