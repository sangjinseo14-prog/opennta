"""Unit bridge between the U-Net bundle field and opennta drift correction.

Bundle field: velocity in m/s, calibrated at the fixed bundle frame rate
    (ACTUAL_FPS = 25) and ACTUAL_PIXEL_SIZE_UM.
opennta:      per-frame displacement in um/frame at the user's fps, subtracted
    from dX_um / dY_um.
Correspondence:
    um/frame = m/s * 1e6 / fps * (fps / bundle_fps) = m/s * 1e6 / bundle_fps
The (fps / bundle_fps) factor restores physical scale because the field is
calibrated at bundle_fps (inference always computes velocities at bundle DT_OBS).
"""
from __future__ import annotations

import numpy as np

from .field_sampler import FieldSampler
from .unet_bundle.coordinate_transforms import transform_positions_to_reference


def _bilinear_sample(u, v, xs, ys, qx, qy):
    # Local copy of numerical_field.node_field.bilinear_sample so this
    # subpackage stays independent of numerical_field.
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)

    ix = np.clip(np.searchsorted(xs, qx) - 1, 0, xs.size - 2)
    iy = np.clip(np.searchsorted(ys, qy) - 1, 0, ys.size - 2)

    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    tx = np.where(x1 != x0, (qx - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 != y0, (qy - y0) / (y1 - y0), 0.0)

    def _interp(a):
        a = np.where(np.isfinite(a), a, 0.0)
        return (
            a[iy, ix] * (1.0 - tx) * (1.0 - ty)
            + a[iy, ix + 1] * tx * (1.0 - ty)
            + a[iy + 1, ix] * (1.0 - tx) * ty
            + a[iy + 1, ix + 1] * tx * ty
        )

    return _interp(np.asarray(u, dtype=float)), _interp(np.asarray(v, dtype=float))


def resample_to_cell_centers(u, v, xs_ref, ys_ref, tform):
    # Relocate the N x N field from its native grid (endpoints fixed at 0 and
    # W-1, 127 equal divisions) onto numerical-style cell-center nodes:
    # N equal-area cells over [0, W] with node_i = (i + 0.5) * W / N. The
    # coordinate transform itself is unchanged; nodes are moved purely by
    # bilinear re-interpolation. Returns the resampled field plus the new
    # reference-space grids (ascending) for the sampler.
    xs_ref = np.asarray(xs_ref, dtype=float)
    ys_ref = np.asarray(ys_ref, dtype=float)
    n = xs_ref.shape[0]
    from .unet_bundle.config import ACTUAL_IMAGE_HEIGHT_PX, ACTUAL_IMAGE_WIDTH_PX

    w = float(tform.get("actual_width_px", ACTUAL_IMAGE_WIDTH_PX))
    h = float(tform.get("actual_height_px", ACTUAL_IMAGE_HEIGHT_PX))

    new_xs_px = (np.arange(n) + 0.5) * w / n
    new_ys_px = (np.arange(n) + 0.5) * h / n
    # Map cell-center pixels -> reference (SWAP_XY=False, so per-axis calls with
    # a dummy other axis are safe; cell centers are interior so the clip is a
    # no-op). FLIP_Y makes the y mapping descending, so sort to ascending.
    new_xs_ref, _ = transform_positions_to_reference(new_xs_px, np.zeros_like(new_xs_px), tform)
    _, new_ys_ref = transform_positions_to_reference(np.zeros_like(new_ys_px), new_ys_px, tform)
    new_xs_ref = np.sort(np.asarray(new_xs_ref, dtype=float))
    new_ys_ref = np.sort(np.asarray(new_ys_ref, dtype=float))

    qx, qy = np.meshgrid(new_xs_ref, new_ys_ref)
    u2, v2 = _bilinear_sample(u, v, xs_ref, ys_ref, qx.ravel(), qy.ravel())
    return u2.reshape(n, n), v2.reshape(n, n), new_xs_ref, new_ys_ref


def m_per_s_to_um_per_frame(v_m_per_s, fps, bundle_fps=None):
    # v [um/frame] = v [m/s] * 1e6 / fps.
    # The model's velocity field is calibrated against bundle_fps, so when
    # the user's fps differs scale by (fps / bundle_fps) to restore physical
    # correctness (e.g. fps=50, bundle_fps=25 -> x2).
    v = np.asarray(v_m_per_s, dtype=np.float32) * (1e6 / float(fps))
    if bundle_fps is not None and abs(float(fps) - float(bundle_fps)) > 1e-3:
        v = v * (float(fps) / float(bundle_fps))
    return v


def check_config_compatibility(opennta_config, bundle_pixel_size_m, bundle_fps):
    # Returns (ok, warning_message). Mismatches are warned but not blocked:
    # pixel-size is not auto-corrected, fps is auto-scaled by the caller via
    # (config_fps / bundle_fps).
    px_um_opennta = float(opennta_config.pixel_size)
    fps_opennta = float(opennta_config.fps)
    px_um_bundle = float(bundle_pixel_size_m) * 1e6
    fps_bundle = float(bundle_fps)

    msgs = []
    if abs(px_um_opennta - px_um_bundle) > 1e-4:
        msgs.append(
            f"pixel size mismatch: opennta={px_um_opennta:.4f} um, "
            f"model trained for {px_um_bundle:.4f} um"
        )
    if abs(fps_opennta - fps_bundle) > 1e-3:
        ratio = fps_opennta / fps_bundle
        msgs.append(
            f"fps mismatch: opennta={fps_opennta:g}, "
            f"model trained for {fps_bundle:g} — velocity field will be "
            f"auto-scaled by x{ratio:g} (config_fps / bundle_fps)"
        )
    if msgs:
        return False, "; ".join(msgs)
    return True, ""


def build_field_sampler(pred_u_ac_m_per_s, pred_v_ac_m_per_s, xs_ref, ys_ref, tform, fps, bundle_fps):
    # pred_*_ac is the 128x128 U-Net field (AC convention, m/s). Convert to
    # um/frame once here so the sampler returns values that match dX_um / dY_um.
    u = m_per_s_to_um_per_frame(pred_u_ac_m_per_s, fps, bundle_fps=bundle_fps)
    v = m_per_s_to_um_per_frame(pred_v_ac_m_per_s, fps, bundle_fps=bundle_fps)
    u, v, xs_ref, ys_ref = resample_to_cell_centers(u, v, xs_ref, ys_ref, tform)
    return FieldSampler(u=u, v=v, xs_ref=xs_ref, ys_ref=ys_ref, tform=tform)
