from __future__ import annotations

import numpy as np
import pandas as pd

from .types import FieldStats, NumericalFieldParams


def compute_velocity_field(
    df: pd.DataFrame,
    n_windows: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    outlier_k: float = 4.0,
) -> FieldStats:
    # Requires df columns: ID, X, Y, FRAME, dX_um, dY_um.
    # mean_dx, mean_dy returned in um/frame (matching df["dX_um"]).
    xs = df["X"].to_numpy(dtype=float)
    ys = df["Y"].to_numpy(dtype=float)
    vx = df["dX_um"].to_numpy(dtype=float)
    vy = df["dY_um"].to_numpy(dtype=float)

    # 2D MAD outlier rejection (z-score radius outlier_k).
    medx = np.nanmedian(vx)
    medy = np.nanmedian(vy)
    madx = np.nanmedian(np.abs(vx - medx))
    mady = np.nanmedian(np.abs(vy - medy))
    sx = max(1.4826 * madx, 1e-12)
    sy = max(1.4826 * mady, 1e-12)
    zx = (vx - medx) / sx
    zy = (vy - medy) / sy
    r2 = zx * zx + zy * zy
    keep = np.isfinite(r2) & (r2 <= outlier_k * outlier_k)

    xs, ys, vx, vy = xs[keep], ys[keep], vx[keep], vy[keep]

    x_min, x_max = x_range
    y_min, y_max = y_range
    n = int(n_windows)
    ww = (x_max - x_min) / n
    wh = (y_max - y_min) / n

    if ww <= 0 or wh <= 0:
        raise ValueError("Invalid range: max must be greater than min.")

    ii = np.floor((xs - x_min) / ww).astype(int)
    jj = np.floor((ys - y_min) / wh).astype(int)
    valid = (ii >= 0) & (ii < n) & (jj >= 0) & (jj < n)
    ii, jj, vx, vy = ii[valid], jj[valid], vx[valid], vy[valid]

    count = np.zeros((n, n), dtype=float)
    sum_dx = np.zeros((n, n), dtype=float)
    sum_dy = np.zeros((n, n), dtype=float)
    sum_dx2 = np.zeros((n, n), dtype=float)
    sum_dy2 = np.zeros((n, n), dtype=float)

    np.add.at(count, (jj, ii), 1.0)
    np.add.at(sum_dx, (jj, ii), vx)
    np.add.at(sum_dy, (jj, ii), vy)
    np.add.at(sum_dx2, (jj, ii), vx * vx)
    np.add.at(sum_dy2, (jj, ii), vy * vy)

    mean_dx = np.full((n, n), np.nan)
    mean_dy = np.full((n, n), np.nan)
    std_dx = np.full((n, n), np.nan)
    std_dy = np.full((n, n), np.nan)

    ok1 = count > 0
    mean_dx[ok1] = sum_dx[ok1] / count[ok1]
    mean_dy[ok1] = sum_dy[ok1] / count[ok1]

    ok2 = count >= 2
    denom_safe = np.where(ok2, count - 1.0, 1.0)
    count_safe = np.where(ok1, count, 1.0)

    var_dx = (sum_dx2 - (sum_dx ** 2) / count_safe) / denom_safe
    var_dy = (sum_dy2 - (sum_dy ** 2) / count_safe) / denom_safe
    var_dx = np.where(ok2 & (var_dx >= 0), var_dx, np.nan)
    var_dy = np.where(ok2 & (var_dy >= 0), var_dy, np.nan)

    std_dx = np.sqrt(var_dx)
    std_dy = np.sqrt(var_dy)

    sigma_vec = np.sqrt(std_dx ** 2 + std_dy ** 2)
    se_vec = sigma_vec / np.sqrt(np.where(ok1, count, np.nan))
    ci95_vec = 1.96 * se_vec

    return FieldStats(
        n_windows=n,
        x_range=(float(x_min), float(x_max)),
        y_range=(float(y_min), float(y_max)),
        count=count,
        mean_dx=mean_dx,
        mean_dy=mean_dy,
        std_dx=std_dx,
        std_dy=std_dy,
        sigma_vec=sigma_vec,
        ci95_vec=ci95_vec,
    )


__all__ = ["FieldStats", "NumericalFieldParams", "compute_velocity_field"]
