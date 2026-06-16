from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def se_weighted_gaussian_smooth(
    mean_dx: NDArray[np.floating],
    mean_dy: NDArray[np.floating],
    se_vec: NDArray[np.floating],
    count: NDArray[np.floating],
    min_count: int = 5,
    eps: float = 1e-12,
    ksize: int = 3,
    n_iter: int = 2,
    sigma: float | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # SE-weighted Gaussian spatial smoothing.
    #
    # At every iteration each cell is replaced by a weighted average of its
    # neighbours within a ksize x ksize window. Weight at offset (dj, di) is
    #     G(dj, di) * 1 / (se^2 + eps)  for valid cells, else 0,
    # where G is an isotropic 2D Gaussian of width sigma and se is the standard
    # error of the cell's mean velocity. Since se^2 is the variance of that
    # mean, 1 / se^2 is inverse-variance (precision) weighting.
    #
    # Invalid cells (count < min_count, count < 2, non-finite or non-positive
    # se) carry zero reliability weight and are filled in from valid
    # neighbours (diffusion-like). Cells whose denominator is zero in a given
    # iteration (no valid neighbour inside the kernel) are left unchanged.
    #
    # sigma defaults to max(ksize / 6.0, 0.5) so the kernel radius is ~3 sigma.
    ksize = max(1, int(ksize))
    if ksize % 2 == 0:
        ksize += 1
    radius = ksize // 2

    if sigma is None:
        sigma = max(ksize / 6.0, 0.5)
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    offsets = np.arange(-radius, radius + 1, dtype=float)
    dj_grid, di_grid = np.meshgrid(offsets, offsets, indexing="ij")
    gauss_kernel = np.exp(-(dj_grid * dj_grid + di_grid * di_grid)
                          / (2.0 * sigma * sigma))

    pass_N = count >= int(min_count)
    pass_se = (count >= 2) & np.isfinite(se_vec) & (se_vec > 0)
    pass_both = pass_N & pass_se

    w = np.zeros_like(se_vec, dtype=float)
    w[pass_both] = 1.0 / (se_vec[pass_both] ** 2 + eps)

    u_cur = np.where(pass_both & np.isfinite(mean_dx), mean_dx, 0.0).astype(float)
    v_cur = np.where(pass_both & np.isfinite(mean_dy), mean_dy, 0.0).astype(float)

    h, wdim = u_cur.shape

    # Per-offset loop instead of scipy.ndimage.gaussian_filter so each
    # contribution can be weighted by per-cell reliability w[src]; a separable
    # convolution would mix valid and invalid cells equally.
    for _ in range(max(int(n_iter), 1)):
        num_u = np.zeros_like(u_cur)
        num_v = np.zeros_like(v_cur)
        den = np.zeros_like(w)

        for dj in range(-radius, radius + 1):
            if dj >= 0:
                dst_r0, dst_r1 = 0, h - dj
                src_r0, src_r1 = dj, h
            else:
                dst_r0, dst_r1 = -dj, h
                src_r0, src_r1 = 0, h + dj

            for di in range(-radius, radius + 1):
                if di >= 0:
                    dst_c0, dst_c1 = 0, wdim - di
                    src_c0, src_c1 = di, wdim
                else:
                    dst_c0, dst_c1 = -di, wdim
                    src_c0, src_c1 = 0, wdim + di

                g = gauss_kernel[dj + radius, di + radius]

                w_nb = w[src_r0:src_r1, src_c0:src_c1]
                u_nb = u_cur[src_r0:src_r1, src_c0:src_c1]
                v_nb = v_cur[src_r0:src_r1, src_c0:src_c1]

                contrib = g * w_nb  # Gaussian * inverse-variance (1/se^2) weight
                num_u[dst_r0:dst_r1, dst_c0:dst_c1] += contrib * u_nb
                num_v[dst_r0:dst_r1, dst_c0:dst_c1] += contrib * v_nb
                den[dst_r0:dst_r1, dst_c0:dst_c1] += contrib

        upd = den > 0
        u_new, v_new = u_cur.copy(), v_cur.copy()
        u_new[upd] = num_u[upd] / den[upd]
        v_new[upd] = num_v[upd] / den[upd]
        u_cur, v_cur = u_new, v_new

    return u_cur, v_cur
