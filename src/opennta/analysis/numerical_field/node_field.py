"""Map a window-center velocity field onto a regular node grid.

``compute_velocity_field`` gives one (mean_dx, mean_dy) per window center on an
``n x n`` grid. When ``node_count == n_windows`` the nodes are those window
centers (identity map). Otherwise nodes are cell centers spanning
``[x_min, x_max]`` in ``node_count`` equal cells (``node_i = (i + 0.5) * W/N``),
matching the U-Net ``resample_to_cell_centers`` grid: the window field is
bilinearly interpolated onto them, and nodes beyond the outermost window centers
are linearly extrapolated from the nearest 2x2 block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NodeField:
    # Velocity at grid nodes, um/frame (matching df["dX_um"]). xs are column
    # (x) node positions, ys are row (y) node positions, both in pixels.
    u: NDArray[np.floating]
    v: NDArray[np.floating]
    xs: NDArray[np.floating]
    ys: NDArray[np.floating]


def bilinear_sample(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    xs: NDArray[np.floating],
    ys: NDArray[np.floating],
    qx: NDArray | object,
    qy: NDArray | object,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # Bilinear lookup of (u, v) defined on the grid ys (rows) x xs (cols) at
    # scattered query points (qx, qy). Queries beyond the grid extent are
    # linearly extrapolated from the nearest 2x2 cell (tx/ty fall outside
    # [0, 1]); NaN grid values are treated as 0 (no drift), so empty windows
    # contribute nothing.
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

    def _interp(a: NDArray[np.floating]) -> NDArray[np.floating]:
        a = np.where(np.isfinite(a), a, 0.0)
        return (
            a[iy, ix] * (1.0 - tx) * (1.0 - ty)
            + a[iy, ix + 1] * tx * (1.0 - ty)
            + a[iy + 1, ix] * (1.0 - tx) * ty
            + a[iy + 1, ix + 1] * tx * ty
        )

    return _interp(np.asarray(u, dtype=float)), _interp(np.asarray(v, dtype=float))


def nearest_sample(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    xs: NDArray[np.floating],
    ys: NDArray[np.floating],
    qx: NDArray | object,
    qy: NDArray | object,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # Nearest-node lookup of (u, v) defined on the grid ys (rows) x xs (cols) at
    # scattered query points (qx, qy): each query takes the value of the closest
    # node. Queries are clamped to the grid extent (edge replicate) and NaN node
    # values are treated as 0 (no drift), matching bilinear_sample's NaN handling.
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    qx = np.clip(np.asarray(qx, dtype=float), xs[0], xs[-1])
    qy = np.clip(np.asarray(qy, dtype=float), ys[0], ys[-1])

    ix = _nearest_index(xs, qx)
    iy = _nearest_index(ys, qy)

    def _pick(a: NDArray[np.floating]) -> NDArray[np.floating]:
        a = np.where(np.isfinite(a), a, 0.0)
        return a[iy, ix]

    return _pick(np.asarray(u, dtype=float)), _pick(np.asarray(v, dtype=float))


def _nearest_index(grid: NDArray[np.floating], q: NDArray[np.floating]) -> NDArray[np.intp]:
    # Index of the nearest node per query. grid is sorted ascending and q is
    # assumed clamped to [grid[0], grid[-1]]; ties resolve to the lower index.
    idx = np.clip(np.searchsorted(grid, q), 1, grid.size - 1)
    left = grid[idx - 1]
    right = grid[idx]
    return np.where((q - left) <= (right - q), idx - 1, idx)


def build_node_field(
    u_win: NDArray[np.floating],
    v_win: NDArray[np.floating],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    node_count: int,
) -> NodeField:
    # u_win/v_win are window-center values (n x n) in um/frame. Interpolate them
    # onto an (N x N) node grid where N = node_count.
    n = int(np.asarray(u_win).shape[0])
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])

    ww = (x_max - x_min) / n
    wh = (y_max - y_min) / n
    xc = x_min + (np.arange(n) + 0.5) * ww
    yc = y_min + (np.arange(n) + 0.5) * wh

    N = max(int(node_count), 2)
    if N == n:
        # No interpolation: nodes are the window centers (identity map).
        xs, ys = xc, yc
    else:
        # N cell-center nodes over [x_min, x_max], each owning a W/N-pixel cell
        # (matches U-Net resample_to_cell_centers). Nodes outside the outermost
        # window centers are extrapolated by bilinear_sample.
        xs = x_min + (np.arange(N) + 0.5) * (x_max - x_min) / N
        ys = y_min + (np.arange(N) + 0.5) * (y_max - y_min) / N

    qx, qy = np.meshgrid(xs, ys)
    u_node, v_node = bilinear_sample(u_win, v_win, xc, yc, qx.reshape(-1), qy.reshape(-1))
    return NodeField(
        u=u_node.reshape(N, N),
        v=v_node.reshape(N, N),
        xs=xs,
        ys=ys,
    )


__all__ = ["NodeField", "bilinear_sample", "build_node_field", "nearest_sample"]
