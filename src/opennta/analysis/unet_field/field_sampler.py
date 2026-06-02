from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _nearest_index(coords: NDArray[np.floating], q: NDArray[np.floating]) -> NDArray[np.intp]:
    # coords must be sorted ascending (reference grid axis is monotonic).
    idx = np.searchsorted(coords, q)
    idx = np.clip(idx, 1, len(coords) - 1)
    left = coords[idx - 1]
    right = coords[idx]
    return np.where(np.abs(q - left) <= np.abs(q - right), idx - 1, idx)


class FieldSampler:
    # Samples the U-Net's 128x128 reference-grid prediction at spot positions:
    # raw-pixel (x, y) -> reference coords -> nearest grid node.
    # Field values are expected in um/frame (AC convention), matching
    # df["dX_um"], so apply() can subtract them directly.
    # Kept local to UNetField so this subpackage does not depend on another
    # corrector's subpackage.

    def __init__(
        self,
        u: NDArray[np.floating],
        v: NDArray[np.floating],
        xs_ref: NDArray[np.floating],
        ys_ref: NDArray[np.floating],
        tform: dict,
    ):
        if u.shape != v.shape:
            raise ValueError(f"u/v shape mismatch: u={u.shape}, v={v.shape}")
        # Deferred so importing this module stays light (no cv2/TF pulled in);
        # by the time a sampler is built, inference has already loaded the bundle.
        from .unet_bundle.coordinate_transforms import transform_positions_to_reference

        self.u = np.asarray(u, dtype=np.float32)
        self.v = np.asarray(v, dtype=np.float32)
        self.xs_ref = np.asarray(xs_ref, dtype=np.float32)
        self.ys_ref = np.asarray(ys_ref, dtype=np.float32)
        self.tform = tform
        self._transform = transform_positions_to_reference

    def sample(
        self,
        x: pd.Series | NDArray,
        y: pd.Series | NDArray,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        rx, ry = self._transform(x, y, self.tform)
        ix = _nearest_index(self.xs_ref, rx)
        iy = _nearest_index(self.ys_ref, ry)
        u_at = self.u[iy, ix]
        v_at = self.v[iy, ix]
        u_at = np.where(np.isfinite(u_at), u_at, 0.0)
        v_at = np.where(np.isfinite(v_at), v_at, 0.0)
        return u_at, v_at
