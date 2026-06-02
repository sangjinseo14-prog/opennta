from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .node_field import nearest_sample


class FieldSampler:

    def __init__(
        self,
        u: NDArray[np.floating],
        v: NDArray[np.floating],
        xs: NDArray[np.floating],
        ys: NDArray[np.floating],
    ):
        self.u = np.asarray(u, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)

    def sample(
        self,
        x: pd.Series | NDArray,
        y: pd.Series | NDArray,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return nearest_sample(self.u, self.v, self.xs, self.ys, x, y)
