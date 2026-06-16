"""Shared numerical-field value types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FieldStats:
    n_windows: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    count: NDArray[np.floating]
    mean_dx: NDArray[np.floating]
    mean_dy: NDArray[np.floating]
    std_dx: NDArray[np.floating]
    std_dy: NDArray[np.floating]
    sigma_vec: NDArray[np.floating]
    se_vec: NDArray[np.floating]


@dataclass
class NumericalFieldParams:
    # Captured from the dialog once and reused across batch files.
    n_windows: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    outlier_k: float
    min_count: int
    ksize: int
    n_iter: int
    sigma: float = 1.0
    # Node grid resolution for the interpolated correction field. None means
    # "auto" = n_windows (the window centers; 10 windows -> 10 nodes).
    node_count: int | None = None

    def resolved_node_count(self) -> int:
        if self.node_count is not None and self.node_count >= 2:
            return int(self.node_count)
        return int(self.n_windows)


__all__ = ["FieldStats", "NumericalFieldParams"]
