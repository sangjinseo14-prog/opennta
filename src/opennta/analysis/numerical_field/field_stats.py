"""Descriptive statistics for numerical velocity-field components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ComponentStats:
    """min / max / mean / sd of one velocity-field component (NaNs ignored)."""

    min: float
    max: float
    mean: float
    std: float


def component_stats(values: NDArray[np.floating]) -> ComponentStats | None:
    """Summarise the finite entries of ``values``.

    Empty grid cells are stored as NaN, so the summary is taken over the finite
    entries only. Returns ``None`` when nothing finite is present (an uncomputed
    or fully empty field) so callers can render a placeholder instead of NaNs.
    The standard deviation is the population sd (``ddof=0``) of the cell values.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return ComponentStats(
        min=float(finite.min()),
        max=float(finite.max()),
        mean=float(finite.mean()),
        std=float(finite.std()),
    )


__all__ = ["ComponentStats", "component_stats"]
