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
    """min/max/mean/sd over the finite entries of ``values``.

    Empty grid cells are NaN, so non-finite entries are dropped; an all-NaN or
    empty field yields ``None`` so callers can show a placeholder instead of
    NaNs. ``std`` is the population sd (``ddof=0``).
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
