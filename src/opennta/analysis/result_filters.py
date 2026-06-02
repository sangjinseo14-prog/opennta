"""Qt-free post-analysis filters applied to per-particle result maps."""

from __future__ import annotations

from typing import Any


def filter_by_r2(
    element_by_id: dict[int, Any],
    r2_by_id: dict[int, float],
    r2_threshold: float,
) -> dict[int, float]:
    valid_by_id: dict[int, float] = {}
    for pid, r2 in r2_by_id.items():
        if pid is None:
            continue
        if r2_threshold <= r2 <= 1:
            element = element_by_id.get(pid, -1)
            if element > 0:
                valid_by_id[pid] = element
    return valid_by_id
