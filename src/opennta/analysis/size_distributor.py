from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from .types import AnalysisConfig

logger = logging.getLogger(__name__)


class SizeDistributor(ABC):

    mode_key: str = ""    # e.g. "direct"
    ui_label: str = ""    # e.g. "Direct"

    @abstractmethod
    def compute(
        self,
        diameters_by_id: dict[int, float] | None,
        min_d: float = 0.1,
        max_d: float = 1000.0,
        num_bins: int = 100,
        *,
        msd_by_id: dict[int, tuple[NDArray, NDArray]] | None = None,
        df: pd.DataFrame | None = None,
        config: AnalysisConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...

    def get_display_name(self) -> str:
        return self.ui_label or self.__class__.__name__


_REGISTRY: dict[str, type[SizeDistributor]] = {}


def register_distributor(cls: type[SizeDistributor]) -> type[SizeDistributor]:
    if not cls.mode_key:
        raise ValueError(
            f"{cls.__name__} must define class attr 'mode_key' before @register_distributor"
        )
    if not cls.ui_label:
        raise ValueError(
            f"{cls.__name__} must define class attr 'ui_label' before @register_distributor"
        )
    existing = _REGISTRY.get(cls.mode_key)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"mode_key '{cls.mode_key}' already registered by "
            f"{existing.__name__} (trying to register {cls.__name__})"
        )
    _REGISTRY[cls.mode_key] = cls
    return cls


def get_size_distributor(mode: str = "direct") -> SizeDistributor:
    mode = (mode or "direct").lower()
    if mode not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown size distribution mode: {mode!r}. Available: {available}"
        )
    return _REGISTRY[mode]()


def get_size_distribution_ui_labels() -> list[str]:
    return [cls.ui_label for cls in _REGISTRY.values()]


def size_distribution_ui_label_to_mode(label: str) -> str:
    for cls in _REGISTRY.values():
        if cls.ui_label == label:
            return cls.mode_key
    return "direct"


def registered_size_distribution_modes() -> list[str]:
    return list(_REGISTRY.keys())


@register_distributor
class DirectDistributor(SizeDistributor):

    mode_key = "direct"
    ui_label = "Direct"

    def compute(
        self,
        diameters_by_id: dict[int, float] | None,
        min_d: float = 0.1,
        max_d: float = 1000.0,
        num_bins: int = 100,
        *,
        log_scale: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        if diameters_by_id is None:
            return {
                "diameters": None,
                "bin_edges": None,
                "counts": None,
                "bin_centers": None,
            }

        diameters = np.array([d for d in diameters_by_id.values() if d > 0])

        if log_scale:
            lo = max(min_d, 0.0001)
            hi = max(max_d, lo + 1)
            bin_edges = np.logspace(np.log10(lo), np.log10(hi), num_bins + 1)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        else:
            lo = max(min_d, 0.0)
            hi = max(max_d, lo + 1)
            bin_edges = np.linspace(lo, hi, num_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        counts, _ = np.histogram(diameters, bins=bin_edges)

        return {
            "diameters": diameters,
            "bin_edges": bin_edges,
            "counts": counts,
            "bin_centers": bin_centers,
        }


__all__ = [
    "SizeDistributor",
    "DirectDistributor",
    "register_distributor",
    "get_size_distributor",
    "get_size_distribution_ui_labels",
    "size_distribution_ui_label_to_mode",
    "registered_size_distribution_modes",
]
