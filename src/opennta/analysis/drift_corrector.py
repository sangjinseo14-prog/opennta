from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from .types import AnalysisConfig

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationResult:
    apply_to: Callable[[DriftCorrector], None]


@dataclass
class ExportField:
    # Velocity field returned by a corrector for CSV export; u, v are in m/s.
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    u: NDArray[np.floating]
    v: NDArray[np.floating]


class DriftCorrector(ABC):

    mode_key: str = ""
    ui_label: str = ""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        mean_vx: float,
        mean_vy: float,
    ) -> None: ...

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def get_display_name(self) -> str:
        return self.ui_label or self.__class__.__name__

    def requires_configuration(self) -> bool:
        return False

    def get_parameters(self) -> dict[str, Any]:
        return {}

    def get_export_field(self) -> ExportField | None:
        # Correctors that estimate a velocity field override this to expose it
        # for CSV export. Returns None when there is nothing to export.
        return None

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.get_display_name()} was not fitted before apply(). "
                "Call fit() first."
            )


_REGISTRY: dict[str, type[DriftCorrector]] = {}


def register_corrector(cls: type[DriftCorrector]) -> type[DriftCorrector]:
    if not cls.mode_key:
        raise ValueError(
            f"{cls.__name__} must define class attr 'mode_key' before @register_corrector"
        )
    if not cls.ui_label:
        raise ValueError(
            f"{cls.__name__} must define class attr 'ui_label' before @register_corrector"
        )
    existing = _REGISTRY.get(cls.mode_key)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"mode_key '{cls.mode_key}' already registered by "
            f"{existing.__name__} (trying to register {cls.__name__})"
        )
    _REGISTRY[cls.mode_key] = cls
    return cls


def get_drift_corrector(config: AnalysisConfig, mode: str = "no") -> DriftCorrector:
    mode = (mode or "no").lower()
    if mode not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown drift correction mode: {mode!r}. Available: {available}"
        )
    return _REGISTRY[mode](config)


def get_drift_correction_ui_labels() -> list[str]:
    return [cls.ui_label for cls in _REGISTRY.values()]


def drift_correction_ui_label_to_mode(label: str) -> str:
    for cls in _REGISTRY.values():
        if cls.ui_label == label:
            return cls.mode_key
    return "no"


def registered_drift_correction_modes() -> list[str]:
    return list(_REGISTRY.keys())


@register_corrector
class MeanCorrector(DriftCorrector):
    mode_key = "mean"
    ui_label = "Global: Mean"

    def fit(
        self,
        df: pd.DataFrame,
        mean_vx: float,
        mean_vy: float,
    ) -> None:
        self._mean_vx = float(mean_vx)
        self._mean_vy = float(mean_vy)
        self._fitted = True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        correction = (
            df.groupby("ID")["FRAME"].transform(lambda f: f - f.min())
            * self.config.dt
        )
        df = df.copy()
        df["X_diff_corr"] = df["X_diff"] - self._mean_vx * correction
        df["Y_diff_corr"] = df["Y_diff"] - self._mean_vy * correction
        return df

    def get_parameters(self) -> dict[str, Any]:
        return {
            "Mean Vx": (float(getattr(self, "_mean_vx", 0.0)), "um/s"),
            "Mean Vy": (float(getattr(self, "_mean_vy", 0.0)), "um/s"),
        }


@register_corrector
class NoDriftCorrector(DriftCorrector):
    mode_key = "no"
    ui_label = "No Correction"

    def fit(
        self,
        df: pd.DataFrame,
        mean_vx: float,
        mean_vy: float,
    ) -> None:
        self._fitted = True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        df = df.copy()
        df["X_diff_corr"] = df["X_diff"]
        df["Y_diff_corr"] = df["Y_diff"]
        return df


__all__ = [
    "ConfigurationResult",
    "ExportField",
    "DriftCorrector",
    "MeanCorrector",
    "NoDriftCorrector",
    "register_corrector",
    "get_drift_corrector",
    "get_drift_correction_ui_labels",
    "drift_correction_ui_label_to_mode",
    "registered_drift_correction_modes",
]
