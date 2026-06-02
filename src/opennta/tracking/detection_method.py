from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..common.progress import ProgressEmitter
from .types import ProcessingParams


@dataclass
class DetectionResult:
    spots_csv_path: str | None = None
    spots_df: pd.DataFrame | None = None
    stack_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class DetectionMethod(ABC):

    mode_key: str = ""
    ui_label: str = ""

    def __init__(
        self,
        params: ProcessingParams,
        emitter: ProgressEmitter,
        save_path: str | None = None,
    ):
        self.params = params
        self.emitter = emitter
        self.save_path = save_path

    @abstractmethod
    def detect_spots(
        self,
        sub_path: str,
        tiff_paths: Sequence[str],
        nta_folder: str,
        sub: str,
    ) -> DetectionResult | None: ...

    def get_parameter_log_lines(self) -> list[str]:
        return []

    def get_display_name(self) -> str:
        return self.ui_label or self.__class__.__name__


_REGISTRY: dict[str, type[DetectionMethod]] = {}


def register_detector(cls: type[DetectionMethod]) -> type[DetectionMethod]:
    if not cls.mode_key:
        raise ValueError(
            f"{cls.__name__} must define class attr 'mode_key' before @register_detector"
        )
    if not cls.ui_label:
        raise ValueError(
            f"{cls.__name__} must define class attr 'ui_label' before @register_detector"
        )
    existing = _REGISTRY.get(cls.mode_key)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"detector mode_key '{cls.mode_key}' already registered by "
            f"{existing.__name__} (trying to register {cls.__name__})"
        )
    _REGISTRY[cls.mode_key] = cls
    return cls


def get_detection_method(
    mode: str,
    params: ProcessingParams,
    emitter: ProgressEmitter,
    save_path: str | None = None,
) -> DetectionMethod:
    mode = (mode or "").lower()
    if mode not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise ValueError(
            f"Unknown detection method: {mode!r}. Available: {available}"
        )
    return _REGISTRY[mode](params=params, emitter=emitter, save_path=save_path)


def get_detector_ui_labels() -> list[str]:
    return [cls.ui_label for cls in _REGISTRY.values()]


def detector_ui_label_to_mode(label: str) -> str:
    for cls in _REGISTRY.values():
        if cls.ui_label == label:
            return cls.mode_key
    return ""


def registered_detector_modes() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = [
    "DetectionMethod",
    "DetectionResult",
    "detector_ui_label_to_mode",
    "get_detection_method",
    "get_detector_ui_labels",
    "register_detector",
    "registered_detector_modes",
]
