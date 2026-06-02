from __future__ import annotations

from abc import ABC, abstractmethod

from ..common.progress import ProgressEmitter
from .detection_method import DetectionResult
from .types import ProcessingParams


class LinkingMethod(ABC):

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
    def link_spots_into_tracks(
        self,
        detection: DetectionResult,
        sub_path: str,
        nta_folder: str,
        sub: str,
    ) -> bool: ...

    def get_parameter_log_lines(self) -> list[str]:
        return []

    def get_display_name(self) -> str:
        return self.ui_label or self.__class__.__name__


_REGISTRY: dict[str, type[LinkingMethod]] = {}


def register_linker(cls: type[LinkingMethod]) -> type[LinkingMethod]:
    if not cls.mode_key:
        raise ValueError(
            f"{cls.__name__} must define class attr 'mode_key' before @register_linker"
        )
    if not cls.ui_label:
        raise ValueError(
            f"{cls.__name__} must define class attr 'ui_label' before @register_linker"
        )
    existing = _REGISTRY.get(cls.mode_key)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"linker mode_key '{cls.mode_key}' already registered by "
            f"{existing.__name__} (trying to register {cls.__name__})"
        )
    _REGISTRY[cls.mode_key] = cls
    return cls


def get_linking_method(
    mode: str,
    params: ProcessingParams,
    emitter: ProgressEmitter,
    save_path: str | None = None,
) -> LinkingMethod:
    mode = (mode or "").lower()
    if mode not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise ValueError(
            f"Unknown linking method: {mode!r}. Available: {available}"
        )
    return _REGISTRY[mode](params=params, emitter=emitter, save_path=save_path)


def get_linker_ui_labels() -> list[str]:
    return [cls.ui_label for cls in _REGISTRY.values()]


def linker_ui_label_to_mode(label: str) -> str:
    for cls in _REGISTRY.values():
        if cls.ui_label == label:
            return cls.mode_key
    return ""


def registered_linker_modes() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = [
    "LinkingMethod",
    "get_linker_ui_labels",
    "get_linking_method",
    "linker_ui_label_to_mode",
    "register_linker",
    "registered_linker_modes",
]
