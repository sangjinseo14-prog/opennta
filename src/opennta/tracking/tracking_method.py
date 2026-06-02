from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..common.progress import ProgressEmitter
from .detection_method import (
    DetectionMethod,
    get_detection_method,
    registered_detector_modes,
)
from .linking_method import (
    LinkingMethod,
    get_linking_method,
    registered_linker_modes,
)
from .status_codes import TrackingStatus
from .types import ProcessingParams

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfigurationResult:
    params: ProcessingParams


class TrackingMethod(ABC):

    mode_key: str = ""
    ui_label: str = ""
    is_combined: bool = True  # False = method uses Detector + Linker registries

    def __init__(
        self,
        params: ProcessingParams,
        emitter: ProgressEmitter,
        save_path: str | None = None,
        fiji_path: str | None = None,
    ):
        self.params = params
        self.emitter = emitter
        self.save_path = save_path
        self.fiji_path = fiji_path

    @abstractmethod
    def track_subfolder(
        self,
        sub_path: str,
        tiff_paths: Sequence[str],
        nta_folder: str,
        sub: str,
    ) -> bool: ...

    def get_parameter_log_lines(self) -> list[str]:
        return []

    def get_display_name(self) -> str:
        return self.ui_label or self.__class__.__name__

    def requires_configuration(self) -> bool:
        return False

    def _move_output_file(self, nta_folder: str, file_path: str | None) -> None:
        if not self.save_path:
            return
        if not file_path or not os.path.exists(file_path):
            return

        dst_dir = Path(self.save_path) / nta_folder
        dst_dir.mkdir(parents=True, exist_ok=True)

        file_dst = dst_dir / Path(file_path).name
        if file_dst.exists():
            stem = file_dst.stem
            suffix = file_dst.suffix
            i = 1
            while True:
                candidate = dst_dir / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    file_dst = candidate
                    break
                i += 1
        os.replace(file_path, file_dst)


_REGISTRY: dict[str, type[TrackingMethod]] = {}


def register_method(cls: type[TrackingMethod]) -> type[TrackingMethod]:
    if not cls.mode_key:
        raise ValueError(
            f"{cls.__name__} must define class attr 'mode_key' before @register_method"
        )
    if not cls.ui_label:
        raise ValueError(
            f"{cls.__name__} must define class attr 'ui_label' before @register_method"
        )
    existing = _REGISTRY.get(cls.mode_key)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"mode_key '{cls.mode_key}' already registered by "
            f"{existing.__name__} (trying to register {cls.__name__})"
        )
    _REGISTRY[cls.mode_key] = cls
    return cls


def get_tracking_method(
    mode: str,
    params: ProcessingParams,
    emitter: ProgressEmitter,
    save_path: str | None = None,
    fiji_path: str | None = None,
) -> TrackingMethod:
    mode = (mode or "trackmate").lower()
    if mode not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown tracking method: {mode!r}. Available: {available}"
        )
    return _REGISTRY[mode](
        params=params,
        emitter=emitter,
        save_path=save_path,
        fiji_path=fiji_path,
    )


def get_method_ui_labels() -> list[str]:
    return [cls.ui_label for cls in _REGISTRY.values()]


def get_combined_method_ui_labels() -> list[str]:
    return [cls.ui_label for cls in _REGISTRY.values() if cls.is_combined]


def method_ui_label_to_mode(label: str) -> str:
    for cls in _REGISTRY.values():
        if cls.ui_label == label:
            return cls.mode_key
    return "trackmate"


def registered_method_modes() -> list[str]:
    return list(_REGISTRY.keys())


@register_method
class SplitMethod(TrackingMethod):

    mode_key = "split"
    ui_label = "Split: Detector + Linker"
    is_combined = False

    def __init__(
        self,
        params: ProcessingParams,
        emitter: ProgressEmitter,
        save_path: str | None = None,
        fiji_path: str | None = None,
    ):
        super().__init__(params, emitter, save_path, fiji_path)

        detector_mode = params.detection_method
        linker_mode = params.linking_method

        if not detector_mode:
            available = ", ".join(registered_detector_modes()) or "<none>"
            raise ValueError(
                "SplitMethod requires params.detection_method. "
                f"Available detectors: {available}"
            )
        if not linker_mode:
            available = ", ".join(registered_linker_modes()) or "<none>"
            raise ValueError(
                "SplitMethod requires params.linking_method. "
                f"Available linkers: {available}"
            )

        self.detector: DetectionMethod = get_detection_method(
            detector_mode, params=params, emitter=emitter, save_path=save_path
        )
        self.linker: LinkingMethod = get_linking_method(
            linker_mode, params=params, emitter=emitter, save_path=save_path
        )

    def get_parameter_log_lines(self) -> list[str]:
        lines = [
            "[ Detector ]",
            f"  Name: {self.detector.get_display_name()}",
            f"  Mode: {self.detector.mode_key}",
            "",
        ]
        lines.extend(self.detector.get_parameter_log_lines())
        lines.extend([
            "[ Linker ]",
            f"  Name: {self.linker.get_display_name()}",
            f"  Mode: {self.linker.mode_key}",
            "",
        ])
        lines.extend(self.linker.get_parameter_log_lines())
        return lines

    def track_subfolder(
        self,
        sub_path: str,
        tiff_paths: Sequence[str],
        nta_folder: str,
        sub: str,
    ) -> bool:
        try:
            self.emitter.emit(
                code=TrackingStatus.STARTED,
                msg=f"[detect] {self.detector.get_display_name()}",
            )
            detection = self.detector.detect_spots(sub_path, tiff_paths, nta_folder, sub)
            if detection is None:
                self.emitter.emit(
                    code=TrackingStatus.ERROR_THRESHOLD_DETECTION,
                    msg=f"Detection failed: {self.detector.get_display_name()}",
                )
                return False

            self.emitter.emit(
                code=TrackingStatus.TRACKING_START,
                msg=f"[link] {self.linker.get_display_name()}",
            )
            if not self.linker.link_spots_into_tracks(detection, sub_path, nta_folder, sub):
                self.emitter.emit(
                    code=TrackingStatus.ERROR_TRACKING,
                    msg=f"Linking failed: {self.linker.get_display_name()}",
                )
                return False

            self.emitter.emit(
                code=TrackingStatus.ITEM_COMPLETE,
                msg="Detect + Link complete",
            )
            return True

        except Exception as e:
            logger.exception("Split tracking pipeline failed")
            self.emitter.emit(code=TrackingStatus.ERROR_GENERAL, msg=str(e))
            return False


__all__ = [
    "SplitMethod",
    "TrackingConfigurationResult",
    "TrackingMethod",
    "register_method",
    "get_tracking_method",
    "get_combined_method_ui_labels",
    "get_method_ui_labels",
    "method_ui_label_to_mode",
    "registered_method_modes",
]
