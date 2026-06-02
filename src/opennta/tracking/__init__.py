"""Tracking and fitting utilities used by OpenNTA."""

import logging as _logging

from ..common.progress import ProgressEmitter
from .detection_method import (
    DetectionMethod,
    DetectionResult,
    detector_ui_label_to_mode,
    get_detection_method,
    get_detector_ui_labels,
    register_detector,
    registered_detector_modes,
)
from .linking_method import (
    LinkingMethod,
    get_linker_ui_labels,
    get_linking_method,
    linker_ui_label_to_mode,
    register_linker,
    registered_linker_modes,
)
from .tracking_method import (
    SplitMethod,
    TrackingConfigurationResult,
    TrackingMethod,
    get_combined_method_ui_labels,
    get_method_ui_labels,
    get_tracking_method,
    method_ui_label_to_mode,
    register_method,
    registered_method_modes,
)
from .types import FolderItem

# Each optional plugin subpackage is imported here so that its
# @register_method / @register_detector / @register_linker decorators
# execute and populate the registries. A missing optional dependency in
# one plugin must not abort the rest of the package, hence per-plugin
# try/except.
_logger = _logging.getLogger(__name__)

try:
    from .trackmate import FijiRunner, TrackMateMethod
except Exception as _exc:  # pragma: no cover
    _logger.warning("TrackMateMethod unavailable: %s", _exc)
    TrackMateMethod = None  # type: ignore[assignment]
    FijiRunner = None  # type: ignore[assignment]

from .tracking_processor import (  # noqa: E402 - imported after optional-plugin guards above
    ProcessingParams,
    TrackingProcessor,
)

__all__ = [
    "DetectionMethod",
    "DetectionResult",
    "FijiRunner",
    "FolderItem",
    "LinkingMethod",
    "ProcessingParams",
    "ProgressEmitter",
    "SplitMethod",
    "TrackMateMethod",
    "TrackingConfigurationResult",
    "TrackingMethod",
    "TrackingProcessor",
    "detector_ui_label_to_mode",
    "get_combined_method_ui_labels",
    "get_detection_method",
    "get_detector_ui_labels",
    "get_linker_ui_labels",
    "get_linking_method",
    "get_method_ui_labels",
    "get_tracking_method",
    "linker_ui_label_to_mode",
    "method_ui_label_to_mode",
    "register_detector",
    "register_linker",
    "register_method",
    "registered_detector_modes",
    "registered_linker_modes",
    "registered_method_modes",
]
