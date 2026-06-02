"""Analysis utilities and processing helpers for OpenNTA."""

import logging

from .analysis_processor import (
    AnalysisConfig,
    AnalysisProcessor,
    AnalysisResults,
)
from .data_preprocessor import DataPreprocessor
from .drift_analyzer import DriftAnalyzer
from .drift_corrector import (
    ConfigurationResult,
    DriftCorrector,
    MeanCorrector,
    NoDriftCorrector,
    drift_correction_ui_label_to_mode,
    get_drift_correction_ui_labels,
    get_drift_corrector,
    register_corrector,
    registered_drift_correction_modes,
)
from .size_distributor import (
    DirectDistributor,
    SizeDistributor,
    get_size_distribution_ui_labels,
    get_size_distributor,
    register_distributor,
    registered_size_distribution_modes,
    size_distribution_ui_label_to_mode,
)

_logger = logging.getLogger(__name__)

# Each optional plugin subpackage is imported here so the @register_corrector
# and @register_distributor side effects populate the registries. A missing
# optional dependency in one plugin must not abort the rest of the package,
# so each import is wrapped in its own try/except.

try:
    from .numerical_field import NumericalCorrector  # noqa: E402
except Exception as _exc:  # pragma: no cover
    _logger.warning("NumericalCorrector unavailable: %s", _exc)
    NumericalCorrector = None  # type: ignore[assignment]

try:
    from .unet_field import UNetCorrector  # noqa: E402
except Exception as _exc:  # pragma: no cover
    _logger.warning("UNetCorrector unavailable: %s", _exc)
    UNetCorrector = None  # type: ignore[assignment]

try:
    from .mle import FTLADistributor, IterativeDistributor  # noqa: E402
except Exception as _exc:  # pragma: no cover
    _logger.warning(
        "MLE distributors (FTLA/Iterative) unavailable: %s", _exc
    )
    FTLADistributor = None  # type: ignore[assignment]
    IterativeDistributor = None  # type: ignore[assignment]

# Core modules are imported after the optional plugin blocks so a missing
# plugin dep does not abort the rest of the analysis package.
from ..common.progress import ProgressEmitter  # noqa: E402
from .corrected_track_exporter import write_corrected_track_csv  # noqa: E402
from .diffusion_estimator import DiffusionEstimator  # noqa: E402
from .msd_calculator import MSDCalculator  # noqa: E402
from .result_filters import filter_by_r2  # noqa: E402

__all__ = [
    "AnalysisConfig",
    "AnalysisProcessor",
    "AnalysisResults",
    "ConfigurationResult",
    "DataPreprocessor",
    "DiffusionEstimator",
    "DirectDistributor",
    "DriftAnalyzer",
    "DriftCorrector",
    "FTLADistributor",
    "filter_by_r2",
    "IterativeDistributor",
    "MeanCorrector",
    "MSDCalculator",
    "NoDriftCorrector",
    "NumericalCorrector",
    "ProgressEmitter",
    "SizeDistributor",
    "UNetCorrector",
    "drift_correction_ui_label_to_mode",
    "get_drift_correction_ui_labels",
    "get_drift_corrector",
    "get_size_distribution_ui_labels",
    "get_size_distributor",
    "register_corrector",
    "register_distributor",
    "registered_drift_correction_modes",
    "registered_size_distribution_modes",
    "size_distribution_ui_label_to_mode",
    "write_corrected_track_csv",
]
