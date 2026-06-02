"""Per-registry test factory maps and skip metadata.

Registry-driven contract tests parametrize over every key returned by the
``registered_*_modes()`` enumerators. Some modules need pre-configuration
that the bare constructor doesn't perform (e.g. the numerical/U-Net drift
correctors expect a velocity field to be supplied via a configurator dialog
before ``fit()`` is called; TrackMate needs a Fiji binary; SplitMethod needs
already-registered detector and linker plugins). For those modes:

* If a "fit-ready" factory can be expressed in code, add it to the matching
  factory dict here.
* If the mode genuinely cannot run under CI (external binary, GPU weights,
  registered-plugin prerequisite, etc.), list it in ``SKIPPED_MODES`` with
  a human-readable reason.

When a brand-new module is registered, the contract test will start running
against it on the next ``pytest`` invocation with no edits. If its default
constructor isn't fit-ready, the test fails — making "add an entry here" the
intended forcing function.

The companion ``tests/contracts/test_registry_meta.py`` enforces
that every key in ``SKIPPED_MODES`` corresponds to a currently-registered
mode in at least one registry, so a stale skip key fails CI rather than
silently masking a removed plugin.
"""

from __future__ import annotations

from collections.abc import Callable

from opennta.analysis.drift_corrector import DriftCorrector, get_drift_corrector
from opennta.analysis.numerical_field.types import NumericalFieldParams
from opennta.analysis.size_distributor import SizeDistributor
from opennta.analysis.types import AnalysisConfig
from opennta.common.progress import ProgressEmitter
from opennta.tracking.detection_method import DetectionMethod
from opennta.tracking.linking_method import LinkingMethod
from opennta.tracking.tracking_method import TrackingMethod, get_tracking_method
from opennta.tracking.types import ProcessingParams

DriftFactory = Callable[[AnalysisConfig], DriftCorrector]
SizeFactory = Callable[[], SizeDistributor]
DetectorFactory = Callable[[ProcessingParams, ProgressEmitter], DetectionMethod]
LinkerFactory = Callable[[ProcessingParams, ProgressEmitter], LinkingMethod]
TrackingFactory = Callable[[ProcessingParams, ProgressEmitter], TrackingMethod]


def _numerical_corrector_factory(config: AnalysisConfig) -> DriftCorrector:
    """Build a ``numerical`` corrector seeded with a headless velocity-field
    config, standing in for the dialog the GUI would normally supply. The
    returned corrector is *configured but not fitted*, so the contract test's
    ``apply()``-before-``fit()`` guard still trips before the real fit runs.
    """
    corr = get_drift_corrector(config, "numerical")
    corr.set_params(
        NumericalFieldParams(
            n_windows=4,
            x_range=(0.0, 500.0),
            y_range=(0.0, 500.0),
            outlier_k=3.0,
            min_count=1,
            ksize=3,
            n_iter=1,
            sigma=1.0,
        )
    )
    return corr


def _split_method_factory(
    params: ProcessingParams, emitter: ProgressEmitter
) -> TrackingMethod:
    """Build a ``split`` tracking method wired to the registered fake detector
    and linker, exercising ``SplitMethod`` composition headlessly (no Fiji)."""
    p = ProcessingParams(detection_method="fake", linking_method="fake")
    return get_tracking_method("split", p, emitter)


DRIFT_FACTORIES: dict[str, DriftFactory] = {
    "numerical": _numerical_corrector_factory,
}
SIZE_FACTORIES: dict[str, SizeFactory] = {}
DETECTOR_FACTORIES: dict[str, DetectorFactory] = {}
LINKER_FACTORIES: dict[str, LinkerFactory] = {}
TRACKING_FACTORIES: dict[str, TrackingFactory] = {
    "split": _split_method_factory,
}


SKIPPED_MODES: dict[str, str] = {
    # opennta.analysis.drift_corrector registry
    "unet": (
        "UNetCorrector requires a trained checkpoint and the optional "
        "U-Net runtime dependencies."
    ),
    # opennta.tracking.tracking_method registry
    "trackmate": (
        "TrackMateMethod requires an external Fiji/ImageJ binary."
    ),
}


__all__ = [
    "DRIFT_FACTORIES",
    "SIZE_FACTORIES",
    "DETECTOR_FACTORIES",
    "LINKER_FACTORIES",
    "TRACKING_FACTORIES",
    "SKIPPED_MODES",
    "DriftFactory",
    "SizeFactory",
    "DetectorFactory",
    "LinkerFactory",
    "TrackingFactory",
]
