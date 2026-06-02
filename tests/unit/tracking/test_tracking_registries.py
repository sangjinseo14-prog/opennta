"""Registry-driven contract tests for tracking-side plugins.

Each ``registered_*_modes()`` enumerator drives a ``pytest.mark.parametrize``,
so a freshly ``@register_detector`` / ``@register_linker`` / ``@register_method``
class is automatically exercised at the next pytest run with zero edits here.

When no plugins are registered yet (today's state for detector and linker),
the parametrized cases collapse to zero and only the unknown-mode negative
test runs — the moment a plugin appears, its contract test is collected.
"""

from __future__ import annotations

import pytest

from opennta.tests._helpers.factories import (
    DETECTOR_FACTORIES,
    LINKER_FACTORIES,
    SKIPPED_MODES,
    TRACKING_FACTORIES,
)
from opennta.tracking.detection_method import (
    DetectionMethod,
    get_detection_method,
    registered_detector_modes,
)
from opennta.tracking.linking_method import (
    LinkingMethod,
    get_linking_method,
    registered_linker_modes,
)
from opennta.tracking.tracking_method import (
    TrackingMethod,
    get_tracking_method,
    registered_method_modes,
)

ALL_DETECTOR_MODES = sorted(registered_detector_modes())
ALL_LINKER_MODES = sorted(registered_linker_modes())
ALL_TRACKING_MODES = sorted(registered_method_modes())


@pytest.mark.parametrize("mode", ALL_DETECTOR_MODES)
def test_detector_registry_contract(processing_params, progress_emitter, mode):
    if mode in SKIPPED_MODES:
        pytest.skip(SKIPPED_MODES[mode])
    factory = DETECTOR_FACTORIES.get(
        mode,
        lambda p, e, _mode=mode: get_detection_method(_mode, p, e),
    )
    obj = factory(processing_params, progress_emitter)
    assert isinstance(obj, DetectionMethod)
    assert obj.mode_key == mode
    assert obj.get_display_name()


def test_detection_unknown_mode_raises(processing_params, progress_emitter):
    with pytest.raises(ValueError, match="Unknown detection method"):
        get_detection_method("does_not_exist", processing_params, progress_emitter)


@pytest.mark.parametrize("mode", ALL_LINKER_MODES)
def test_linker_registry_contract(processing_params, progress_emitter, mode):
    if mode in SKIPPED_MODES:
        pytest.skip(SKIPPED_MODES[mode])
    factory = LINKER_FACTORIES.get(
        mode,
        lambda p, e, _mode=mode: get_linking_method(_mode, p, e),
    )
    obj = factory(processing_params, progress_emitter)
    assert isinstance(obj, LinkingMethod)
    assert obj.mode_key == mode
    assert obj.get_display_name()


def test_linker_unknown_mode_raises(processing_params, progress_emitter):
    with pytest.raises(ValueError, match="Unknown linking method"):
        get_linking_method("does_not_exist", processing_params, progress_emitter)


@pytest.mark.parametrize("mode", ALL_TRACKING_MODES)
def test_tracking_method_registry_contract(
    processing_params, progress_emitter, mode
):
    if mode in SKIPPED_MODES:
        pytest.skip(SKIPPED_MODES[mode])
    factory = TRACKING_FACTORIES.get(
        mode,
        lambda p, e, _mode=mode: get_tracking_method(_mode, p, e),
    )
    obj = factory(processing_params, progress_emitter)
    assert isinstance(obj, TrackingMethod)
    assert obj.mode_key == mode
    assert obj.get_display_name()


def test_tracking_method_unknown_mode_raises(processing_params, progress_emitter):
    with pytest.raises(ValueError, match="Unknown tracking method"):
        get_tracking_method("does_not_exist", processing_params, progress_emitter)
