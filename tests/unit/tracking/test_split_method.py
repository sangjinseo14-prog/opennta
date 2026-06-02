"""Unit tests for ``SplitMethod`` orchestration and detector/linker registries.

``SplitMethod`` composes a registered detector and linker into one tracking
run. Its construction validation and the detect -> link -> status orchestration
are Fiji-independent (the built-in TrackMate path is separate), so they are
driven here with the in-package Fake plugins plus a couple of ad-hoc failing
plugins registered on the fly.
"""
from __future__ import annotations

import pytest

from opennta.common.progress import ProgressEmitter
from opennta.tracking.detection_method import (
    DetectionMethod,
    get_detection_method,
    register_detector,
    registered_detector_modes,
)
from opennta.tracking.linking_method import (
    LinkingMethod,
    register_linker,
    registered_linker_modes,
)
from opennta.tracking.status_codes import TrackingStatus
from opennta.tracking.tracking_method import get_tracking_method
from opennta.tracking.types import ProcessingParams


@pytest.fixture
def emitter(tmp_path) -> ProgressEmitter:
    return ProgressEmitter(save_path=str(tmp_path))


def _split(emitter, *, detector="fake", linker="fake"):
    params = ProcessingParams(detection_method=detector, linking_method=linker)
    return get_tracking_method("split", params, emitter)


def test_split_requires_detection_method(emitter):
    params = ProcessingParams(detection_method="", linking_method="fake")
    with pytest.raises(ValueError, match="requires params.detection_method"):
        get_tracking_method("split", params, emitter)


def test_split_requires_linking_method(emitter):
    params = ProcessingParams(detection_method="fake", linking_method="")
    with pytest.raises(ValueError, match="requires params.linking_method"):
        get_tracking_method("split", params, emitter)


def test_split_unknown_detector_raises(emitter):
    params = ProcessingParams(detection_method="nope", linking_method="fake")
    with pytest.raises(ValueError, match="Unknown detection method"):
        get_tracking_method("split", params, emitter)


def test_fake_plugins_are_registered():
    assert "fake" in registered_detector_modes()
    assert "fake" in registered_linker_modes()


def test_get_detection_method_unknown_lists_available(emitter, processing_params):
    with pytest.raises(ValueError, match="Available: .*fake"):
        get_detection_method("does_not_exist", processing_params, emitter)


def test_duplicate_detector_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):
        @register_detector
        class DupDetector(DetectionMethod):
            mode_key = "fake"  # collides with the in-package FakeDetector
            ui_label = "Dup"

            def detect_spots(self, *a, **k):  # pragma: no cover - never called
                return None


def test_detector_missing_mode_key_rejected():
    with pytest.raises(ValueError, match="mode_key"):
        @register_detector
        class NoKeyDetector(DetectionMethod):
            ui_label = "No key"

            def detect_spots(self, *a, **k):  # pragma: no cover - never called
                return None


def test_split_happy_path_returns_true_and_emits_complete(emitter):
    codes = []
    emitter.progress_callback = codes.append
    method = _split(emitter)

    ok = method.track_subfolder("sub", ["a.tif"], "nta", "s")

    assert ok is True
    assert TrackingStatus.ITEM_COMPLETE in codes


def test_split_returns_false_when_detection_returns_none(emitter):
    @register_detector
    class NoneDetector(DetectionMethod):
        mode_key = "none_detect"
        ui_label = "None Detector (test)"

        def detect_spots(self, *a, **k):
            return None

    codes = []
    emitter.progress_callback = codes.append
    method = _split(emitter, detector="none_detect")

    ok = method.track_subfolder("sub", ["a.tif"], "nta", "s")

    assert ok is False
    assert TrackingStatus.ERROR_THRESHOLD_DETECTION in codes


def test_split_returns_false_when_linking_fails(emitter):
    @register_linker
    class FailLinker(LinkingMethod):
        mode_key = "fail_link"
        ui_label = "Failing Linker (test)"

        def link_spots_into_tracks(self, *a, **k):
            return False

    codes = []
    emitter.progress_callback = codes.append
    method = _split(emitter, linker="fail_link")

    ok = method.track_subfolder("sub", ["a.tif"], "nta", "s")

    assert ok is False
    assert TrackingStatus.ERROR_TRACKING in codes


def test_split_catches_detector_exception(emitter):
    @register_detector
    class BoomDetector(DetectionMethod):
        mode_key = "boom_detect"
        ui_label = "Boom Detector (test)"

        def detect_spots(self, *a, **k):
            raise RuntimeError("kaboom")

    codes = []
    emitter.progress_callback = codes.append
    method = _split(emitter, detector="boom_detect")

    # The pipeline swallows the exception and reports a general error rather
    # than propagating, so a single bad subfolder doesn't abort the batch.
    ok = method.track_subfolder("sub", ["a.tif"], "nta", "s")

    assert ok is False
    assert TrackingStatus.ERROR_GENERAL in codes
