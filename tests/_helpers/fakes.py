"""Test-only detector / linker plugins.

opennta ships these two registries *empty*: the only built-in tracking method,
TrackMate, is monolithic (detection + linking happen inside one Fiji run), and
``SplitMethod`` is designed to compose detector/linker plugins that a
downstream deployment registers. With nothing registered, the registry-driven
contract tests in ``test_tracking_registries.py`` collapse to zero
parametrizations — they go *green without asserting anything*, which is exactly
the silent-no-op failure mode the contract machinery exists to prevent.

Registering one minimal fake of each restores a real assertion surface: the
``@register_*`` decorator, the ``get_*_method`` lookup, the ``mode_key`` /
``ui_label`` plumbing, and ``SplitMethod`` composition all get exercised
headlessly. The fakes do no real detection/linking — they return a trivial
synthetic result — because the contract under test is *registration and
construction*, not tracking quality (that is covered end-to-end by the
Fiji-bound integration test).

Importing this module has the side effect of registering both fakes; the test
``conftest`` imports it once so the registries are populated before any test
enumerates them.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from opennta.tracking.detection_method import (
    DetectionMethod,
    DetectionResult,
    register_detector,
)
from opennta.tracking.linking_method import LinkingMethod, register_linker


@register_detector
class FakeDetector(DetectionMethod):
    """Returns a single synthetic spot regardless of input."""

    mode_key = "fake"
    ui_label = "Fake Detector (test)"

    def detect_spots(
        self,
        sub_path: str,
        tiff_paths: Sequence[str],
        nta_folder: str,
        sub: str,
    ) -> DetectionResult:
        spots = pd.DataFrame(
            {"ID": [1], "X": [0.0], "Y": [0.0], "FRAME": [0], "QUALITY": [1.0]}
        )
        return DetectionResult(spots_df=spots, stack_path=sub_path)


@register_linker
class FakeLinker(LinkingMethod):
    """Accepts any detection result and reports success without writing files."""

    mode_key = "fake"
    ui_label = "Fake Linker (test)"

    def link_spots_into_tracks(
        self,
        detection: DetectionResult,
        sub_path: str,
        nta_folder: str,
        sub: str,
    ) -> bool:
        return True


__all__ = ["FakeDetector", "FakeLinker"]
