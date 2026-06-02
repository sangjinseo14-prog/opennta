"""Meta-tests guarding the registry-driven contract tests.

The contract tests in ``test_drift_corrector.py``, ``test_size_distributor.py``
and ``test_tracking_registries.py`` parametrize over each ``registered_*_modes()``
enumerator and skip entries listed in ``factories.SKIPPED_MODES``. Two
classes of regression silently neuter that machinery:

1. A skip key in ``SKIPPED_MODES`` that no longer matches any registered
   mode — stale skips that mask removed plugins or typos.
2. A registry that yields no modes for a category we expect to have at
   least one (drift / size / tracking-method are always populated by the
   built-in plugins).

These tests fail loudly when either invariant breaks.
"""
from __future__ import annotations

from opennta.analysis.drift_corrector import registered_drift_correction_modes
from opennta.analysis.size_distributor import registered_size_distribution_modes
from opennta.tests._helpers.factories import SKIPPED_MODES
from opennta.tracking.detection_method import registered_detector_modes
from opennta.tracking.linking_method import registered_linker_modes
from opennta.tracking.tracking_method import registered_method_modes


def _all_registered_modes() -> set[str]:
    return (
        set(registered_drift_correction_modes())
        | set(registered_size_distribution_modes())
        | set(registered_detector_modes())
        | set(registered_linker_modes())
        | set(registered_method_modes())
    )


def test_skipped_modes_are_all_currently_registered():
    """Every key in ``SKIPPED_MODES`` must correspond to a real registered
    mode in at least one registry. A stale key suggests either a plugin
    was removed (clean up the skip) or a typo (the skip is silently inert).
    """
    registered = _all_registered_modes()
    stale = sorted(set(SKIPPED_MODES) - registered)
    assert not stale, (
        f"SKIPPED_MODES contains keys no longer registered by any plugin: "
        f"{stale}. Remove them from tests/_helpers/factories.py or fix the typo."
    )


def test_drift_registry_is_populated():
    """Built-in correctors (no, mean) must always be registered, otherwise
    the registry contract tests collapse to zero parametrizations."""
    modes = set(registered_drift_correction_modes())
    assert {"no", "mean"}.issubset(modes), (
        f"Expected built-in drift correctors 'no' and 'mean' in registry; got {modes}"
    )


def test_size_registry_is_populated():
    modes = set(registered_size_distribution_modes())
    assert "direct" in modes, (
        f"Expected built-in 'direct' size distributor in registry; got {modes}"
    )


def test_tracking_method_registry_is_populated():
    modes = set(registered_method_modes())
    # ``trackmate`` ships in-package; ``split`` ships in-package too.
    assert modes, "Tracking-method registry is empty — plugin imports broke."
