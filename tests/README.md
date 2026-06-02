# Test Suite Direction

This document explains *why* the suite is shaped the way it is, so a
future change knows when it should add a unit test, when it should add
an integration test, and when it should add nothing at all.

## Layout

```
tests/
├── conftest.py            # session fixtures + Qt/PyQt5 stubs (headless OK)
├── _helpers/
│   ├── synth.py           # analytic input generators (spots / TIFFs / tracks)
│   └── factories.py       # registry factory maps + SKIPPED_MODES
├── unit/                  # one test file per source class, mirrors src tree
│   ├── analysis/
│   │   ├── mle/           # general-purpose math — kept granular
│   │   └── ...
│   ├── common/
│   └── tracking/
├── integration/           # registry-driven, end-to-end on synthetic input
└── contracts/             # registry-enumeration / skip-list invariants
```

## Inputs are synthesised, never recorded

The suite reads no sample TIFF, no recorded `spots.csv`, no checked-in
golden CSV. Every input — Brownian trajectories, drifted spots, PSF
image stacks — is produced in-process by `_helpers/synth.py` from
analytic parameters, so each test verifies the unit under test against
ground truth that the test itself constructed.

Why: an asset that lives on disk drifts silently from the model it was
captured under (different `pixel_size`, different `dt`, different
calibration). Synthesised inputs collapse the test contract to *one*
source of truth — the parameters in `synth.py`. Anything else is a
dead artefact.

If a test needs a new input shape (different drift profile, different
particle density, longer track), extend `synth.py` rather than commit a
fixture file.

## Per-module unit vs registry-level integration

Two kinds of code live under `opennta/`:

1. **General-purpose math** — `analysis/msd_calculator.py`,
   `analysis/diffusion_estimator.py`, `analysis/mle/*`, `analysis/size_distributor.py`,
   `common/progress.py`, etc. These are reused across multiple
   pipelines, have analytic ground truth, and don't depend on optional
   runtimes.

   → **Test as units, one file per source class.** Cover edge cases,
   error paths, parametrised regimes. The investment compounds because
   the math is consumed by many callers.

2. **Plugin modules** — `analysis/numerical_field/*`, `analysis/unet_field/*`,
   `tracking/trackmate/*`. Each is one strategy plugged into a
   registry (`@register_corrector`, `@register_method`); each has a
   meaningful surface only when invoked through `get_drift_corrector()`
   / `get_tracking_method()`; each pulls in heavy or external
   dependencies (TensorFlow + bundled weights, Fiji binary, headless
   `NumericalFieldParams` injection).

   → **Test as a single result-oriented integration.** Load the module
   from the real registry, feed a synthetic input the user would
   actually feed it, assert on the output. Per-helper unit tests on the
   internal scaffolding (preprocessing kernels, coordinate transforms,
   subprocess command assembly, …) duplicate what the integration test
   already verifies and tend to ossify private structure.

This is why the suite has, e.g., one `test_numerical_corrector.py`
asserting "drift is removed end-to-end" rather than separate tests for
`FieldSampler`, `FieldSmoother`, and `VelocityField`. If the integration
test catches a regression but you can't tell which sub-step broke,
*then* a targeted unit test earns its place.

## Optional dependencies skip cleanly

The U-Net plugin needs TensorFlow plus a bundled weights file; the
TrackMate plugin needs an external Fiji binary. CI environments
routinely have neither.

Integration tests for those plugins skip — *not* fail — when the
prerequisites are absent:

* `test_unet_corrector` calls `pytest.importorskip("tensorflow")` and
  checks `WEIGHTS_PATH.exists()`.
* `test_trackmate_tracking` checks `OPENNTA_FIJI_PATH` (or `FIJI_PATH`)
  for an executable Fiji binary.

The "is registered" half of each test (`test_*_mode_is_registered`)
*does* always run, so a regression that drops a plugin from the
registry still fails CI even when the heavy half is skipped.

To run the heavy halves locally:

```sh
# U-Net
pip install "opennta[unet]"
pytest tests/integration/test_unet_corrector.py

# TrackMate
export OPENNTA_FIJI_PATH=/path/to/Fiji.app/ImageJ-linux64
pytest tests/integration/test_trackmate_tracking.py
```

## Registry-driven contracts

`unit/analysis/test_drift_corrector.py`,
`unit/analysis/test_size_distributor.py`, and
`unit/tracking/test_tracking_registries.py` parametrise over every key
returned by the matching `registered_*_modes()` enumerator. The intent:
when a brand-new plugin is registered, it picks up the basic contract
test on the next `pytest` invocation with no edits to the test file.

Plugins that genuinely cannot run headlessly (Fiji binary, GPU
weights, dialog-only seed path) are listed in
`_helpers/factories.SKIPPED_MODES` with a human-readable reason. The
companion `contracts/test_registry_meta.py` then enforces that every
key in `SKIPPED_MODES` corresponds to a currently-registered mode, so
a stale skip key fails CI rather than silently masking a removed
plugin.

## What *not* to add

* **Tests for UI code under `src/opennta/application/`.**
  These are excluded from coverage in `pyproject.toml` on purpose; UI
  glue is exercised manually. Keep the `unit/application/` tests scoped
  to the Qt-free helpers (`ConfigManager`, `StatisticsUtils`, batch
  output writer, plot data shaping).

* **Tests that mock the module under test.** A test whose body is "the
  thing called the thing it depends on" rephrases the implementation in
  test form and fails on harmless refactors. Mock only at process
  boundaries (`subprocess.Popen`, network, filesystem when relevant) —
  and even then, prefer feeding the real module a real synthetic input
  if you can.

* **Tests for trivial plumbing (passthrough properties, `__repr__`,
  obvious enum lookups).** They inflate the LOC count without changing
  the bug surface.

## Markers

Defined in `pyproject.toml`:

* `@pytest.mark.integration` — slower, end-to-end pipeline tests.
* `@pytest.mark.slow` — anything that takes noticeably longer than 1s
  (currently the TF-bound and Fiji-bound integration tests).

Skip the slow halves with `pytest -m "not slow"`; skip integration
entirely with `pytest -m "not integration"`. The default `pytest`
invocation runs everything.
