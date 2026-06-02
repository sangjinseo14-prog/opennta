"""Shared pytest fixtures for opennta tests.

Layout (modelled after ``OpenMS/src/tests``):

    tests/
    ├── conftest.py               # session-scoped fixtures (this file)
    ├── _helpers/                 # synth data generators + registry factories
    │   ├── synth.py
    │   └── factories.py
    ├── unit/                     # mirrors the source tree: one test file per
    │   ├── analysis/             # source class, named after the class.
    │   │   ├── mle/
    │   │   └── numerical_field/
    │   ├── common/
    │   └── tracking/
    │       └── trackmate/
    ├── integration/              # end-to-end pipeline tests
    └── contracts/                # registry / skip-list invariants

Tests intentionally never read pre-recorded sample files from disk.
Anything resembling experimental input (spot CSVs, TIFF stacks,
configuration metadata) is synthesised in-process from analytic
parameters so each test verifies the unit under test against ground
truth it constructed itself, free of imaging or labelling artefacts.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# Force a non-interactive matplotlib backend before any test imports pyplot,
# so plot-creator smoke tests run in headless CI without a display server.
os.environ.setdefault("MPLBACKEND", "Agg")


_APP_ROOT = Path(__file__).resolve().parent.parent / "src" / "opennta"


# Importing opennta triggers ``from . import application`` in
# opennta/__init__.py, which pulls in PyQt5/QtWebEngine. Lean CI environments
# don't have Qt available, so register a stub module under that name before
# the first opennta import below so the package import resolves without Qt.
#
# We still set ``__path__`` on the stub to the real package directory so that
# submodule resolution (``import opennta.application.config.manager``)
# keeps working for the Qt-independent helpers — the stub merely skips the
# Qt-laden eager imports in the real ``__init__`` body.
def _stub_qt_gui_modules() -> None:
    name = "opennta.application"
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(_APP_ROOT / "application")]
        sys.modules[name] = mod


# Minimal PyQt5 stub: a handful of Qt-free helpers under opennta.application
# transitively import a Qt symbol only for a TypedDict annotation (e.g.
# ``ReportTemplates`` -> ``QTreeWidgetItem``). Registering a stub module is
# cheaper than mocking each consumer at test time, and only kicks in when
# PyQt5 is genuinely absent.
def _stub_pyqt_for_annotations() -> None:
    if "PyQt5" in sys.modules:
        return
    pyqt5 = types.ModuleType("PyQt5")
    qtwidgets = types.ModuleType("PyQt5.QtWidgets")
    qtwidgets.QTreeWidgetItem = type("QTreeWidgetItem", (), {})
    pyqt5.QtWidgets = qtwidgets
    sys.modules["PyQt5"] = pyqt5
    sys.modules["PyQt5.QtWidgets"] = qtwidgets


_stub_qt_gui_modules()
try:
    import PyQt5  # noqa: F401
except ImportError:
    _stub_pyqt_for_annotations()


# With src-layout, the tests/ tree lives at the repository root rather than
# inside the installed opennta package, but the existing test modules import
# helpers as ``opennta.tests._helpers``. Register a namespace stub that points
# ``opennta.tests`` at this directory so those imports keep resolving without
# touching every test file.
def _register_tests_namespace() -> None:
    if "opennta.tests" in sys.modules:
        return
    tests_root = Path(__file__).resolve().parent
    mod = types.ModuleType("opennta.tests")
    mod.__path__ = [str(tests_root)]
    sys.modules["opennta.tests"] = mod


_register_tests_namespace()

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Importing the analysis and tracking packages triggers @register_corrector /
# @register_distributor / @register_method / @register_detector / @register_linker
# side effects on every plugin subpackage, so registry-driven parametrized
# tests can enumerate the complete set of registered modules at collection
# time. Without this, plugins like numerical/ml/iterative/ftla/trackmate would
# stay invisible and their contract tests would never be generated.
import opennta.analysis  # noqa: E402,F401

# Importing this module registers the test-only Fake detector/linker plugins
# (@register_detector / @register_linker), so the detector/linker and
# SplitMethod contract tests have a real plugin to exercise instead of
# collapsing to an empty, assertion-free parametrization.
import opennta.tests._helpers.fakes  # noqa: E402,F401
import opennta.tracking  # noqa: E402,F401
from opennta.analysis.types import AnalysisConfig  # noqa: E402
from opennta.common.progress import ProgressEmitter  # noqa: E402
from opennta.tests._helpers import synth as _synth  # noqa: E402
from opennta.tracking.types import ProcessingParams  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: end-to-end pipeline tests; slower than unit tests.",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that take noticeably longer (>1s) and may be deselected "
        "in pre-commit runs via ``pytest -m 'not slow'``.",
    )


@pytest.fixture(scope="session")
def analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        sensor_size=6.5,
        magnification=20,
        fps=25.0,
        temp=298.0,
        eta=0.00089,
    )


@pytest.fixture
def default_analysis_config() -> AnalysisConfig:
    return AnalysisConfig()


@pytest.fixture
def processing_params() -> ProcessingParams:
    return ProcessingParams()


@pytest.fixture
def progress_emitter(tmp_path) -> ProgressEmitter:
    return ProgressEmitter(save_path=str(tmp_path))


@pytest.fixture(scope="session")
def synthetic_brownian_params() -> dict[str, float | int]:
    return {
        "target_diameter_nm": _synth.DEFAULT_TARGET_DIAMETER_NM,
        "n_tracks": _synth.DEFAULT_N_TRACKS,
        "n_frames": _synth.DEFAULT_N_FRAMES,
        "lag_frame": _synth.DEFAULT_LAG_FRAME,
        "drift_vx_um_per_s": _synth.DEFAULT_DRIFT_VX_UM_PER_S,
        "drift_vy_um_per_s": _synth.DEFAULT_DRIFT_VY_UM_PER_S,
    }


@pytest.fixture(scope="session")
def synthetic_spots_df(analysis_config: AnalysisConfig) -> pd.DataFrame:
    return _synth.brownian_spots_df(analysis_config)


@pytest.fixture(scope="session")
def synthetic_spots_csv(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_spots_df: pd.DataFrame,
) -> Path:
    out_dir = tmp_path_factory.mktemp("synthetic_spots")
    return _synth.write_spots_csv(synthetic_spots_df, out_dir)
