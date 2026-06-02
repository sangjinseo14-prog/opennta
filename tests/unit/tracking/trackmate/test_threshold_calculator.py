"""Unit tests for ``ThresholdCalculator`` — the CSV-facing threshold step.

``ThresholdCalculator`` parses the quality CSV that Fiji writes, locates the
``QUALITY`` column (headered, headerless, or oddly-cased), and delegates to
``FittingEngine``. The parsing and error-handling branches are entirely
Fiji-independent, so they are driven here with synthetic quality CSVs written
to ``tmp_path``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opennta.common.progress import ProgressEmitter
from opennta.tests._helpers import synth as _synth
from opennta.tracking.trackmate.threshold_calculator import ThresholdCalculator
from opennta.tracking.trackmate.types import ThresholdResult


@pytest.fixture
def emitter(tmp_path) -> ProgressEmitter:
    return ProgressEmitter(save_path=str(tmp_path))


def _write_quality_csv(path, values, *, header) -> None:
    df = pd.DataFrame({"QUALITY": values})
    df.to_csv(path, index=False, header=header)


def test_headered_quality_csv_yields_threshold(tmp_path, emitter):
    values, n_spike, _mu, _sigma, _spike = _synth.quality_samples(seed=10)
    csv = tmp_path / "quality.csv"
    _write_quality_csv(csv, values, header=True)

    calc = ThresholdCalculator(emitter, model_name="Gaussian")
    out = calc.calculate_threshold_from_csv(str(csv), alpha=1e-4)

    assert out is not None
    assert out["ok"]  # np.bool_(True); truthy check (not identity)
    assert np.isfinite(out["u_star_alpha"])
    assert out["n_ge_thresh"] >= n_spike


def test_headerless_single_column_csv_is_parsed(tmp_path, emitter):
    values, *_ = _synth.quality_samples(seed=11)
    csv = tmp_path / "headerless.csv"
    _write_quality_csv(csv, values, header=False)

    calc = ThresholdCalculator(emitter, model_name="Gaussian")
    out = calc.calculate_threshold_from_csv(str(csv), alpha=1e-4)

    assert out is not None
    assert out["ok"]


def test_lowercase_quality_header_is_matched(tmp_path, emitter):
    values, *_ = _synth.quality_samples(seed=12)
    csv = tmp_path / "lower.csv"
    pd.DataFrame({"quality": values}).to_csv(csv, index=False)

    calc = ThresholdCalculator(emitter, model_name="Gaussian")
    out = calc.calculate_threshold_from_csv(str(csv), alpha=1e-4)

    assert out is not None and out["ok"]


def test_missing_quality_column_returns_none(tmp_path, emitter):
    csv = tmp_path / "no_quality.csv"
    # Multi-column, headerless, no quality column → cannot recover.
    pd.DataFrame(np.zeros((50, 3))).to_csv(csv, index=False, header=False)

    calc = ThresholdCalculator(emitter, model_name="Gaussian")
    assert calc.calculate_threshold_from_csv(str(csv)) is None


def test_file_not_found_returns_none(tmp_path, emitter):
    calc = ThresholdCalculator(emitter, model_name="Gaussian")
    assert calc.calculate_threshold_from_csv(str(tmp_path / "missing.csv")) is None


def test_empty_csv_returns_none(tmp_path, emitter):
    # A header-only quality CSV (zero rows) can't be fit: pandas yields an
    # empty float column and the downstream fit raises, which the calculator
    # converts to a None return rather than propagating.
    csv = tmp_path / "empty.csv"
    pd.DataFrame({"QUALITY": []}).to_csv(csv, index=False)

    calc = ThresholdCalculator(emitter, model_name="Gaussian")
    assert calc.calculate_threshold_from_csv(str(csv)) is None


def test_calculate_threshold_from_array_returns_result(emitter):
    values, *_ = _synth.quality_samples(seed=13)
    calc = ThresholdCalculator(emitter, model_name="Gaussian")
    result = calc.calculate_threshold_from_array(values, alpha=1e-4)
    assert isinstance(result, ThresholdResult)
    assert result.ok
