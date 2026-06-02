"""Unit tests for ``opennta.analysis.mle.ftla_distributor``.

The bare-registry contract test in ``test_size_distributor.py`` only checks
the ``compute(None)`` empty-input path. These tests drive the distributor
against a fully synthesised monodisperse Brownian sample and verify that
the recovered size distribution peaks at the expected diameter.
"""
from __future__ import annotations

import numpy as np
import pytest

from opennta.analysis.mle.ftla_distributor import FTLADistributor


def test_recovers_monodisperse_peak(monodisperse_pipeline_state):
    state = monodisperse_pipeline_state
    out = FTLADistributor().compute(
        state["diameters_by_id"],
        min_d=10.0, max_d=1000.0, num_bins=80,
        msd_by_id=state["msd_by_id"],
        df=state["df"],
        config=state["config"],
        log_scale=True,
    )
    counts = out["counts"]
    centers = out["bin_centers"]
    assert counts is not None and centers is not None
    assert float(counts.sum()) > 0.0

    peak_nm = float(centers[int(np.argmax(counts))])
    assert peak_nm == pytest.approx(100.0, rel=0.30)

    fit = out["fit"]
    assert fit["method"] == "saveyn2010-parametric"
    assert fit["family"] == "lognormal"
    # theta = [mu_log_d, log_sigma]; mu_log_d should be near log(100).
    assert fit["theta"][0] == pytest.approx(np.log(100.0), abs=0.4)


def test_requires_pipeline_inputs():
    with pytest.raises(ValueError, match="FTLA"):
        FTLADistributor().compute({1: 100.0})


def test_empty_diameters_returns_canonical_none():
    out = FTLADistributor().compute({})
    assert out == {"diameters": None, "bin_edges": None, "counts": None, "bin_centers": None}
