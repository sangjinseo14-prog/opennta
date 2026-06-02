"""Unit tests for ``opennta.analysis.mle.iterative_distributor``.

The bare-registry contract test in ``test_size_distributor.py`` only checks
the ``compute(None)`` empty-input path. These tests drive the distributor
against a fully synthesised monodisperse Brownian sample and verify that
the recovered size distribution peaks at the expected diameter.
"""
from __future__ import annotations

import numpy as np
import pytest

from opennta.analysis.mle.iterative_distributor import IterativeDistributor


def test_recovers_monodisperse_peak(monodisperse_pipeline_state):
    state = monodisperse_pipeline_state
    out = IterativeDistributor().compute(
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
    assert counts.shape == centers.shape
    assert float(counts.sum()) > 0.0

    peak_nm = float(centers[int(np.argmax(counts))])
    assert peak_nm == pytest.approx(100.0, rel=0.30)

    # Most of the mass should sit in a factor-of-2 band around the truth.
    in_band = (centers > 50.0) & (centers < 200.0)
    assert counts[in_band].sum() / counts.sum() > 0.5

    fit = out["fit"]
    assert fit["method"] == "walker2012-em"
    assert fit["n_tracks"] > 0


def test_requires_pipeline_inputs():
    """compute() with a non-empty diameters dict but missing msd/df/config
    must raise — the iterative distributor cannot synthesise the Gamma
    likelihood without them."""
    with pytest.raises(ValueError, match="Iterative"):
        IterativeDistributor().compute({1: 100.0})


def test_empty_diameters_returns_canonical_none():
    """Empty input should short-circuit before requiring the pipeline state."""
    out = IterativeDistributor().compute({})
    assert out == {"diameters": None, "bin_edges": None, "counts": None, "bin_centers": None}
