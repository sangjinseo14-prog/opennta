from __future__ import annotations

import numpy as np
import pytest

from opennta.analysis.size_distributor import (
    DirectDistributor,
    SizeDistributor,
    get_size_distributor,
    registered_size_distribution_modes,
)
from opennta.tests._helpers.factories import SIZE_FACTORIES, SKIPPED_MODES

ALL_SIZE_MODES = sorted(registered_size_distribution_modes())


def test_direct_log_binning_shapes():
    diameters = {i: float(d) for i, d in enumerate(np.linspace(10, 500, 200))}
    out = DirectDistributor().compute(
        diameters, min_d=1.0, max_d=1000.0, num_bins=50, log_scale=True
    )

    assert out["bin_edges"].shape == (51,)
    assert out["counts"].shape == (50,)
    assert out["bin_centers"].shape == (50,)
    assert out["counts"].sum() == len(diameters)
    # log spacing => geometric centers
    assert np.allclose(
        out["bin_centers"], np.sqrt(out["bin_edges"][:-1] * out["bin_edges"][1:])
    )


def test_direct_linear_binning_shapes():
    diameters = {i: float(d) for i, d in enumerate(np.linspace(10, 100, 50))}
    out = DirectDistributor().compute(
        diameters, min_d=0.0, max_d=120.0, num_bins=12, log_scale=False
    )

    assert out["bin_edges"].shape == (13,)
    assert np.allclose(np.diff(out["bin_edges"]), 10.0)
    # linear midpoints
    assert np.allclose(
        out["bin_centers"], 0.5 * (out["bin_edges"][:-1] + out["bin_edges"][1:])
    )


def test_direct_filters_non_positive_diameters():
    diameters = {1: 50.0, 2: 0.0, 3: -3.0, 4: 100.0}
    out = DirectDistributor().compute(diameters)
    assert set(out["diameters"].tolist()) == {50.0, 100.0}


# Note: DirectDistributor.compute(None) returning the canonical 4-key all-None
# dict is asserted by the parametrized test_size_distributor_empty_input_contract
# below ("direct" is a registered, non-skipped mode), so it is not duplicated here.


def test_size_distributor_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown size distribution mode"):
        get_size_distributor("does_not_exist")


@pytest.mark.parametrize("mode", ALL_SIZE_MODES)
def test_size_distributor_registry_returns_subclass(mode):
    d = get_size_distributor(mode)
    assert isinstance(d, SizeDistributor)
    assert d.mode_key == mode
    assert d.get_display_name()


@pytest.mark.parametrize("mode", ALL_SIZE_MODES)
def test_size_distributor_empty_input_contract(mode):
    """Every registered distributor must return the canonical 4-key result
    dict (all None) when given empty input."""
    if mode in SKIPPED_MODES:
        pytest.skip(SKIPPED_MODES[mode])

    factory = SIZE_FACTORIES.get(mode, lambda _mode=mode: get_size_distributor(_mode))
    out = factory().compute(None)
    for key in ("diameters", "bin_edges", "counts", "bin_centers"):
        assert key in out
        assert out[key] is None
