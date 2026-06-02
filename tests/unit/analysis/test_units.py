"""Unit tests for ``opennta.analysis.units``.

These pin the conversions that fix the historical "viscosity off by 1000x"
trap: the dialog enters viscosity in mPa·s but the math consumes Pa·s.
"""
from __future__ import annotations

import pytest

from opennta.analysis import units


def test_eta_mpas_to_pas():
    # Water ~0.89 mPa·s == 0.00089 Pa·s.
    assert units.to_canonical("eta", 0.89, "mPa·s") == pytest.approx(0.00089)


def test_eta_pas_is_identity():
    assert units.to_canonical("eta", 0.00089, "Pa·s") == pytest.approx(0.00089)


def test_temp_celsius_to_kelvin():
    assert units.to_canonical("temp", 25.0, "°C") == pytest.approx(298.15)


def test_temp_kelvin_is_identity():
    assert units.to_canonical("temp", 298.0, "K") == pytest.approx(298.0)


def test_from_canonical_uses_default_display_unit():
    assert units.from_canonical("eta", 0.00089) == pytest.approx(0.89)


@pytest.mark.parametrize(
    "field, value, unit",
    [("eta", 0.89, "mPa·s"), ("temp", 37.0, "°C"), ("sensor_size", 6.5, "µm")],
)
def test_round_trip(field, value, unit):
    canonical = units.to_canonical(field, value, unit)
    assert units.from_canonical(field, canonical, unit) == pytest.approx(value)


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown unit"):
        units.to_canonical("eta", 1.0, "poise")


def test_default_display_units():
    assert units.default_display_unit("eta") == "mPa·s"
    assert units.default_display_unit("temp") == "K"
