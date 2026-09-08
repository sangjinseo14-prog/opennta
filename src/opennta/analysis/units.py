"""Unit registry and conversions for the user-facing acquisition config.

Values are persisted next to their unit so a number entered in display units
(e.g. viscosity in mPa·s) is unambiguously converted to the SI-canonical unit
the analysis math consumes (Pa·s); a bare number cannot be misread as the wrong
unit.

    canonical = display * factor + offset
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSpec:
    canonical: str
    default_display: str
    # display unit -> (factor, offset) such that canonical = display*factor + offset
    conversions: dict[str, tuple[float, float]]


# Canonical units must match how ``AnalysisConfig`` fields are consumed
# downstream: eta in Pa·s (Stokes-Einstein, SI) and temp in Kelvin (Boltzmann
# term). Field names are the ``AnalysisConfig`` user_config field names.
CONFIG_UNITS: dict[str, UnitSpec] = {
    "sensor_size": UnitSpec("µm", "µm", {"µm": (1.0, 0.0), "um": (1.0, 0.0)}),
    "magnification": UnitSpec("X", "X", {"X": (1.0, 0.0), "": (1.0, 0.0)}),
    "fps": UnitSpec("Hz", "Hz", {"Hz": (1.0, 0.0), "fps": (1.0, 0.0), "": (1.0, 0.0)}),
    "exposure_time": UnitSpec(
        "s",
        "ms",
        {
            "s": (1.0, 0.0),
            "ms": (1e-3, 0.0),
            "µs": (1e-6, 0.0),
            "us": (1e-6, 0.0),
        },
    ),
    "temp": UnitSpec(
        "K", "K",
        {"K": (1.0, 0.0), "°C": (1.0, 273.15), "C": (1.0, 273.15), "celsius": (1.0, 273.15)},
    ),
    "eta": UnitSpec(
        "Pa·s", "mPa·s",
        {
            "Pa·s": (1.0, 0.0), "Pa.s": (1.0, 0.0), "Pas": (1.0, 0.0),
            "mPa·s": (1e-3, 0.0), "mPa.s": (1e-3, 0.0), "mPas": (1e-3, 0.0),
            "cP": (1e-3, 0.0),
        },
    ),
}


def has_units(field: str) -> bool:
    return field in CONFIG_UNITS


def accepted_units(field: str) -> list[str]:
    return list(CONFIG_UNITS[field].conversions)


def default_display_unit(field: str) -> str:
    return CONFIG_UNITS[field].default_display


def _lookup_conversion(field: str, unit: str | None) -> tuple[float, float, str]:
    spec = CONFIG_UNITS[field]
    u = spec.default_display if unit is None else str(unit).strip()
    if u not in spec.conversions:
        raise ValueError(
            f"Unknown unit {unit!r} for field {field!r}; "
            f"accepted units: {accepted_units(field)}"
        )
    factor, offset = spec.conversions[u]
    return factor, offset, u


def to_canonical(field: str, value: float, unit: str | None) -> float:
    # Raises ValueError on an unrecognised unit or a non-numeric value.
    factor, offset, _ = _lookup_conversion(field, unit)
    return float(value) * factor + offset


def from_canonical(field: str, value: float, unit: str | None = None) -> float:
    # unit defaults to the field's display unit.
    factor, offset, _ = _lookup_conversion(field, unit)
    return (float(value) - offset) / factor


__all__ = [
    "UnitSpec",
    "CONFIG_UNITS",
    "has_units",
    "accepted_units",
    "default_display_unit",
    "to_canonical",
    "from_canonical",
]
