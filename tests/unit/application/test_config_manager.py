"""Smoke tests for ``opennta.application.config.manager``.

``ConfigManager`` is Qt-free except for being shipped under the
Qt-bearing ``application`` package, so we can exercise its config
round-trip headlessly by redirecting its temp-dir helper to ``tmp_path``.
"""
from __future__ import annotations

import json

import pytest

from opennta.analysis.types import AnalysisConfig, user_config_fields
from opennta.application.config import manager as cm_mod
from opennta.application.config.manager import ConfigManager

_USER_FIELDS = user_config_fields(AnalysisConfig)


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    """Redirect ``ensure_app_temp_dir`` so ConfigManager writes inside
    pytest's tmp_path instead of the real platformdirs location."""
    monkeypatch.setattr(cm_mod, "ensure_app_temp_dir", lambda: tmp_path)
    return tmp_path


def test_init_writes_default_config_with_all_user_fields(isolated_config_dir):
    manager = ConfigManager(parent=None)

    assert (isolated_config_dir / "config.json").exists()
    with open(manager.config_path, encoding="utf-8") as f:
        data = json.load(f)
    assert set(_USER_FIELDS).issubset(data.keys())

    # Values are persisted in the nested {value, unit} layout, in display units.
    assert data["eta"]["unit"] == "mPa·s"
    assert data["eta"]["value"] == pytest.approx(0.9544)
    assert data["temp"]["unit"] == "K"

    # ...and round-trip back to the canonical AnalysisConfig defaults.
    defaults = AnalysisConfig()
    config = manager.get_config()
    for field in _USER_FIELDS:
        assert getattr(config, field) == pytest.approx(getattr(defaults, field))


def test_get_config_converts_display_units_to_canonical(isolated_config_dir):
    """Viscosity entered in mPa·s must reach AnalysisConfig as Pa·s (the
    historical 1000x trap), and temperature accepts °C."""
    manager = ConfigManager(parent=None)
    data = {
        "sensor_size": {"value": 6.5, "unit": "µm"},
        "magnification": {"value": 20, "unit": "X"},
        "fps": {"value": 30.0, "unit": "Hz"},
        "temp": {"value": 25.0, "unit": "°C"},
        "eta": {"value": 0.89, "unit": "mPa·s"},
    }
    manager.save_to_temp(data)

    config = manager.get_config()
    assert config.eta == pytest.approx(0.00089)      # 0.89 mPa·s -> Pa·s
    assert config.temp == pytest.approx(298.15)      # 25 °C -> K


def test_get_config_rejects_unknown_unit(isolated_config_dir):
    manager = ConfigManager(parent=None)
    data = {f: {"value": 1.0, "unit": "X"} for f in _USER_FIELDS}
    data["eta"] = {"value": 1.0, "unit": "poise"}
    manager.save_to_temp(data)

    with pytest.raises(RuntimeError, match="Unknown unit"):
        manager.get_config()


def test_get_config_round_trip_returns_analysis_config(isolated_config_dir):
    manager = ConfigManager(parent=None)
    manager.save_to_temp({field: 1.5 for field in _USER_FIELDS})

    config = manager.get_config()

    assert isinstance(config, AnalysisConfig)
    for field in _USER_FIELDS:
        assert getattr(config, field) == pytest.approx(1.5)


def test_get_config_raises_on_missing_keys(isolated_config_dir):
    manager = ConfigManager(parent=None)
    # Intentionally drop a required user field; get_config must refuse to
    # silently fall back so a corrupted config file is surfaced loudly.
    partial = {field: 1.0 for field in _USER_FIELDS[:-1]}
    manager.save_to_temp(partial)

    with pytest.raises(RuntimeError, match="missing required keys"):
        manager.get_config()
