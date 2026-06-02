import json
import logging
import os
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from ...analysis import units
from ...analysis.types import AnalysisConfig, user_config_fields

logger = logging.getLogger(__name__)

_USER_FIELDS: tuple[str, ...] = user_config_fields(AnalysisConfig)


def config_to_persisted(config: AnalysisConfig) -> dict[str, Any]:
    """Serialise a config to the on-disk nested ``{value, unit}`` layout, with
    each value expressed in its default *display* unit (e.g. eta in mPa·s)."""
    out: dict[str, Any] = {}
    for name in _USER_FIELDS:
        canonical = getattr(config, name)
        if units.has_units(name):
            unit = units.default_display_unit(name)
            out[name] = {"value": units.from_canonical(name, canonical, unit), "unit": unit}
        else:
            out[name] = canonical
    return out


def persisted_to_canonical(data: dict[str, Any]) -> dict[str, float]:
    """Read the persisted config into canonical float values, validating units.

    Accepts both the nested ``{value, unit}`` layout and bare scalars. A bare
    scalar is treated as already-canonical for backward compatibility with
    legacy flat config files."""
    kwargs: dict[str, float] = {}
    for name in _USER_FIELDS:
        entry = data[name]
        if isinstance(entry, dict):
            if "value" not in entry:
                raise ValueError(f"Config field {name!r} is missing its 'value'.")
            kwargs[name] = units.to_canonical(name, entry["value"], entry.get("unit"))
        else:
            # Legacy flat value: already stored in the canonical unit.
            kwargs[name] = float(entry)
    return kwargs

def ensure_app_config_dir() -> Path:
    config_dir = Path(user_config_dir()) / "opennta"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def ensure_app_temp_dir() -> Path:
    temp_dir = ensure_app_config_dir() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


class ConfigManager:
    def __init__(self, parent):
        temp_dir = ensure_app_temp_dir()
        self.temp_dir: str = str(temp_dir)
        self.config_path: str = os.path.join(self.temp_dir, "config.json")
        self.parent = parent

        if not os.path.exists(self.config_path):
            self._write_default_config()

    def _write_default_config(self) -> None:
        data = config_to_persisted(AnalysisConfig())
        try:
            self.save_to_temp(data)
        except OSError as e:
            logger.warning("Could not write default config: %s", e)

    def load_from_temp(self) -> dict[str, Any] | None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Config load error: %s", e)
                return None
        return None

    def save_to_temp(self, data):
        # Propagate write errors so callers (e.g. the Config dialog) can surface
        # a real failure instead of falsely reporting success.
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: str | Path) -> dict[str, Any]:
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def save_to_file(self, path: str | Path, data: dict[str, Any]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def desktop_path():
        desktop = Path.home() / "Desktop"
        return str(desktop if desktop.exists() else Path.home())

    def get_config(self) -> AnalysisConfig:
        data = self.load_from_temp()
        if data is None:
            logger.warning("Config file missing; falling back to defaults.")
            self._write_default_config()
            return AnalysisConfig()

        missing = [k for k in _USER_FIELDS if k not in data]
        if missing:
            raise RuntimeError(
                f"Config is missing required keys: {', '.join(missing)}\n"
                "Please update settings via the Config dialog."
            )

        try:
            return AnalysisConfig(**persisted_to_canonical(data))
        except (ValueError, TypeError) as e:
            raise RuntimeError(
                f"Invalid config values in {self.config_path}: {e}\n"
                "Please fix settings via the Config dialog."
            ) from e
