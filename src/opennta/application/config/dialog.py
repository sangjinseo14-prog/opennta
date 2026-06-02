import logging

from PyQt5.QtWidgets import QDialog, QMessageBox

from ...analysis import units
from ..common.fonts import get_app_font
from ..common.path_selector import PathSelector
from ._ui_builder import _UIBuilderMixin
from .manager import ConfigManager

logger = logging.getLogger(__name__)


class ConfigDialog(_UIBuilderMixin, QDialog):

    CONFIG_FIELDS = {
        "le_sensor_size": "sensor_size",
        "le_magnification": "magnification",
        "le_fps": "fps",
        "le_temp": "temp",
        "le_eta": "eta",
    }

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.setFont(get_app_font())

        self.config_manager = config_manager
        self.parent_ui = parent

        self.btn_update.clicked.connect(self._apply_dialog_to_temp_config)
        self.btn_load.clicked.connect(self._load_config)
        self.btn_save.clicked.connect(self._save_config)

        self._load_temp_config_into_ui()

    def _apply_dialog_to_temp_config(self):
        try:
            previous = self.config_manager.load_from_temp() or {}
            data = self._collect_ui_field_values()

            before = [previous.get(k, "") for k in self.CONFIG_FIELDS.values()]
            after = [data.get(k, "") for k in self.CONFIG_FIELDS.values()]
            logger.info("Config updated: %s -> %s", before, after)

            self.config_manager.save_to_temp(data)

            QMessageBox.information(self, "Updated", f"Settings saved to:\n{self.config_manager.config_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Update failed:\n{e}")

    def _load_config(self):
        path, _ = PathSelector.load_config_file(
            self,
            self.config_manager.desktop_path()
        )

        if not path:
            return

        try:
            data = self.config_manager.load_from_file(path)

            # Preview-only: do not persist to temp from the load flow; only the
            # Update button is allowed to overwrite the live config.
            self._apply_config_to_ui(data)

            QMessageBox.information(self, "Loaded", f"Settings loaded from:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed:\n{e}")

    def _save_config(self):
        path, _ = PathSelector.save_config_file(
            self,
            self.config_manager.desktop_path(),
            "config.json"
        )

        if not path:
            return

        try:
            # Export-only: write to the chosen path without touching temp/config.json.
            data = self._collect_ui_field_values()
            self.config_manager.save_to_file(path, data)

            QMessageBox.information(self, "Saved", f"Settings saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed:\n{e}")

    def _field_display_units(self):
        # attr (le_eta) -> unit label shown next to the field ("mPa·s", ...).
        return {attr: unit for attr, _label, unit in self._FIELDS}

    def _collect_ui_field_values(self):
        # Persist each value next to the unit it was entered in, so the
        # downstream canonical conversion is unambiguous and verifiable.
        display_units = self._field_display_units()
        data = {}

        for ui_field, config_key in self.CONFIG_FIELDS.items():
            widget = getattr(self, ui_field, None)
            if widget and hasattr(widget, "text"):
                data[config_key] = {
                    "value": widget.text(),
                    "unit": display_units.get(ui_field, ""),
                }

        return data

    def _load_temp_config_into_ui(self):
        try:
            data = self.config_manager.load_from_temp()
            self._apply_config_to_ui(data)
        except Exception as e:
            logger.warning("Failed to load temp config into UI: %s", e)


    def _apply_config_to_ui(self, data):
        display_units = self._field_display_units()
        for ui_field, config_key in self.CONFIG_FIELDS.items():
            entry = data.get(config_key, data.get(ui_field))
            if entry is None:
                continue
            widget = getattr(self, ui_field, None)
            if not (widget and hasattr(widget, "setText")):
                continue
            widget.setText(
                self._entry_to_display_text(
                    config_key, entry, display_units.get(ui_field, "")
                )
            )

    @staticmethod
    def _entry_to_display_text(config_key, entry, display_unit):
        # Render an entry (nested {value, unit} or a legacy bare scalar) in the
        # field's fixed display unit. Units in the file are authoritative, so a
        # value saved in a different unit is converted before being shown.
        if not units.has_units(config_key):
            value = entry["value"] if isinstance(entry, dict) else entry
            return str(value)
        try:
            if isinstance(entry, dict):
                canonical = units.to_canonical(config_key, entry["value"], entry.get("unit"))
            else:
                canonical = float(entry)  # legacy: already canonical
            return str(units.from_canonical(config_key, canonical, display_unit))
        except (ValueError, TypeError, KeyError):
            # Fall back to the raw text so a malformed entry is still visible
            # and editable rather than silently dropped.
            value = entry["value"] if isinstance(entry, dict) else entry
            return str(value)
