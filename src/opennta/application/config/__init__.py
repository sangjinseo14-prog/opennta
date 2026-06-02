"""User-config dialog and persistent settings store."""

from .dialog import ConfigDialog
from .manager import ConfigManager, ensure_app_config_dir, ensure_app_temp_dir

__all__ = [
    "ConfigDialog",
    "ConfigManager",
    "ensure_app_config_dir",
    "ensure_app_temp_dir",
]
