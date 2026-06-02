"""Cross-tab utilities shared by Qt application controllers.

Most helpers are imported directly from their submodules
(``path_selector``, ``report_templates``, ``statistics``).
Only ``get_app_font`` is re-exported here for the application entry point.
"""

from .fonts import get_app_font

__all__ = [
    "get_app_font",
]
