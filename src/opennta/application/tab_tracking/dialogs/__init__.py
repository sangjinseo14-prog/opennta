"""Qt configurator dialogs for tracking methods.

This package owns the application-side dispatcher that decides which dialog
matches a given tracking method. Domain method classes live under
`opennta.tracking.<...>` and stay PyQt-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from PyQt5.QtWidgets import QWidget

from opennta.tracking.tracking_method import (
    TrackingConfigurationResult,
    TrackingMethod,
)
from opennta.tracking.trackmate.method import TrackMateMethod
from opennta.tracking.types import FolderItem

from .trackmate.dialog import TrackMateDialog

__all__ = [
    "TrackMateDialog",
    "configure_method",
]


def configure_method(
    method: TrackingMethod,
    *,
    folder_items: Sequence[FolderItem],
    default_index: int,
    parent: QWidget | None,
) -> TrackingConfigurationResult | None:
    if isinstance(method, TrackMateMethod):
        dialog = TrackMateDialog(
            initial_params=method.params,
            fiji_path=method.fiji_path,
            folder_items=list(folder_items),
            default_index=default_index,
            script_threshold_path=method.script_threshold_path,
            parent=parent,
        )
        if dialog.exec_() != dialog.Accepted:
            return None
        return TrackingConfigurationResult(params=dialog.get_processing_params())

    raise TypeError(f"Unsupported tracking method: {type(method).__name__}")
