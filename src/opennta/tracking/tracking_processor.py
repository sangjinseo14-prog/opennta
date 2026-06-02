from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..common.progress import (
    LogCallback,
    ProgressCallback,
    ProgressEmitter,
    noop_log_callback,
    noop_progress_callback,
)
from .status_codes import TrackingStatus
from .tracking_method import TrackingMethod, get_tracking_method
from .types import FolderItem, ProcessingParams

logger = logging.getLogger(__name__)


class TrackingProcessor:

    def __init__(
        self,
        folder_items: Sequence[FolderItem],
        fiji_path: str,
        params: ProcessingParams | None = None,
        save_path: str | None = None,
        root_folder: str | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback| None = None,
    ):
        self.folder_items: Sequence[FolderItem] = folder_items
        self.fiji_path: str = fiji_path
        self.params: ProcessingParams = params or ProcessingParams()
        self.save_path: str | None = save_path
        self.root_folder: str | None = root_folder
        self.progress_callback: ProgressCallback = progress_callback or noop_progress_callback
        self.log_callback: LogCallback = log_callback or noop_log_callback

        self.emitter = ProgressEmitter(
            save_path=self.save_path,
            progress_callback=self.progress_callback,
            log_callback=self.log_callback,
        )

        self.success: bool = False
        self.total_count: int = len(folder_items)
        self.method: TrackingMethod | None = None
        self._cancelled = False

    def cancel(self) -> None:
        # Stop the batch between items and kill any in-flight Fiji subprocess.
        self._cancelled = True
        if self.method is not None and hasattr(self.method, "cancel"):
            self.method.cancel()

    def _get_or_create_save_path(self) -> str:
        if self.save_path and os.path.exists(self.save_path):
            return self.save_path

        desktop = str(Path.home() / "Desktop")

        root = Path(self.root_folder or desktop)
        root.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = root / f"OpenNTA_Results_{timestamp}"
        save_path.mkdir(exist_ok=True)

        self.save_path = str(save_path)
        return self.save_path

    def run_tracking_batch(self) -> bool:
        try:
            save_path = self._get_or_create_save_path()
            self.emitter.save_path = save_path

            self.method = get_tracking_method(
                mode=self.params.method,
                params=self.params,
                emitter=self.emitter,
                save_path=save_path,
                fiji_path=self.fiji_path,
            )

            self.emit_parameter_summary(
                params=self.params,
                fiji_path=self.fiji_path,
                folder_items=self.folder_items,
                )
            time.sleep(0.1)
            self.emitter.save_log()

            total_items = len(self.folder_items)
            groups: OrderedDict[str, list] = OrderedDict()
            for item in self.folder_items:
                groups.setdefault(item[2], []).append(item)
            n_groups = len(groups)

            self.emitter.emit(msg="=" * 80)
            self.emitter.emit(
                code=TrackingStatus.STARTED,
                msg=f"Tracking started: {n_groups} groups, {total_items} total folders",
            )
            self.emitter.emit(msg="=" * 80)

            total_success = 0
            total_failed = 0
            idx = 0

            for nta_folder, items in groups.items():
                if self._cancelled:
                    break
                group_size = len(items)
                group_success = 0
                group_failed = 0
                self.emitter.emit(
                    code=TrackingStatus.STARTED,
                    msg=f"[Group {nta_folder}] started: {group_size} folders",
                )

                for sub_path, tiff_paths, _nta_folder, sub in items:
                    if self._cancelled:
                        break
                    idx += 1
                    self.emitter.emit(
                        code=TrackingStatus.STARTED,
                        msg=f"Processing {idx}/{total_items}: {nta_folder}_{sub}",
                    )

                    if self.method.track_subfolder(sub_path, tiff_paths, nta_folder, sub):
                        group_success += 1
                    else:
                        group_failed += 1

                    time.sleep(0.1)
                    self.emitter.save_log()

                total_success += group_success
                total_failed += group_failed
                self.emitter.emit(
                    code=TrackingStatus.ITEM_COMPLETE,
                    msg=f"This group done: {group_success} success, {group_failed} failed",
                )
                self.emitter.emit(code=TrackingStatus.ITEM_COMPLETE)

            time.sleep(0.1)
            self.emitter.save_log()
            self.emitter.emit(
                code=TrackingStatus.ALL_COMPLETE,
                msg=f"All group done: {total_success} success, {total_failed} failed",
            )
            self.success = total_failed == 0
            return self.success

        except Exception as e:
            self.emitter.emit(code=TrackingStatus.ERROR_GENERAL, msg=str(e))
            return False

    def emit_parameter_summary(
        self,
        params: ProcessingParams,
        fiji_path: str | None = None,
        folder_items: Sequence[Any] | None = None,
    ) -> None:

        param_log = []
        param_log.append(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        param_log.append("")

        param_log.append("=" * 80)
        param_log.append("Tracking - Configuration Parameters")
        param_log.append("=" * 80)
        param_log.append("")

        method_name = self.method.get_display_name() if self.method else params.method
        param_log.append("[ Tracking Method ]")
        param_log.append(f"  Name: {method_name}")
        param_log.append(f"  Mode: {params.method}")
        param_log.append("")

        param_log.append("[ Normalization ]")
        param_log.append(f"  Normalization percentile: {params.normalization_percentile}")
        param_log.append("")

        param_log.append("[ LoG Detection ]")
        param_log.append(f"  Particle radius (pixels): {params.particle_radius_pixels}")
        param_log.append(f"  Fit fraction (FRAC): {params.fit_fraction}")
        param_log.append(f"  Significance alpha (ALPHA): {params.significance_alpha}")
        param_log.append("")

        if self.method is not None:
            param_log.extend(self.method.get_parameter_log_lines())

        param_log.append("[ Processing ]")
        param_log.append(f"  FIJI Path: {fiji_path}")
        param_log.append(f"  Save Path: {self.save_path}")
        param_log.append(f"  Total Items: {len(folder_items)}")

        for line in param_log:
            self.emitter.emit(msg=line)


__all__ = [
    "FolderItem",
    "ProcessingParams",
    "TrackingProcessor",
]
