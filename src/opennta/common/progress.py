"""Single ProgressEmitter shared by the analysis and tracking subsystems."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


def noop_progress_callback(status_code: int) -> None:
    pass


def noop_log_callback(message: str) -> None:
    pass


class ProgressEmitter:
    def __init__(
        self,
        save_path: str | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.save_path = save_path
        self.progress_callback: ProgressCallback = progress_callback or noop_progress_callback
        self.log_callback: LogCallback = log_callback or noop_log_callback
        self._log_buffer: list[str] = []
        self._log_filename: str | None = None

    def emit(self, code: int | None = None, msg: str | None = None) -> None:
        if code is not None:
            self.progress_callback(code)

        if msg:
            self._log_buffer.append(msg)
            self.log_callback(msg)

    def save_log(self) -> None:
        if not self._log_buffer:
            return

        log_dir = self._resolve_log_dir()

        try:
            if self._log_filename is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                self._log_filename = os.path.join(log_dir, f"trackmate_log_{ts}.txt")

            with open(self._log_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(self._log_buffer))

        except OSError as e:
            # Don't re-emit through self.emit() — would recurse if log_callback
            # is wired back into the same buffer.
            self._log_buffer.append(f"[Log] Failed to write log: {e}")

    def _resolve_log_dir(self) -> str:
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            return self.save_path

        desktop = Path.home() / "Desktop"
        base = desktop if desktop.exists() else Path.home()
        fallback = str(base / f"result_{time.strftime('%Y%m%d')}")
        os.makedirs(fallback, exist_ok=True)
        return fallback


__all__ = [
    "ProgressCallback",
    "LogCallback",
    "noop_progress_callback",
    "noop_log_callback",
    "ProgressEmitter",
]
