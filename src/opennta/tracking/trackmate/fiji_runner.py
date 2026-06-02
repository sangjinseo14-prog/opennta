from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ...common.progress import ProgressEmitter
from ..types import ProcessingParams


class FijiRunner:

    def __init__(self, fiji_path: str, emitter: ProgressEmitter | None = None):
        self.fiji_path: str = fiji_path
        self.emitter = emitter or ProgressEmitter()
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def cancel(self) -> None:
        # Cooperatively stop a running Fiji subprocess. Safe to call from another
        # thread (dialog reject / window close); the read loop sees _cancelled.
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def run_threshold_detection(self, script_path: str, image_path: str, radius: int) -> bool:
        extra_params: dict[str, Any] = {"RADIUS": radius}
        success, fiji_lines = self._execute_fiji(script_path, image_path, extra_params, full=False)

        if not success:
            self._dump_fiji_output(fiji_lines)
            self.emitter.emit(msg="TrackMate execution failed")
            return False

        quality_csv = os.path.splitext(image_path)[0] + "_quality.csv"

        if not os.path.exists(quality_csv):
            self._dump_fiji_output(fiji_lines)
            self.emitter.emit(msg="Expected quality CSV not found")
            return False

        try:
            os.remove(image_path)
        except Exception as e:
            self.emitter.emit(msg=f"First-frame cleanup skipped: {e}")

        return True

    def run_full_tracking(
        self,
        script_path: str,
        image_path: str,
        threshold: float,
        params: ProcessingParams,
    ) -> bool:
        extra_params: dict[str, Any] = {
            "qualityThreshold": float(round(threshold, 6)),
            "RADIUS": params.particle_radius_pixels,
            "LINKING_MAX_DISTANCE": params.linking_max_distance,
            "GAP_CLOSING_MAX_DISTANCE": params.gap_closing_max_distance,
            "MAX_FRAME_GAP": params.max_frame_gap,
            "TRACK_LENGTH_MIN": params.min_track_frames,
            "borderMargin": params.border_margin_pixels,
        }

        success, fiji_lines = self._execute_fiji(script_path, image_path, extra_params, full=True)

        if not success:
            self._dump_fiji_output(fiji_lines)
            return False

        try:
            os.remove(image_path)
        except Exception as exc:
            self.emitter.emit(msg=f"Could not remove processed stack: {exc}")

        return True

    def _dump_fiji_output(self, lines: list[str]) -> None:
        for ln in lines:
            self.emitter.emit(msg=f"[Fiji] {ln}")

    def _execute_fiji(
        self,
        script_path: str,
        image_path: str,
        params: dict[str, Any] | None = None,
        full: bool = True,
        only_tag: str | None = "#TM",
    ) -> tuple[bool, list[str]]:
        script_win = os.path.normpath(script_path)
        buffer: list[str] = []

        def _sci_literal(v: Any) -> str | None:
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, int):
                return str(int(v))
            if isinstance(v, float):
                return f"{v:.6f}" if np.isfinite(v) else None
            s = str(v).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{s}"'

        img_lit = _sci_literal(Path(image_path).as_posix())
        kv: list[str] = [f"inputFile={img_lit}"]
        if params:
            for k, v in params.items():
                lit = _sci_literal(v)
                if lit is None:
                    self.emitter.emit(msg=f"Invalid non-finite parameter for Fiji: {k}={v}")
                    return False, buffer
                kv.append(f"{k}={lit}")
        param_str = ",".join(kv)

        cmd: list[str] = [self.fiji_path, "--headless", "--console", "--run", script_win, param_str]

        env = os.environ.copy()
        env.setdefault("SCIJAVA_LOG_LEVEL", "info")

        if self._cancelled:
            return False, buffer

        timeout_sec = 3600
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        ) as proc:
            self._proc = proc
            try:
                for line in iter(proc.stdout.readline, ""):
                    if self._cancelled:
                        break
                    s = line.rstrip()
                    if not s:
                        continue
                    buffer.append(s)
                    if only_tag:
                        if only_tag in s:
                            self.emitter.emit(msg=s)
                    else:
                        self.emitter.emit(msg=f"[Fiji] {s}")
                proc.stdout.close()
                if self._cancelled:
                    self.emitter.emit(msg="Fiji execution cancelled")
                    buffer.append("Fiji execution cancelled")
                    return False, buffer
                ret = proc.wait(timeout=timeout_sec)
                return ret == 0, buffer
            except subprocess.TimeoutExpired:
                self.emitter.emit(msg=f"Fiji execution timed out after {timeout_sec}s")
                buffer.append(f"Fiji execution timed out after {timeout_sec}s")
                proc.kill()
                proc.wait()
                return False, buffer
            finally:
                self._proc = None
