from __future__ import annotations

import gc
import os
from collections.abc import Sequence

import numpy as np
import tifffile as tiff
from numpy.typing import NDArray

from ...common.progress import ProgressEmitter


class ImageProcessor:

    def __init__(self, emitter: ProgressEmitter | None = None):
        self.emitter = emitter or ProgressEmitter()

    def normalize_stack(
        self,
        tiff_paths: Sequence[str],
        normalization_percentile: float,
    ) -> NDArray[np.uint16]:
        if not tiff_paths:
            raise ValueError("No TIFF files provided for processing.")

        stack = self._load_stack(tiff_paths[0])
        self.emitter.emit(msg=f"Loaded stack: {stack.shape[0]} frames")

        normalized = self._normalize_and_scale(stack, float(normalization_percentile))

        del stack
        gc.collect()
        return normalized

    def _load_stack(self, path: str) -> NDArray[np.uint16]:
        with tiff.TiffFile(path) as tf:
            arr: NDArray[np.uint16] = tf.asarray()
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        return arr

    def _normalize_and_scale(
        self,
        combined: NDArray[np.uint16],
        normalization_percentile: float,
    ) -> NDArray[np.uint16]:
        frame0:NDArray[np.uint16] = combined[0]

        vmin = float(frame0.min())
        percentile_val = float(np.percentile(frame0, normalization_percentile))
        vmax = min(3.0 * percentile_val, 65535.0)

        if vmax <= vmin:
            self.emitter.emit(msg="Warning: Invalid range detected")
            vmax = vmin + 1.0

        scale = 65535.0 / (vmax - vmin)

        out: NDArray[np.uint16] = np.empty_like(combined, dtype=np.uint16)
        CHUNK = 128
        total_frames = int(combined.shape[0])
        for start in range(0, total_frames, CHUNK):
            end = min(start + CHUNK, total_frames)
            tmp = combined[start:end].astype(np.float32, copy=False)
            tmp -= vmin
            tmp *= scale
            np.clip(tmp, 0.0, 65535.0, out=tmp)
            out[start:end] = tmp.astype(np.uint16, copy=False)
            del tmp
        self.emitter.emit(msg=f"Normalization completed (min: {vmin:.1f}, max: {vmax:.1f})")
        return out

    def save_stack_and_first_frame(self, result_stack: NDArray[np.uint16], output_path: str, sample_name: str) -> tuple[str, str]:
        stack_path = os.path.join(output_path, f"{sample_name}.tiff")
        first_frame_path = os.path.join(output_path, f"{sample_name}_first_frame.tiff")

        tiff.imwrite(stack_path, result_stack)
        tiff.imwrite(first_frame_path, result_stack[0])

        return stack_path, first_frame_path
