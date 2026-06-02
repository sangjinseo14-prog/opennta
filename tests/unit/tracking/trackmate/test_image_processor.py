"""Unit tests for ``ImageProcessor`` — TIFF normalisation for the Fiji step.

``normalize_stack`` percentile-normalises a TIFF stack to full uint16 range
before handing it to Fiji. The loading (2D vs 3D), the chunked rescale, and the
degenerate flat-frame guard are all Fiji-independent and are exercised here on
synthetic stacks written to ``tmp_path``.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile as tiff

from opennta.common.progress import ProgressEmitter
from opennta.tracking.trackmate.image_processor import ImageProcessor


@pytest.fixture
def emitter(tmp_path) -> ProgressEmitter:
    return ProgressEmitter(save_path=str(tmp_path))


def _write_stack(path, arr) -> str:
    tiff.imwrite(path, arr)
    return str(path)


def test_empty_paths_raises(emitter):
    with pytest.raises(ValueError, match="No TIFF files"):
        ImageProcessor(emitter).normalize_stack([], normalization_percentile=99.0)


def test_normalize_stretches_and_saturates_bright_outliers(tmp_path, emitter):
    # The transform is vmin=frame0.min(), vmax=min(3*p99, 65535): a dim
    # background maps to 0 and any pixel above 3x the 99th percentile saturates
    # to 65535. A flat background plus a sparse bright spot exercises both ends.
    frame = np.full((32, 32), 200, dtype=np.uint16)
    frame[0:3, 0:3] = 60000  # < 1% of pixels, so p99 stays at the background
    stack = np.repeat(frame[np.newaxis, ...], 5, axis=0)
    path = _write_stack(tmp_path / "stack.tif", stack)

    out = ImageProcessor(emitter).normalize_stack([path], normalization_percentile=99.0)

    assert out.dtype == np.uint16
    assert out.shape == stack.shape
    assert out.max() == 65535  # the bright spot saturates
    assert out.min() == 0      # the background floor maps to zero


def test_load_promotes_2d_frame_to_single_frame_stack(tmp_path, emitter):
    frame = np.full((16, 16), 500, dtype=np.uint16)
    frame[0, 0] = 4000  # give the percentile something to work with
    path = _write_stack(tmp_path / "single.tif", frame)

    out = ImageProcessor(emitter).normalize_stack([path], normalization_percentile=99.0)

    assert out.ndim == 3
    assert out.shape[0] == 1


def test_flat_frame_does_not_crash(tmp_path, emitter):
    # vmax <= vmin degenerate case: a perfectly constant first frame.
    stack = np.full((4, 8, 8), 1234, dtype=np.uint16)
    path = _write_stack(tmp_path / "flat.tif", stack)

    out = ImageProcessor(emitter).normalize_stack([path], normalization_percentile=99.0)

    assert out.dtype == np.uint16
    assert out.shape == stack.shape


def test_chunked_rescale_matches_unchunked_reference(tmp_path, emitter):
    # More than one 128-frame chunk, so the chunk loop is exercised and must
    # agree with a direct single-shot computation of the same transform.
    rng = np.random.default_rng(1)
    stack = rng.integers(0, 2000, size=(200, 8, 8), dtype=np.uint16)
    path = _write_stack(tmp_path / "big.tif", stack)

    out = ImageProcessor(emitter).normalize_stack([path], normalization_percentile=99.0)

    frame0 = stack[0].astype(float)
    vmin = frame0.min()
    vmax = min(3.0 * np.percentile(frame0, 99.0), 65535.0)
    scale = 65535.0 / (vmax - vmin)
    expected = np.clip((stack.astype(np.float32) - vmin) * scale, 0, 65535).astype(np.uint16)

    assert np.array_equal(out, expected)


def test_save_stack_and_first_frame_writes_two_files(tmp_path, emitter):
    stack = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    stack_path, first_frame_path = ImageProcessor(emitter).save_stack_and_first_frame(
        stack, str(tmp_path), "sample"
    )

    assert tiff.imread(stack_path).shape == stack.shape
    assert tiff.imread(first_frame_path).shape == stack[0].shape
