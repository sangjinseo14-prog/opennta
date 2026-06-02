"""End-to-end check for the registered ``"trackmate"`` tracking method.

Replaces per-component unit tests of TrackMateRunner / ImageProcessor /
ThresholdCalculator / Fitting / FittingModels with a single
result-oriented assertion: the registered tracking method, fed a noise-
free synthetic TIFF stack of N Gaussian-PSF particles, recovers
approximately N tracks via the real Fiji + TrackMate pipeline.

Fiji is an external binary so the test skips cleanly when
``OPENNTA_FIJI_PATH`` (or ``FIJI_PATH``) is not set or doesn't point at
an executable.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import tifffile

from opennta.common.progress import ProgressEmitter
from opennta.tests._helpers import synth as _synth
from opennta.tracking.tracking_method import (
    get_tracking_method,
    registered_method_modes,
)
from opennta.tracking.types import ProcessingParams

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _resolve_fiji_path() -> str | None:
    for var in ("OPENNTA_FIJI_PATH", "FIJI_PATH"):
        path = os.environ.get(var)
        if path and Path(path).is_file() and os.access(path, os.X_OK):
            return path
    return None


def test_trackmate_mode_is_registered():
    assert "trackmate" in registered_method_modes()


def test_trackmate_recovers_noiseless_particle_tracks(tmp_path):
    fiji_path = _resolve_fiji_path()
    if fiji_path is None:
        pytest.skip(
            "TrackMate requires Fiji; set OPENNTA_FIJI_PATH to an executable "
            "Fiji/ImageJ binary to enable this test."
        )

    n_particles = 12
    n_frames = 30
    stack = _synth.noiseless_particle_tiff_stack(
        n_frames=n_frames, n_particles=n_particles, seed=0,
    )

    sub_path = tmp_path / "sample"
    sub_path.mkdir()
    tiff_path = sub_path / "frames.tif"
    tifffile.imwrite(tiff_path, stack)

    save_path = tmp_path / "out"
    save_path.mkdir()

    params = ProcessingParams(
        method="trackmate",
        particle_radius_pixels=3,
        linking_max_distance=8,
        gap_closing_max_distance=8,
        max_frame_gap=1,
        min_track_frames=int(0.6 * n_frames),
    )
    method = get_tracking_method(
        mode="trackmate",
        params=params,
        emitter=ProgressEmitter(save_path=str(save_path)),
        save_path=str(save_path),
        fiji_path=fiji_path,
    )

    ok = method.track_subfolder(
        sub_path=str(sub_path),
        tiff_paths=[str(tiff_path)],
        nta_folder="sample",
        sub="frames",
    )
    assert ok, "TrackMate pipeline reported failure on a noiseless stack"

    spots_csvs = list((save_path / "sample").glob("*_spots.csv"))
    assert spots_csvs, f"No spots CSV produced under {save_path / 'sample'}"

    spots_df = pd.read_csv(spots_csvs[0])
    cols_upper = {str(c).strip().upper(): c for c in spots_df.columns}
    assert "ID" in cols_upper, f"Spots CSV missing ID column; got {list(spots_df.columns)}"

    n_recovered = spots_df[cols_upper["ID"]].nunique()
    assert n_recovered >= int(0.75 * n_particles), (
        f"Recovered {n_recovered} tracks from {n_particles} noiseless "
        f"particles (expected >= 75%)"
    )
