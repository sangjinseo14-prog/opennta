"""Shared synthetic-data generators for opennta tests.

All test inputs are built in-process from analytic parameters so the unit
under test is checked against ground truth the test constructed itself,
free of imaging or labelling artefacts. These generators were previously
re-implemented across several test modules with slight drift between
copies; consolidating them here makes the noise / drift / diffusion model
a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opennta.analysis.types import AnalysisConfig

# Brownian-trajectory parameters chosen so that with the default
# ``AnalysisConfig`` and ``lag_frame=2`` the recovered diameter distribution
# converges to ``target_diameter_nm`` within a few percent across the
# ``n_tracks`` ensemble, while leaving a known mean drift for the drift
# corrector to remove.
DEFAULT_TARGET_DIAMETER_NM = 100.0
DEFAULT_N_TRACKS = 80
DEFAULT_N_FRAMES = 120
DEFAULT_DRIFT_VX_UM_PER_S = 0.5
DEFAULT_DRIFT_VY_UM_PER_S = -0.3
DEFAULT_RNG_SEED = 0
DEFAULT_LAG_FRAME = 2


def _D_um2_per_s_for_diameter(config: AnalysisConfig, diameter_nm: float) -> float:
    # Stokes-Einstein converted to um^2/s.
    d_m = float(diameter_nm) * 1e-9
    D_si = (config.KB * config.temp) / (3.0 * np.pi * config.eta * d_m)
    return float(D_si * 1e12)


def brownian_spots_df(
    config: AnalysisConfig,
    *,
    target_diameter_nm: float = DEFAULT_TARGET_DIAMETER_NM,
    n_tracks: int = DEFAULT_N_TRACKS,
    n_frames: int = DEFAULT_N_FRAMES,
    drift_vx_um_per_s: float = DEFAULT_DRIFT_VX_UM_PER_S,
    drift_vy_um_per_s: float = DEFAULT_DRIFT_VY_UM_PER_S,
    seed: int = DEFAULT_RNG_SEED,
) -> pd.DataFrame:
    """5-column TrackMate-style spots dataframe of 2D Brownian trajectories
    with a uniform linear drift superimposed. Step variance derived from
    Stokes-Einstein for ``target_diameter_nm``. Track origins are drawn
    uniformly from a fixed central field; QUALITY is a constant placeholder.
    """
    D_um2_per_s = _D_um2_per_s_for_diameter(config, target_diameter_nm)
    sigma_step_um = np.sqrt(2.0 * D_um2_per_s * config.dt)
    sigma_step_px = sigma_step_um / config.pixel_size
    drift_px_per_frame_x = drift_vx_um_per_s * config.dt / config.pixel_size
    drift_px_per_frame_y = drift_vy_um_per_s * config.dt / config.pixel_size

    rng = np.random.default_rng(seed)
    rows: list[tuple[int, float, float, int, float]] = []
    frame_idx = np.arange(n_frames)
    for pid in range(1, n_tracks + 1):
        x0 = float(rng.uniform(50.0, 450.0))
        y0 = float(rng.uniform(50.0, 450.0))
        steps_x = rng.normal(0.0, sigma_step_px, size=n_frames)
        steps_y = rng.normal(0.0, sigma_step_px, size=n_frames)
        x = x0 + np.cumsum(steps_x) + drift_px_per_frame_x * frame_idx
        y = y0 + np.cumsum(steps_y) + drift_px_per_frame_y * frame_idx
        for f in range(n_frames):
            rows.append((pid, float(x[f]), float(y[f]), int(f), 100.0))

    return pd.DataFrame(rows, columns=["ID", "X", "Y", "FRAME", "QUALITY"])


def write_spots_csv(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    filename: str = "synthetic_spots.csv",
    header: bool = True,
) -> Path:
    """Persist a spots dataframe to ``out_dir/filename`` and return the path.

    ``header=False`` produces a headerless legacy-TrackMate CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, header=header, index=False)
    return path


def drift_only_spots_df(
    config: AnalysisConfig,
    *,
    vx_um_per_s: float = 2.0,
    vy_um_per_s: float = -1.0,
    n_tracks: int = 20,
    n_frames: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """5-column spots dataframe whose particles move with a uniform drift
    plus small isotropic position noise (no Brownian diffusion). Used by
    drift tests that want a near-noiseless target velocity.
    """
    rng = np.random.default_rng(seed)
    dt = config.dt
    pixel = config.pixel_size
    rows: list[tuple[int, float, float, int, float]] = []
    for pid in range(1, n_tracks + 1):
        x0 = float(rng.uniform(50.0, 150.0))
        y0 = float(rng.uniform(50.0, 150.0))
        for frame in range(n_frames):
            x = x0 + (vx_um_per_s * frame * dt) / pixel + float(rng.normal(0.0, 0.1))
            y = y0 + (vy_um_per_s * frame * dt) / pixel + float(rng.normal(0.0, 0.1))
            rows.append((pid, x, y, frame, 1.0))
    return pd.DataFrame(rows, columns=["ID", "X", "Y", "FRAME", "QUALITY"])


def drifting_displacements_df(
    config: AnalysisConfig,
    out_dir: Path,
    *,
    vx_um_s: float = 2.0,
    vy_um_s: float = -1.0,
    n_tracks: int = 20,
    n_frames: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """A ``drift_only_spots_df`` round-tripped through a headerless CSV and
    :class:`DataPreprocessor`, yielding a displacement dataframe (``X_diff`` /
    ``Y_diff`` columns) with a known uniform drift. Shared by the drift
    analyzer and drift corrector unit tests so the synth→CSV→preprocess
    sequence lives in one place rather than being copied per test module.
    """
    # Imported lazily so this pure-data module isn't coupled to the analysis
    # package at import time (only callers that need preprocessing pull it in).
    from opennta.analysis.data_preprocessor import DataPreprocessor

    raw = drift_only_spots_df(
        config,
        vx_um_per_s=vx_um_s,
        vy_um_per_s=vy_um_s,
        n_tracks=n_tracks,
        n_frames=n_frames,
        seed=seed,
    )
    csv = write_spots_csv(raw, out_dir, filename="drift.csv", header=False)
    df, _ = DataPreprocessor(config).load_and_compute_displacements(str(csv))
    return df


def brownian_tracks_df_corr(
    *,
    D_um2_per_s: float,
    dt: float,
    n_tracks: int = 40,
    n_frames: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Dataframe with ``ID``, ``X_diff_corr``, ``Y_diff_corr``, ``FRAME``
    columns suitable for :class:`MSDCalculator` directly. Each step is iid
    Normal(0, sqrt(2*D*dt)). Position is the cumulative sum of steps.
    """
    rng = np.random.default_rng(seed)
    sigma = float(np.sqrt(2.0 * D_um2_per_s * dt))
    rows: list[tuple[int, float, float, int]] = []
    for pid in range(1, n_tracks + 1):
        x = np.cumsum(rng.normal(0.0, sigma, n_frames))
        y = np.cumsum(rng.normal(0.0, sigma, n_frames))
        for frame in range(n_frames):
            rows.append((pid, float(x[frame]), float(y[frame]), int(frame)))
    return pd.DataFrame(rows, columns=["ID", "X_diff_corr", "Y_diff_corr", "FRAME"])


def quality_samples(
    *,
    n_background: int = 4000,
    n_spike: int = 100,
    bg_mu: float = 50.0,
    bg_sigma: float = 5.0,
    spike_offset_sigma: float = 20.0,
    seed: int = 0,
) -> tuple[np.ndarray, int, float, float, float]:
    """Synthetic TrackMate peak-quality sample: a Gaussian *noise bulk* plus a
    sparse, far-above high-quality *spike* standing in for real particle
    detections.

    This mirrors what the Fiji threshold step actually produces — a unimodal
    cloud of spurious-peak qualities with a handful of genuine particles in the
    upper tail — so the truncated-MLE threshold fitter can be exercised against
    ground truth without invoking Fiji. The bulk is drawn from
    ``Normal(bg_mu, bg_sigma)`` so a Gaussian fit must recover the parent
    ``mu``/``sigma`` (the fitter upper-truncates at the fit-range cutoff, which
    excludes the spike), and the per-peak threshold must land between the bulk
    and the spike.

    Returns ``(values, n_spike, bg_mu, bg_sigma, spike_value)`` with ``values``
    shuffled so no ordering can be relied on downstream.
    """
    rng = np.random.default_rng(seed)
    background = rng.normal(bg_mu, bg_sigma, size=n_background)
    spike_value = float(bg_mu + spike_offset_sigma * bg_sigma)
    spike = np.full(n_spike, spike_value, dtype=float)
    values = np.concatenate([background, spike])
    rng.shuffle(values)
    return values, n_spike, float(bg_mu), float(bg_sigma), spike_value


def noiseless_particle_tiff_stack(
    *,
    n_frames: int = 30,
    height: int = 128,
    width: int = 128,
    n_particles: int = 12,
    psf_sigma_px: float = 1.8,
    step_std_px: float = 0.6,
    peak_intensity: float = 230.0,
    margin_px: int = 14,
    seed: int = 0,
) -> np.ndarray:
    """Noise-free Gaussian-PSF particle stack for end-to-end tracking
    verification.

    Background is exactly zero — no detector reads, no shot/read noise.
    The only intensity comes from the seeded Gaussian PSFs themselves,
    so a tracker that fails to recover ~all particles is broken in a
    way unrelated to imaging artefacts. The integration test asserts on
    the *number* of recovered tracks, not per-frame positions, so the
    ground-truth centres are intentionally not returned.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(margin_px, width - margin_px, size=n_particles)
    ys = rng.uniform(margin_px, height - margin_px, size=n_particles)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    two_sigma_sq = 2.0 * psf_sigma_px ** 2

    stack = np.zeros((n_frames, height, width), dtype=np.uint8)
    for f in range(n_frames):
        frame = np.zeros((height, width), dtype=np.float32)
        for p in range(n_particles):
            frame += peak_intensity * np.exp(
                -((xx - xs[p]) ** 2 + (yy - ys[p]) ** 2) / two_sigma_sq
            )
        stack[f] = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        xs = np.clip(xs + rng.normal(0.0, step_std_px, size=n_particles),
                     margin_px, width - margin_px)
        ys = np.clip(ys + rng.normal(0.0, step_std_px, size=n_particles),
                     margin_px, height - margin_px)

    return stack


__all__ = [
    "DEFAULT_TARGET_DIAMETER_NM",
    "DEFAULT_N_TRACKS",
    "DEFAULT_N_FRAMES",
    "DEFAULT_DRIFT_VX_UM_PER_S",
    "DEFAULT_DRIFT_VY_UM_PER_S",
    "DEFAULT_RNG_SEED",
    "DEFAULT_LAG_FRAME",
    "brownian_spots_df",
    "write_spots_csv",
    "drift_only_spots_df",
    "drifting_displacements_df",
    "brownian_tracks_df_corr",
    "quality_samples",
    "noiseless_particle_tiff_stack",
]
