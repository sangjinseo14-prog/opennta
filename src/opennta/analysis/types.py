from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class AnalysisConfig:
    sensor_size: float = field(default=6.5, metadata={"user_config": True})
    magnification: float = field(default=20, metadata={"user_config": True})
    fps: float = field(default=30.0, metadata={"user_config": True})
    temp: float = field(default=295.15, metadata={"user_config": True})
    eta: float = field(default=0.0009544, metadata={"user_config": True})
    KB: float = 1.380649e-23
    pixel_size: float = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self) -> None:
        self.pixel_size = self.sensor_size / self.magnification
        self.dt = 1.0 / self.fps


def user_config_fields(cls) -> tuple[str, ...]:
    return tuple(f.name for f in fields(cls) if f.metadata.get("user_config"))

@dataclass
class AnalysisResults:
    df: pd.DataFrame | None = None
    n_ids: int = 0

    mean_vx: float = 0.0
    mean_vy: float = 0.0
    drift_speeds: NDArray[np.floating] | None = None
    drift_by_id: dict[int, tuple[float, float]] | None = None

    post_correction_mean_vx: float = 0.0
    post_correction_mean_vy: float = 0.0
    post_correction_drift_speeds: NDArray[np.floating] | None = None
    post_correction_drift_by_id: dict[int, tuple[float, float]] | None = None

    msd_by_id: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]] | None = None
    D_by_id: dict[int, float] | None = None
    msd_r2_by_id: dict[int, float] | None = None
    msd_r2_array: NDArray[np.floating] | None = None

    diameters_by_id: dict[int, float] | None = None
    log_std_mean: float = 0.0
    log_std_median: float = 0.0

    mean_drift_velocity: tuple[float, float] = (0.0, 0.0)
    mean_diffusion_coeff: float = 0.0
    size_mean: float = 0.0
    size_median: float = 0.0

    correction_mode: str = "no"
    correction_label: str = "No Correction"
    correction_params: dict = field(default_factory=dict)

    lag_frame: int = 0
    msd_max_lag: int = 0
    msd_extra_points: int = 2
    msd_min_points: int = 3
    r2_threshold: float = 0.0

    last_insufficient_tracks: list[tuple[int, int]] | None = None
    last_nonpositive_d_tracks: list[tuple[int, float]] | None = None
