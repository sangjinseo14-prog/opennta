"""Shared MLE value types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PhysicalParams:

    temp_K: float
    eta_Pa_s: float
    frame_interval_s: float
    exposure_time_s: float = 0.0
    kB: float = 1.380649e-23
    noise_m2: float = 0.0  # additive per-frame variance from localization error

    def effective_lag_time(self, lag: int = 1) -> float:
        return lag * self.frame_interval_s - self.exposure_time_s / 3.0

    def msd_scale(
        self, d_m: NDArray[np.floating], lag: int = 1
    ) -> NDArray[np.floating]:
        effective_time = self.effective_lag_time(lag)
        if effective_time <= 0:
            raise ValueError("Effective lag time must be positive.")

        # D(d) = kB T / (3 pi eta d). Inputs in m, m^2 out.
        D = (self.kB * self.temp_K) / (3.0 * np.pi * self.eta_Pa_s * d_m)
        return 4.0 * D * effective_time + self.noise_m2


__all__ = ["PhysicalParams"]
