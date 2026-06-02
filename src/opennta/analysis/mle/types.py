"""Shared MLE value types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PhysicalParams:

    temp_K: float
    eta_Pa_s: float
    tau_s: float
    kB: float = 1.380649e-23
    noise_m2: float = 0.0  # additive per-frame variance from localization error

    def msd_scale(self, d_m: NDArray[np.floating]) -> NDArray[np.floating]:
        # s(d) = 4 D(d) tau + noise_m2, D(d) = kB T / (3 pi eta d). Inputs in m, m^2 out.
        D = (self.kB * self.temp_K) / (3.0 * np.pi * self.eta_Pa_s * d_m)
        return 4.0 * D * self.tau_s + self.noise_m2


__all__ = ["PhysicalParams"]
