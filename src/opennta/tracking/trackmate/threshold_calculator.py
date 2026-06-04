from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...common.progress import ProgressEmitter
from .fitting import FittingEngine
from .types import ThresholdResult


class ThresholdCalculator:

    def __init__(
        self,
        emitter: ProgressEmitter | None = None,
        model_name: str = "Cheng-Schwartzman",
    ):
        self.emitter = emitter or ProgressEmitter()
        self.model_name = model_name
        self.engine = FittingEngine(model_name)

    def calculate_threshold_from_array(
        self,
        values,
        frac: float = 0.80,
        alpha: float = 0.01,
        quality_csv_path: str | None = None,
    ) -> ThresholdResult:
        """Fit the noise-bulk model to an in-memory quality array and solve the
        threshold. Returns the raw ``ThresholdResult`` (the CSV-facing wrapper
        below flattens it to a dict for backward compatibility)."""
        u = np.asarray(values, dtype=float)
        return self.engine.calculate_threshold(
            u=u,
            alpha=alpha,
            frac=frac,
            quality_csv_path=quality_csv_path,
        )

    def calculate_threshold_from_csv(
        self,
        quality_csv_path: str,
        frac: float = 0.80,
        alpha: float = 0.01,
    ) -> dict[str, Any] | None:
        if not os.path.exists(quality_csv_path):
            self.emitter.emit(msg=f"File not found: {quality_csv_path}")
            return None

        try:
            df = pd.read_csv(quality_csv_path)
        except Exception as e:
            self.emitter.emit(msg=f"Failed to read CSV: {e}")
            return None

        if not any(str(col).lower() == "quality" for col in df.columns):
            try:
                df = pd.read_csv(quality_csv_path, header=None)
                if df.shape[1] == 1:
                    df.columns = ["QUALITY"]
                else:
                    self.emitter.emit(msg="QUALITY column not found")
                    return None
            except Exception as e:
                self.emitter.emit(msg=f"Failed to read CSV without header: {e}")
                return None

        quality_col = next((col for col in df.columns if str(col).lower() == "quality"), None)
        if quality_col is None:
            self.emitter.emit(msg="QUALITY column not found")
            return None

        u = df[quality_col].values.astype(float)
        self.emitter.emit(msg=f"    Loaded {len(u)} quality values")

        try:
            result = self.calculate_threshold_from_array(
                u,
                frac=frac,
                alpha=alpha,
                quality_csv_path=quality_csv_path,
            )

            self.emitter.emit(msg=f"    Converged: {result.converged}")
            if not result.converged:
                self.emitter.emit(msg="Threshold fit did not converge")
                return None
            self.emitter.emit(msg=f"    Peaks above threshold: {result.n_ge_thresh}/{result.n_total}")

            if result.plot_path:
                self.emitter.emit(msg=f"    Plot saved: {Path(result.plot_path).name}")

            return result.to_dict()

        except Exception as e:
            self.emitter.emit(msg=f"Threshold calculation failed: {e}")
            return None

