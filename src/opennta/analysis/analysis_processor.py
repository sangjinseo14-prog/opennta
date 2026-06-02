"""Processor for Analysis tab & Batch tab"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..common.progress import (
    LogCallback,
    ProgressCallback,
    ProgressEmitter,
    noop_log_callback,
    noop_progress_callback,
)
from .data_preprocessor import DataPreprocessor
from .diffusion_estimator import DiffusionEstimator
from .drift_analyzer import DriftAnalyzer
from .drift_corrector import DriftCorrector, get_drift_corrector
from .msd_calculator import MSDCalculator
from .status_codes import AnalysisStatus
from .types import AnalysisConfig, AnalysisResults


class AnalysisProcessor:

    def __init__(
        self,
        config: AnalysisConfig,
        correction_mode: str | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
        pre_configured_corrector: DriftCorrector | None = None,
    ):
        self.config = config
        self.results = AnalysisResults()
        self.correction_mode = (correction_mode or "no").lower()

        self.progress_callback: ProgressCallback = progress_callback or noop_progress_callback
        self.log_callback: LogCallback = log_callback or noop_log_callback
        self.emitter = ProgressEmitter(
            progress_callback=self.progress_callback,
            log_callback=self.log_callback,
        )

        self.preprocessor = DataPreprocessor(self.config)
        self.drift_analyzer = DriftAnalyzer(self.config)

        if pre_configured_corrector is not None:
            self.drift_corrector = pre_configured_corrector
        else:
            self.drift_corrector = get_drift_corrector(self.config, self.correction_mode)

        self.msd_calculator = MSDCalculator(self.config)
        self.diffusion_estimator = DiffusionEstimator(self.config)

    def load_and_preprocess(self, csv_path: str) -> pd.DataFrame:
        self.emitter.emit(code=AnalysisStatus.LOADING_DATA)
        df, n_ids = self.preprocessor.load_and_compute_displacements(csv_path)
        self.results.df = df
        self.results.n_ids = n_ids
        self.emitter.emit(code=AnalysisStatus.DATA_LOADED)
        return df

    def calculate_drift(self) -> tuple[float, float, NDArray[np.floating]]:
        self.emitter.emit(code=AnalysisStatus.CALCULATING_DRIFT)
        mean_vx, mean_vy, speeds, drift_by_id = self.drift_analyzer.calculate(self.results.df)

        self.results.mean_vx = mean_vx
        self.results.mean_vy = mean_vy
        self.results.drift_speeds = speeds
        self.results.drift_by_id = drift_by_id
        self.results.mean_drift_velocity = (mean_vx, mean_vy)

        self.emitter.emit(code=AnalysisStatus.DRIFT_CALCULATED)
        return mean_vx, mean_vy, speeds

    def correct_track_by_drift(self) -> None:
        self.emitter.emit(code=AnalysisStatus.CORRECTING_DRIFT)
        self.drift_corrector.fit(
            df=self.results.df,
            mean_vx=self.results.mean_vx,
            mean_vy=self.results.mean_vy,
        )
        self.results.df = self.drift_corrector.apply(self.results.df)
        self.results.correction_mode = self.correction_mode
        self.results.correction_label = self.drift_corrector.get_display_name()
        self.results.correction_params = self.drift_corrector.get_parameters()

        post_vx, post_vy, post_speeds, post_drift_by_id = (
            self.drift_analyzer.calculate(
                self.results.df,
                x_col="X_diff_corr",
                y_col="Y_diff_corr",
            )
        )
        self.results.post_correction_mean_vx = post_vx
        self.results.post_correction_mean_vy = post_vy
        self.results.post_correction_drift_speeds = post_speeds
        self.results.post_correction_drift_by_id = post_drift_by_id

        self.emitter.emit(code=AnalysisStatus.DRIFT_CORRECTED)

    def calculate_msd(self, lag_frame: int) -> dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]]:
        self.emitter.emit(code=AnalysisStatus.CALCULATING_MSD)
        msd_by_id = self.msd_calculator.calculate(self.results.df, lag_frame=lag_frame)
        self.results.msd_by_id = msd_by_id

        extra_points = 2
        min_points = 3
        r2_dict = self.msd_calculator.calculate_r2_per_track(
            msd_by_id,
            lag_frame=lag_frame,
            extra_points=extra_points,
            min_points=min_points,
        )

        self.results.msd_r2_by_id = r2_dict
        self.results.msd_r2_array = np.array(
            [v for v in r2_dict.values() if np.isfinite(v)]
        )
        self.results.lag_frame = int(lag_frame)
        self.results.msd_max_lag = int(5 * lag_frame)
        self.results.msd_extra_points = int(extra_points)
        self.results.msd_min_points = int(min_points)

        self.emitter.emit(code=AnalysisStatus.MSD_CALCULATED)
        return msd_by_id

    def estimate_diffusion_and_diameter(
        self,
        lag_frame: int,
    ) -> tuple[dict[int, float], dict[int, float] | None, list[tuple[int, int]]]:
        self.emitter.emit(code=AnalysisStatus.ESTIMATING_DIFFUSION)
        msd_by_id = self.results.msd_by_id

        D_by_id, diameters_by_id, insufficient, nonpositive_d = (
            self.diffusion_estimator.estimate(msd_by_id, lag_frame=lag_frame)
        )

        self.results.D_by_id = D_by_id
        self.results.diameters_by_id = diameters_by_id
        self.results.last_insufficient_tracks = insufficient
        self.results.last_nonpositive_d_tracks = nonpositive_d

        self.emitter.emit(code=AnalysisStatus.DIFFUSION_ESTIMATED)
        return D_by_id, diameters_by_id, insufficient

    def run_full_analysis(self, csv_path: str, lag_frame: int) -> AnalysisResults:
        try:
            self.emitter.emit(code=AnalysisStatus.STARTED)
            self.load_and_preprocess(csv_path)
            self.calculate_drift()
            self.correct_track_by_drift()
            self.calculate_msd(lag_frame)
            self.estimate_diffusion_and_diameter(lag_frame)
            self.emitter.emit(code=AnalysisStatus.COMPLETE)

            return self.results
        except Exception:
            self.emitter.emit(code=AnalysisStatus.ERROR_GENERAL)
            raise
