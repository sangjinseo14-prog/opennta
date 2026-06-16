from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..drift_corrector import DriftCorrector, ExportField, register_corrector
from .field_sampler import FieldSampler
from .field_smoother import se_weighted_gaussian_smooth
from .node_field import build_node_field
from .types import FieldStats, NumericalFieldParams
from .velocity_field import compute_velocity_field

if TYPE_CHECKING:
    from ..types import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NumericalDiagnosticState:
    stats: FieldStats
    u_sm: np.ndarray | None
    v_sm: np.ndarray | None
    track_df: pd.DataFrame | None
    fps: float


@register_corrector
class NumericalCorrector(DriftCorrector):
    mode_key = "numerical"
    ui_label = "Field: Numerical"

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self._sampler: FieldSampler | None = None
        self._params: NumericalFieldParams | None = None
        self.export_field_enabled: bool = False
        self._last_stats: FieldStats | None = None
        self._last_u_sm = None
        self._last_v_sm = None
        self._last_track_df: pd.DataFrame | None = None

    def set_params(self, params: NumericalFieldParams) -> None:
        self._params = params
        self._sampler = None

    def requires_configuration(self) -> bool:
        return True

    def fit(
        self,
        df: pd.DataFrame,
        mean_vx: float,
        mean_vy: float,
    ) -> None:
        if self._params is None:
            raise RuntimeError(
                "NumericalCorrector requires a velocity field. "
                "Configure it via the Numerical Field dialog first."
            )
        self._recompute_field_from_df(df)
        if {"X", "Y", "ID"}.issubset(df.columns):
            self._last_track_df = df[["X", "Y", "ID"]].copy()
        self._fitted = True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        df = df.copy()

        vx_at, vy_at = self._sampler.sample(df["X"], df["Y"])
        df["_dX_corr"] = df["dX_um"] - vx_at
        df["_dY_corr"] = df["dY_um"] - vy_at
        df[["X_diff_corr", "Y_diff_corr"]] = (
            df.groupby("ID")[["_dX_corr", "_dY_corr"]].cumsum()
        )
        df = df.drop(columns=["_dX_corr", "_dY_corr"])
        return df

    def get_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        p = self._params
        if p is None:
            return params

        params["Window Count"] = (int(p.n_windows), "")
        params["Node Count"] = (int(p.resolved_node_count()), "")
        params["X Range"] = (f"[{p.x_range[0]:.1f}, {p.x_range[1]:.1f}]", "px")
        params["Y Range"] = (f"[{p.y_range[0]:.1f}, {p.y_range[1]:.1f}]", "px")
        params["Outlier k"] = (float(p.outlier_k), "")
        params["Min Count"] = (int(p.min_count), "")
        params["Kernel Size"] = (int(p.ksize), "")
        params["Iterations"] = (int(p.n_iter), "")
        params["Sigma"] = (float(p.sigma), "")
        return params

    def get_export_field(self) -> ExportField | None:
        # The estimated field IS the interpolated node grid (um/frame, N x N).
        # Convert to m/s and report the node pixel coordinates.
        s = self._sampler
        if s is None:
            return None
        fps = float(self.config.fps)
        return ExportField(
            x=np.asarray(s.xs, dtype=float),
            y=np.asarray(s.ys, dtype=float),
            u=np.asarray(s.u, dtype=float) * fps / 1e6,
            v=np.asarray(s.v, dtype=float) * fps / 1e6,
        )

    def get_diagnostic_state(self) -> NumericalDiagnosticState | None:
        if self._last_stats is None:
            return None
        return NumericalDiagnosticState(
            stats=self._last_stats,
            u_sm=self._last_u_sm,
            v_sm=self._last_v_sm,
            track_df=self._last_track_df,
            fps=float(self.config.fps),
        )

    def _recompute_field_from_df(self, df: pd.DataFrame) -> None:
        p = self._params
        if p is None:
            raise RuntimeError(
                "_recompute_field_from_df called without params; "
                "call set_params() first."
            )

        stats = compute_velocity_field(
            df=df,
            n_windows=p.n_windows,
            x_range=p.x_range,
            y_range=p.y_range,
            outlier_k=p.outlier_k,
        )

        u_sm, v_sm = se_weighted_gaussian_smooth(
            mean_dx=stats.mean_dx,
            mean_dy=stats.mean_dy,
            se_vec=stats.se_vec,
            count=stats.count,
            min_count=p.min_count,
            ksize=p.ksize,
            n_iter=p.n_iter,
            sigma=p.sigma,
        )

        node = build_node_field(
            u_sm, v_sm,
            x_range=stats.x_range, y_range=stats.y_range,
            node_count=p.resolved_node_count(),
        )
        self._sampler = FieldSampler(u=node.u, v=node.v, xs=node.xs, ys=node.ys)

        self._last_stats = stats
        self._last_u_sm = u_sm
        self._last_v_sm = v_sm
