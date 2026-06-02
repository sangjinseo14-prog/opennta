from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..drift_corrector import DriftCorrector, ExportField, register_corrector
from .field_sampler import FieldSampler

if TYPE_CHECKING:
    from ..types import AnalysisConfig

logger = logging.getLogger(__name__)


def _export_field_from_result(result: dict, fps: float) -> ExportField:
    # Round-trip m/s -> um/frame -> m/s so the bundle-fps scaling used during
    # correction is also baked into the exported field.
    from .unet_bundle.config import ACTUAL_FPS
    from .unet_bundle.coordinate_transforms import inverse_transform_positions_from_reference
    from .unit_convert import m_per_s_to_um_per_frame, resample_to_cell_centers

    tform = result["position_transform"]
    u_ms = m_per_s_to_um_per_frame(result["pred_u_ac"], fps, bundle_fps=ACTUAL_FPS) * fps / 1e6
    v_ms = m_per_s_to_um_per_frame(result["pred_v_ac"], fps, bundle_fps=ACTUAL_FPS) * fps / 1e6
    u_ms, v_ms, xs_ref, ys_ref = resample_to_cell_centers(
        u_ms, v_ms, result["xs_sc"], result["ys_sc"], tform
    )
    xs_px, _ = inverse_transform_positions_from_reference(xs_ref, np.zeros_like(xs_ref), tform)
    _, ys_px = inverse_transform_positions_from_reference(np.zeros_like(ys_ref), ys_ref, tform)
    return ExportField(
        x=np.asarray(xs_px, dtype=float),
        y=np.asarray(ys_px, dtype=float),
        u=np.asarray(u_ms, dtype=float),
        v=np.asarray(v_ms, dtype=float),
    )


@dataclass(frozen=True)
class UNetDiagnosticState:
    sample: Any
    pred_u_sc: np.ndarray
    pred_v_sc: np.ndarray
    vector_n: int


@register_corrector
class UNetCorrector(DriftCorrector):

    mode_key = "unet"
    ui_label = "Field: U-Net"

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self._sampler: FieldSampler | None = None
        self._vector_n: int | None = None
        self._image_size: int | None = None
        self._diag: dict | None = None
        self._export_field: ExportField | None = None
        self.export_field_enabled: bool = False

    def requires_configuration(self) -> bool:
        return True

    def set_params(self, vector_n: int, image_size: int | None = None) -> None:
        self._sampler = None
        self._vector_n = int(vector_n)
        self._image_size = None if image_size is None else int(image_size)

    def fit(self, df: pd.DataFrame, mean_vx: float, mean_vy: float) -> None:
        if self._vector_n is None:
            raise RuntimeError(
                "UNetCorrector requires configuration. "
                "Configure it via the U-Net Field dialog first."
            )
        self._recompute_field_from_df(df)
        self._fitted = True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        df = df.copy()

        # _sampler returns velocity in um/frame (already unit-converted),
        # matching dX_um / dY_um so the subtraction is direct.
        vx_at, vy_at = self._sampler.sample(df["X"], df["Y"])
        df["_dX_corr"] = df["dX_um"] - vx_at
        df["_dY_corr"] = df["dY_um"] - vy_at
        df[["X_diff_corr", "Y_diff_corr"]] = (
            df.groupby("ID")[["_dX_corr", "_dY_corr"]].cumsum()
        )
        df = df.drop(columns=["_dX_corr", "_dY_corr"])
        return df

    def get_parameters(self) -> dict[str, Any]:
        if self._vector_n is None:
            return {}
        return {"Mode": ("U-Net inference", "")}

    def get_export_field(self) -> ExportField | None:
        return self._export_field

    def get_diagnostic_state(self) -> UNetDiagnosticState | None:
        if self._diag is None or self._vector_n is None:
            return None
        return UNetDiagnosticState(
            sample=self._diag["sample"],
            pred_u_sc=self._diag["pred_u_sc"],
            pred_v_sc=self._diag["pred_v_sc"],
            vector_n=int(self._vector_n),
        )

    def _recompute_field_from_df(self, df: pd.DataFrame) -> None:
        from .unet_bundle import get_shared_runner
        from .unet_bundle.config import ACTUAL_FPS
        from .unit_convert import build_field_sampler

        runner = get_shared_runner()
        result = runner.infer_from_df(
            spots_df=df,
            image_width_px=self._image_size,
            image_height_px=self._image_size,
        )

        self._sampler = build_field_sampler(
            result["pred_u_ac"],
            result["pred_v_ac"],
            result["xs_sc"],
            result["ys_sc"],
            result["position_transform"],
            float(self.config.fps),
            ACTUAL_FPS,
        )
        self._diag = {
            "sample": result["sample"],
            "pred_u_sc": result["pred_u_sc"],
            "pred_v_sc": result["pred_v_sc"],
        }
        self._export_field = _export_field_from_result(result, float(self.config.fps))
