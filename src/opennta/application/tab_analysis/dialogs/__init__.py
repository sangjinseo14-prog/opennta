"""Qt configurator dialogs + matching diagnostic PNG writers for drift correctors.

This package owns the application-side dispatchers that decide which dialog or
which diagnostic-PNG writer matches a given corrector. Domain corrector classes
live under `opennta.analysis.<...>` and stay PyQt-free.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget

from opennta.analysis.drift_corrector import ConfigurationResult, DriftCorrector
from opennta.analysis.numerical_field.numerical_corrector import NumericalCorrector
from opennta.analysis.unet_field.unet_corrector import UNetCorrector

from .numerical.diagnostics import save_numerical_diagnostics
from .numerical.dialog import NumericalFieldDialog
from .unet.diagnostics import save_unet_diagnostics
from .unet.dialog import UNetFieldDialog

logger = logging.getLogger(__name__)

__all__ = [
    "NumericalFieldDialog",
    "UNetFieldDialog",
    "configure_corrector",
    "export_field_csv",
    "save_diagnostics",
]


def configure_corrector(
    corrector: DriftCorrector,
    *,
    df: Any,
    config: Any,
    parent: QWidget | None,
) -> ConfigurationResult | None:
    if isinstance(corrector, NumericalCorrector):
        dialog = NumericalFieldDialog(df=df, config=config, parent=parent)
        if dialog.exec_() != dialog.Accepted:
            return None

        params = dialog.get_params()
        if params is None:
            return None
        export_csv = dialog.get_export_csv()

        def apply_to(c: DriftCorrector) -> None:
            nc = _ensure(c, NumericalCorrector)
            nc.set_params(params)
            nc.export_field_enabled = export_csv

        return ConfigurationResult(apply_to=apply_to)

    if isinstance(corrector, UNetCorrector):
        dialog = UNetFieldDialog(df=df, config=config, parent=parent)
        if dialog.exec_() != dialog.Accepted:
            return None

        vector_n = dialog.get_vector_n()
        image_size = dialog.get_image_size()
        export_csv = dialog.get_export_csv()

        def apply_to(c: DriftCorrector) -> None:
            uc = _ensure(c, UNetCorrector)
            uc.set_params(vector_n, image_size)
            uc.export_field_enabled = export_csv

        return ConfigurationResult(apply_to=apply_to)

    raise TypeError(f"Unsupported corrector: {type(corrector).__name__}")


def save_diagnostics(
    corrector: DriftCorrector,
    out_dir: str,
    basename: str,
) -> None:
    if isinstance(corrector, NumericalCorrector):
        state = corrector.get_diagnostic_state()
        if state is None:
            return
        save_numerical_diagnostics(state, out_dir, basename)
        return

    if isinstance(corrector, UNetCorrector):
        state = corrector.get_diagnostic_state()
        if state is None:
            return
        save_unet_diagnostics(state, out_dir, basename)
        return
    # MeanCorrector, NoDriftCorrector etc. have no diagnostic PNG — silent no-op.

def export_field_csv(corrector: DriftCorrector, out_csv_path: str) -> None:
    # x/y are truncated to 3 dp because the U-Net path round-trips through
    # float32 transforms, surfacing integer-clean cell centers (e.g. 27, 45)
    # as 27.000002 etc.; the sampler still uses the full-precision field.
    field = corrector.get_export_field()
    if field is None:
        logger.warning(
            "export_field_csv: corrector %s has no export field; skipping %s",
            type(corrector).__name__,
            out_csv_path,
        )
        return

    xx, yy = np.meshgrid(field.x, field.y)  # row = j (y), col = i (x)
    u = np.asarray(field.u, dtype=float)
    v = np.asarray(field.v, dtype=float)

    df = pd.DataFrame({
        "x(pixel)": [f"{val:.3f}" for val in xx.reshape(-1)],
        "y(pixel)": [f"{val:.3f}" for val in yy.reshape(-1)],
        "u": [f"{val:.9f}" for val in u.reshape(-1)],
        "v": [f"{val:.9f}" for val in v.reshape(-1)],
    })
    df.to_csv(out_csv_path, index=False)


def _ensure(corrector: DriftCorrector, expected: type) -> Any:
    if not isinstance(corrector, expected):
        raise TypeError(
            f"apply_to expected {expected.__name__}, got {type(corrector).__name__}"
        )
    return corrector
