from __future__ import annotations

import os

from opennta.analysis.unet_field.unet_corrector import UNetDiagnosticState

from . import _plot_style as _ps
from ._plot_style import apply_dark_plot_theme


def save_unet_diagnostics(
    state: UNetDiagnosticState,
    out_dir: str,
    basename: str,
) -> None:
    # Use the OO Agg canvas (not pyplot / matplotlib.use) so saving from a
    # worker thread never switches the process-global backend or touches
    # pyplot's global figure registry.
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from opennta.analysis.unet_field.unet_bundle.visualization import (
        build_ml_preview_figure,
    )

    apply_dark_plot_theme()
    os.makedirs(out_dir, exist_ok=True)
    fig = build_ml_preview_figure(
        stem=basename,
        sample=state.sample,
        pred_u=state.pred_u_sc,
        pred_v=state.pred_v_sc,
        vector_n=state.vector_n,
        vector_color=_ps.VECTOR_COLOR,
    )
    FigureCanvasAgg(fig)
    out_path = os.path.join(out_dir, f"{basename}_ml_field.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
