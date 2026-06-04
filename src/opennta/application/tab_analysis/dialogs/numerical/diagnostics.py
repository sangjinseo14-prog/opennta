from __future__ import annotations

import os

import numpy as np

from opennta.analysis.numerical_field.numerical_corrector import (
    NumericalDiagnosticState,
)

from . import _plot_style as _ps
from ._plot_style import apply_dark_plot_theme, recolor_for_export, style_figure


def save_numerical_diagnostics(
    state: NumericalDiagnosticState,
    out_dir: str,
    basename: str,
) -> None:
    # Titles/colors/overlay style come from _plot_style so the saved diagnostic
    # stays in sync with the configure dialog.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    apply_dark_plot_theme()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}_drift_field.png")

    stats = state.stats
    u_sm = state.u_sm
    v_sm = state.v_sm
    track_df = state.track_df
    fps = state.fps

    # Stored field is in um/frame; display in um/s to match the dialog.
    vbar = np.hypot(stats.mean_dx, stats.mean_dy) * fps
    ci95_vec = stats.ci95_vec * fps
    if u_sm is not None and v_sm is not None:
        vbar_sm = np.hypot(u_sm, v_sm) * fps
    else:
        vbar_sm = None

    fig = Figure(figsize=(11, 9))
    FigureCanvasAgg(fig)
    style_figure(fig)

    axes = fig.subplots(2, 2)
    flat_axes = axes.flatten()
    for ax in flat_axes:
        ax.set_aspect("equal", adjustable="box", anchor="C")

    specs = _ps.PLOT_SPECS
    ax_tracks = flat_axes[0]
    ax_mean = flat_axes[1]
    ax_ci = flat_axes[2]
    ax_sm = flat_axes[3]

    _ps.style_panel_axes(ax_tracks, specs[_ps.TRACKS_INDEX][0])
    sm = _ps.draw_tracks_on_axes(
        ax_tracks, track_df, stats.n_windows, stats.x_range, stats.y_range,
    )
    if sm is not None:
        cb = fig.colorbar(sm, ax=ax_tracks, fraction=0.046, pad=0.04)
        _ps.style_colorbar(cb, _ps.TRACK_TIME_LABEL)

    _draw_field_panel(
        fig, ax_mean,
        data=vbar,
        cmap=specs[_ps.MEAN_INDEX][1],
        title=specs[_ps.MEAN_INDEX][0],
        unit=specs[_ps.MEAN_INDEX][2],
    )
    _ps.draw_quiver_overlay(ax_mean, stats.mean_dx, stats.mean_dy)

    _draw_field_panel(
        fig, ax_ci,
        data=ci95_vec,
        cmap=specs[_ps.CI95_INDEX][1],
        title=specs[_ps.CI95_INDEX][0],
        unit=specs[_ps.CI95_INDEX][2],
    )

    if vbar_sm is not None:
        _draw_field_panel(
            fig, ax_sm,
            data=vbar_sm,
            cmap=specs[_ps.SMOOTHED_INDEX][1],
            title=specs[_ps.SMOOTHED_INDEX][0],
            unit=specs[_ps.SMOOTHED_INDEX][2],
        )
        _ps.draw_quiver_overlay(ax_sm, u_sm, v_sm)
    else:
        _draw_field_panel(
            fig, ax_sm,
            data=vbar,
            cmap=specs[_ps.SMOOTHED_INDEX][1],
            title=f"{specs[_ps.SMOOTHED_INDEX][0]} (smoothing disabled)",
            unit=specs[_ps.SMOOTHED_INDEX][2],
        )
        _ps.draw_quiver_overlay(ax_sm, stats.mean_dx, stats.mean_dy)

    fig.tight_layout()
    recolor_for_export(fig)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)


def _draw_field_panel(fig, ax, data, cmap: str, title: str, unit: str) -> None:
    _ps.style_panel_axes(ax, title)
    im = ax.imshow(data, cmap=cmap, origin="upper", interpolation="nearest")
    ny, nx = data.shape[:2]
    ax.set_xticks(np.arange(0, nx, max(1, nx // 10)))
    ax.set_yticks(np.arange(0, ny, max(1, ny // 10)))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _ps.style_colorbar(cb, unit)
