import matplotlib.pyplot as plt
import numpy as np

from .config import QUIVER_STRIDE_SC

_COLORBAR_KWARGS = dict(fraction=0.046, pad=0.04)

_QUIVER_KWARGS = dict(
    angles="xy",
    scale_units="xy",
    scale=1.0,
    width=0.006,
    headwidth=3.2,
    headlength=4.0,
    headaxislength=3.6,
    pivot="middle",
    zorder=5,
)


def symmetric_percentile_limit(a, pct=99.0, floor=1e-12):
    a = np.asarray(a, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    return float(max(np.percentile(np.abs(a), pct), floor))


def normalized_quiver(ax, u, v, n, *, color, flip_y=False):
    # Arrows are normalized so the longest spans ~90% of a grid cell,
    # independent of the raw velocity units. ``flip_y=True`` matches the
    # dialog's origin='upper' panels (which also sign-flip V); ``flip_y=False``
    # matches origin='lower' panels used by the standalone preview figure.
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if u.ndim != 2 or u.shape != v.shape:
        return None

    if flip_y:
        u = u[::-1, :]
        v = v[::-1, :]

    h, w = u.shape
    n = max(2, min(int(n), min(h, w)))
    xs = np.linspace(0, w - 1, n).round().astype(int)
    ys = np.linspace(0, h - 1, n).round().astype(int)

    X, Y = np.meshgrid(xs, ys)
    U = u[Y, X]
    V = v[Y, X]
    if flip_y:
        V = -V

    mag = np.hypot(U, V)
    finite = np.isfinite(mag)
    if not finite.any():
        return None
    max_mag = float(mag[finite].max())
    if not np.isfinite(max_mag) or max_mag <= 0.0:
        return None

    cell = min(w, h) / max(n - 1, 1)
    scale_factor = (cell * 0.9) / max_mag
    U_plot = np.where(finite, U * scale_factor, 0.0)
    V_plot = np.where(finite, V * scale_factor, 0.0)

    return ax.quiver(X, Y, U_plot, V_plot, color=color, **_QUIVER_KWARGS)


def save_sc_ac_prediction_panel(stem, dual_pred, out_png):
    pred_u_sc = dual_pred["pred_u_sc"]
    pred_v_sc = dual_pred["pred_v_sc"]
    pred_u_ac = dual_pred["pred_u_ac"]
    pred_v_ac = dual_pred["pred_v_ac"]

    def draw_field(ax, arr, title):
        lim = symmetric_percentile_limit(arr, pct=99.0)
        im = ax.imshow(arr, cmap="coolwarm", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, **_COLORBAR_KWARGS)

    def draw_quiver(ax, u, v, title):
        mag = np.sqrt(u ** 2 + v ** 2)
        im = ax.imshow(mag, cmap="viridis", origin="lower", aspect="auto")
        yy, xx = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing="ij")
        step = max(int(QUIVER_STRIDE_SC), 1)
        ax.quiver(
            xx[::step, ::step], yy[::step, ::step],
            u[::step, ::step], v[::step, ::step],
            color="white", angles="xy", scale_units="xy", scale=None, width=0.0022,
        )
        ax.set_title(title)
        plt.colorbar(im, ax=ax, **_COLORBAR_KWARGS)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), constrained_layout=True)
    fig.suptitle(f"Prediction in Simulation vs Actual Coordinates\n{stem}", fontsize=22, weight="bold")

    draw_field(axes[0, 0], pred_u_sc, "pred_u_sc")
    draw_field(axes[0, 1], pred_v_sc, "pred_v_sc")
    draw_quiver(axes[0, 2], pred_u_sc, pred_v_sc, "pred_quiver_sc")

    # pred_u_ac / pred_v_ac share pred_*_sc's physics-y-up layout (row 0 =
    # smallest ys_ref). origin='lower' draws the scene right-side up; only
    # v is sign-flipped in ac (see convert_vector_sc_to_ac).
    draw_field(axes[1, 0], pred_u_ac, "pred_u_ac")
    draw_field(axes[1, 1], pred_v_ac, "pred_v_ac")
    draw_quiver(axes[1, 2], pred_u_ac, pred_v_ac, "pred_quiver_ac")

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def build_ml_preview_figure(stem, sample, pred_u, pred_v, fig=None, vector_n=10, vector_color="#e53935"):
    # Draws into `fig` if given, otherwise creates a new one.
    X = sample["X"]
    pred_speed = np.sqrt(pred_u ** 2 + pred_v ** 2)

    if fig is None:
        fig = plt.Figure(figsize=(11.0, 12.0))
    fig.clear()
    axes = fig.subplots(3, 2)

    lim_u_in = symmetric_percentile_limit(X[..., 0], pct=99.0)
    lim_v_in = symmetric_percentile_limit(X[..., 1], pct=99.0)
    lim_u_out = symmetric_percentile_limit(pred_u, pct=99.0)
    lim_v_out = symmetric_percentile_limit(pred_v, pct=99.0)
    vmax_sp = float(max(np.percentile(pred_speed, 99.0), 1e-12))

    rows = [
        ((X[..., 0],  "Usp_n (input)", "coolwarm", -lim_u_in,  lim_u_in),
         (pred_u,     "U pred (output)", "coolwarm", -lim_u_out, lim_u_out)),
        ((X[..., 1],  "Vsp_n (input)", "coolwarm", -lim_v_in,  lim_v_in),
         (pred_v,     "V pred (output)", "coolwarm", -lim_v_out, lim_v_out)),
        ((X[..., 2],  "mask (input)",  "gray",     0.0,        1.0),
         (pred_speed, "speed pred (output)", "viridis", 0.0,    vmax_sp)),
    ]
    for row_idx, (left, right) in enumerate(rows):
        for col_idx, (arr, title, cmap, vmin, vmax) in enumerate((left, right)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, **_COLORBAR_KWARGS)
            # Overlay velocity vectors on the speed panel since it represents
            # the final inference result.
            if arr is pred_speed:
                normalized_quiver(ax, pred_u, pred_v, n=vector_n, color=vector_color)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    left = min(ax.get_position().x0 for ax in axes.flat)
    right = max(ax.get_position().x1 for ax in axes.flat)
    fig.suptitle(
        "U-Net based velocity-field prediction",
        fontsize=13,
        x=0.5 * (left + right),
        y=0.98,
        ha="center",
    )
    return fig
