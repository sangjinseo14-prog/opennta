from __future__ import annotations

import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._plot_theme import apply_dark_plot_theme

# Result-tab figures (drift, brownian, MSD, size) are created here via
# plt.subplots / plt.figure. Install the dark theme once at module import so
# every figure built afterwards picks up the qdarkstyle-matching palette.
apply_dark_plot_theme()

DEFAULT_R2_THRESHOLD = 0

MSD_PLOT_LAG_MULTIPLE = 5  # show MSD out to 5x the fit lag (display only)

DASH_LW = 1.0
PLOT_BLUE = "#3491dc"
DASH_LIGHT = "#ffffff"


class TabAnalysisOutput:
    # Plot-data builders below are Qt-free so they stay testable without a GUI.

    def sample_brownian_tracks(
        self,
        df: pd.DataFrame,
        n_samples: int = 10,
        random_seed: int | None = None,
    ) -> dict[str, dict[int, dict[str, dict[str, NDArray[np.floating]]]]]:
        if df is None or df.empty:
            return {"sample_ids": [], "tracks": {}}

        if random_seed is not None:
            random.seed(random_seed)

        ids = list(df["ID"].unique())
        if not ids:
            return {"sample_ids": [], "tracks": {}}

        sample_ids = random.sample(ids, min(n_samples, len(ids)))
        tracks: dict[int, dict[str, dict[str, NDArray[np.floating]]]] = {}

        for pid in sample_ids:
            tr = df[df["ID"] == pid]
            tracks[pid] = {
                "original": {
                    "x": tr["X_diff"].values,
                    "y": tr["Y_diff"].values,
                },
                "corrected": {
                    "x": tr.get("X_diff_corr", tr["X_diff"]).values,
                    "y": tr.get("Y_diff_corr", tr["Y_diff"]).values,
                },
            }

        return {"sample_ids": sample_ids, "tracks": tracks}

    def sample_msd_curves(
        self,
        msd_by_id: dict[int, tuple[NDArray[np.floating], NDArray[np.floating]]],
        lag_frame: int,
        n_samples: int = 100,
        fps: float = 25.0,
        dt: float = 0.04,
    ) -> dict[str, Any]:
        if not msd_by_id:
            return {"sample_ids": [], "msd_data": {}, "lag_seconds": 0, "plot_seconds": 0}

        lag_seconds = lag_frame / fps
        sample_ids = random.sample(
            list(msd_by_id.keys()),
            min(n_samples, len(msd_by_id)),
        )

        msd_data: dict[int, dict[str, NDArray[np.floating]]] = {}
        for pid in sample_ids:
            lags, msds = msd_by_id[pid]
            lag_plot = lag_frame * MSD_PLOT_LAG_MULTIPLE
            valid = lags < lag_plot
            if np.any(valid):
                msd_data[pid] = {
                    "time": lags[valid] * dt,
                    "msd": msds[valid],
                }

        return {
            "sample_ids": sample_ids,
            "msd_data": msd_data,
            "lag_seconds": lag_seconds,
            "plot_seconds": MSD_PLOT_LAG_MULTIPLE * lag_seconds,
        }

    # --- Figure creation -------------------------------------------------

    def create_drift_plot(
        self,
        drift_speeds: NDArray[np.floating],
        mean_vx: float,
        mean_vy: float,
        drift_by_id: dict[int, tuple[float, float]] | None = None,
        r2_by_id: dict[int, float] | None = None,
        r2_threshold: float = DEFAULT_R2_THRESHOLD,
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.92)

        good_x, good_y, bad_x, bad_y = [], [], [], []

        if drift_by_id and r2_by_id:
            for pid, (vx, vy) in drift_by_id.items():
                r2 = r2_by_id.get(pid, np.nan)
                if np.isfinite(r2) and r2 >= r2_threshold:
                    good_x.append(vx)
                    good_y.append(vy)
                else:
                    bad_x.append(vx)
                    bad_y.append(vy)

            if good_x:
                ax.scatter(
                    good_x,
                    good_y,
                    alpha=0.5,
                    s=10,
                    color=PLOT_BLUE,
                    label=rf"$R^2 \geq {r2_threshold:.2f}$",
                )
            if bad_x:
                ax.scatter(
                    bad_x,
                    bad_y,
                    alpha=0.5,
                    s=10,
                    color="red",
                    label=rf"$R^2 < {r2_threshold:.2f}$",
                )
        else:
            if len(drift_speeds) > 0:
                ax.scatter(
                    drift_speeds[:, 0],
                    drift_speeds[:, 1],
                    alpha=0.4,
                    label="Samples",
                    s=10,
                    color=PLOT_BLUE,
                )

        if len(drift_speeds) > 0:
            mean_vec = np.mean(drift_speeds, axis=0)
            median_vec = np.median(drift_speeds, axis=0)
            ax.scatter(
                mean_vec[0],
                mean_vec[1],
                color="white",
                label="Mean",
                marker="x",
                s=80,
            )
            ax.scatter(
                median_vec[0],
                median_vec[1],
                color="green",
                label="Median",
                marker="+",
                s=80,
            )

        ax.axhline(0, color="gray", linestyle="--", linewidth=DASH_LW)
        ax.axvline(0, color="gray", linestyle="--", linewidth=DASH_LW)
        ax.set_xlabel(r"$V_x$ ($\mathrm{\mu m/s}$)")
        ax.set_ylabel(r"$V_y$ ($\mathrm{\mu m/s}$)")
        ax.set_title(r"Drift distribution")
        ax.axis("equal")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

        return fig

    def create_brownian_plot(
        self,
        brownian_data: dict,
        r2_by_id: dict | None = None,
        r2_threshold: float = DEFAULT_R2_THRESHOLD,
        gap_frac: float = 0.04,
    ):
        tracks = brownian_data["tracks"]
        if not tracks:
            return None

        sample_ids = brownian_data.get("sample_ids", list(tracks.keys()))

        fig = plt.figure(figsize=(14, 7), constrained_layout=False)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, gap_frac, 1])
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 2])
        fig.subplots_adjust(left=0.1, right=0.90, bottom=0.20, top=0.85, wspace=0)

        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        color_cycle = iter(prop_cycle)

        def _next_color():
            nonlocal color_cycle
            try:
                return next(color_cycle)
            except StopIteration:
                color_cycle = iter(prop_cycle)
                return next(color_cycle)

        for pid in sample_ids:
            if pid not in tracks:
                continue

            track_data = tracks[pid]
            r2 = None
            if r2_by_id is not None:
                r2 = r2_by_id.get(pid, np.nan)

            if r2_by_id is None or (np.isfinite(r2) and r2 >= r2_threshold):
                color = _next_color()
            else:
                color = "red"

            orig = track_data["original"]
            corr = track_data["corrected"]
            axL.plot(orig["x"], orig["y"], lw=1, alpha=0.7, color=color)
            axR.plot(corr["x"], corr["y"], lw=1, alpha=0.7, color=color)

        axL.set_title("Original tracks")
        axL.set_xlabel(r"$\Delta x$ ($\mathrm{\mu m}$)")
        axL.set_ylabel(r"$\Delta y$ ($\mathrm{\mu m}$)")

        axR.set_title("Corrected tracks")
        axR.set_xlabel(r"Corrected $\Delta x$ ($\mathrm{\mu m}$)")
        axR.tick_params(labelleft=False, labelright=True)
        axR.yaxis.set_ticks_position("right")
        axR.yaxis.set_label_position("right")
        axR.set_ylabel(r"Corrected $\Delta y$ ($\mathrm{\mu m}$)")

        for ax in (axL, axR):
            ax.relim()
            ax.autoscale_view()

        xL0, xL1 = axL.get_xlim()
        yL0, yL1 = axL.get_ylim()
        xR0, xR1 = axR.get_xlim()
        yR0, yR1 = axR.get_ylim()
        x_min, x_max = min(xL0, xR0), max(xL1, xR1)
        y_min, y_max = min(yL0, yR0), max(yL1, yR1)

        for ax in (axL, axR):
            ax.update_datalim([(x_min, y_min), (x_max, y_max)])
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.margins(x=0.01, y=0.02)

        return fig

    def create_msd_plot(
        self,
        msd_data: dict,
        r2_by_id: dict | None = None,
        r2_threshold: float = DEFAULT_R2_THRESHOLD,
        gap_frac: float = 0.04,
    ):
        fig = plt.figure(figsize=(14, 7), constrained_layout=False)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, gap_frac, 1])
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 2])
        fig.subplots_adjust(left=0.1, right=0.90, bottom=0.20, top=0.85, wspace=0)

        sample_count = len(msd_data["msd_data"])
        lag_seconds = msd_data["lag_seconds"]
        plot_seconds = msd_data["plot_seconds"]

        for pid, data in msd_data["msd_data"].items():
            r2 = None
            if r2_by_id is not None:
                r2 = r2_by_id.get(pid, np.nan)

            if r2_by_id is None or (np.isfinite(r2) and r2 >= r2_threshold):
                color = PLOT_BLUE
            else:
                color = "red"

            axL.plot(data["time"], data["msd"], lw=0.8, alpha=0.4, color=color)

        axL.set_xlim(0, plot_seconds)
        axL.axvline(
            lag_seconds,
            color="red",
            linestyle="--",
            linewidth=DASH_LW,
            label=rf"fit lag (${lag_seconds:.2f}\,\mathrm{{s}}$)",
        )

        axL.set_xlabel(r"Lag time $\Delta t$ (s)")
        axL.set_ylabel(r"MSD ($\mathrm{\mu m^{2}}$)")
        axL.set_title(f"MSD curves (random {sample_count} tracks)")
        axL.grid(True, linestyle="--", alpha=0.4)
        axL.legend()

        if r2_by_id:
            r2_vals = np.array([v for v in r2_by_id.values() if np.isfinite(v)])
        else:
            r2_vals = np.array([])

        if r2_vals.size > 0:
            axR.hist(r2_vals, bins=30, alpha=0.85, edgecolor="white", color=PLOT_BLUE)
            axR.axvline(
                r2_threshold,
                color="red",
                linestyle="--",
                linewidth=DASH_LW,
                label=rf"$R^2 = {r2_threshold:.2f}$",
            )
            axR.set_xlabel(r"$R^2$")
            axR.set_ylabel("Track count")
            axR.set_title(r"$R^2$ distribution (all tracks)")
            axR.set_xlim(0, 1.05)
            axR.grid(True, linestyle="--", alpha=0.4)
            axR.legend()

            axR.tick_params(labelleft=False, labelright=True)
            axR.yaxis.set_ticks_position("right")
            axR.yaxis.set_label_position("right")
        else:
            axR.text(0.5, 0.5, "No $R^2$ data", ha="center", va="center")
            axR.set_axis_off()

        return fig

    def create_size_plot(
        self,
        hist_data: dict,
        min_d: float = 0.1,
        max_d: float = 1000,
        num_bins: int = 100,
        max_c: float = 500,
        tick_c: float = 50,
        special_ticks: list[float] | None = None,
        log_scale: bool = True,
    ):
        if not hist_data:
            return None

        diameters = hist_data.get("diameters")
        bin_edges = hist_data.get("bin_edges")
        counts = hist_data.get("counts")

        has_samples = diameters is not None and len(diameters) > 0
        has_precomputed = (
            bin_edges is not None and counts is not None and len(counts) > 0
        )
        if not has_samples and not has_precomputed:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.92)

        if log_scale:
            x_min = max(min_d, 0.0001)
            x_max = max(max_d, x_min + 1)
        else:
            x_min = max(min_d, 0.0)
            x_max = max(max_d, x_min + 1)

        if has_samples:
            if log_scale:
                bins = np.logspace(np.log10(x_min), np.log10(x_max), num_bins + 1)
            else:
                bins = np.linspace(x_min, x_max, num_bins + 1)
            ax.hist(diameters, bins=bins, edgecolor="black", color=PLOT_BLUE)
        else:
            edges = np.asarray(bin_edges, dtype=float)
            heights = np.asarray(counts, dtype=float)
            widths = np.diff(edges)
            ax.bar(
                edges[:-1], heights,
                width=widths, align="edge",
                edgecolor="black",
                color=PLOT_BLUE,
            )

        if log_scale:
            ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)

        ax.set_yticks(np.arange(0, max_c + 1, tick_c))
        ax.set_ylim(0, max_c)

        if special_ticks:
            for tick_val in special_ticks:
                if x_min <= tick_val <= x_max:
                    ax.axvline(
                        tick_val,
                        color=DASH_LIGHT,
                        linestyle="--",
                        linewidth=DASH_LW,
                        alpha=0.85,
                    )

        ax.set_xlabel("Estimated particle diameter (nm)")
        ax.set_ylabel("Particle count")
        ax.set_title("Particle size distribution")
        ax.grid(True, linestyle="--", alpha=0.5)

        return fig
