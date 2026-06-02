import logging
import os

import matplotlib.pyplot as plt

from ...analysis.data_preprocessor import DataPreprocessor
from ...analysis.drift_corrector import get_drift_corrector
from ...analysis.result_filters import filter_by_r2
from ...analysis.size_distributor import get_size_distributor
from ..common.statistics import StatisticsUtils
from ..config import ConfigManager
from .adaptor import AnalysisWorker
from .helper import TabAnalysisHelper
from .output import TabAnalysisOutput
from .ui import TabAnalysisUi

logger = logging.getLogger(__name__)

# `dialogs` is an optional sub-module (PyQt dialogs + diagnostic writers). If
# it is unavailable, the tab still loads — corrector configuration just fails
# with a clear runtime error so the user can pick a no-config corrector and
# retry without the app crashing.
try:
    from .dialogs import configure_corrector, export_field_csv
except Exception as _exc:  # pragma: no cover
    logger.warning("configure_corrector unavailable: %s", _exc)

    def configure_corrector(*_args, **_kwargs):
        raise RuntimeError(
            "Corrector configuration dialog is not available. "
            "Pick a corrector that does not require configuration."
        )

    def export_field_csv(*_args, **_kwargs):
        pass


class TabAnalysis:

    def __init__(self, parent):
        self.parent = parent
        self.worker = None
        self.results = None
        self.stats_dict = None
        self.lag_frame = None
        self.config_manager = parent.config_manager

        self.adaptor = None
        self.output = TabAnalysisOutput()

        self.ui = TabAnalysisUi(parent)
        desktop = ConfigManager.desktop_path()
        self.helper = TabAnalysisHelper(parent, desktop, desktop)

        self._setup_connections()
        self.r2_threshold = None

    def _setup_connections(self):
        self.parent.analysis_pushButton_browse.clicked.connect(self._on_browse_clicked)
        self.parent.analysis_pushButton_runAnalysis.clicked.connect(self._on_analysis_clicked)
        self.parent.analysis_pushButton_replot.clicked.connect(self._on_replot_clicked)
        self.parent.analysis_pushButton_exportSize.clicked.connect(self._on_export_size_clicked)
        self.parent.analysis_pushButton_exportReport.clicked.connect(self._on_export_report_clicked)

    def _on_browse_clicked(self):
        self.helper.select_csv_file()
        self.ui.reset_progress_bar()

    def _on_analysis_clicked(self):
        self._run_analysis()

    def _on_replot_clicked(self):
        self._run_replot()

    def _on_export_size_clicked(self):
        self.helper.export_diameter_csv(self.results)

    def _on_export_report_clicked(self):
        self.helper.export_report_html(self.stats_dict)

    def _run_analysis(self):
        csv_path = self.helper.get_csv_path()
        if not csv_path:
            self.ui.show_warning("Please select a CSV file first.")
            return

        if self.worker and self.worker.isRunning():
            self.ui.show_warning("Analysis is already running.")
            return

        self.ui.set_processing_ui_enabled(False)

        try:
            analysis_config = self.config_manager.get_config()
            mode = self.ui.get_correction_mode()

            self.lag_frame = self.ui.read_lag_frame_and_refresh_seconds(
                self.parent.analysis_lineEdit_lagFrame,
                self.parent.analysis_lineEdit_lagSec,
                analysis_config.fps,
            )
            self.r2_threshold = float(self.parent.analysis_lineEdit_r2Threshold.text())

            corrector = get_drift_corrector(analysis_config, mode)

            if corrector.requires_configuration():
                df, _ = DataPreprocessor(analysis_config).load_and_compute_displacements(csv_path)
                result = configure_corrector(
                    corrector,
                    df=df,
                    config=analysis_config,
                    parent=self.parent,
                )
                if result is None:
                    self.ui.set_processing_ui_enabled(True)
                    return
                result.apply_to(corrector)

            self.worker = AnalysisWorker(
                csv_path=csv_path,
                config=analysis_config,
                lag_frame=self.lag_frame,
                correction_mode=mode,
                pre_configured_corrector=corrector,
            )
            self.worker.finished.connect(self._on_analysis_finished)
            self.worker.progress.connect(self._on_analysis_progress)
            self.worker.log.connect(self._on_analysis_log)
            self.worker.start()

        except Exception as e:
            self.ui.update_progress_bar(0, "Error")
            self.ui.show_error(str(e))
            self.ui.set_processing_ui_enabled(True)

    def _on_analysis_progress(self, progress: int, msg: str | None = None):
        self.ui.update_progress_bar(progress, msg)

    def _on_analysis_log(self, message: str):
        logger.info(message)

    def shutdown_active_worker(self, timeout_ms: int = 8000) -> None:
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(timeout_ms)

    def _on_analysis_finished(self, success: bool, message: str):
        try:
            if not success:
                self.ui.show_error(message)
                return

            self.results = self.worker.results

            valid_diameters_by_id = filter_by_r2(
                self.results.diameters_by_id,
                self.results.msd_r2_by_id,
                self.r2_threshold,
            )
            valid_diffusion_by_id = filter_by_r2(
                self.results.D_by_id,
                self.results.msd_r2_by_id,
                self.r2_threshold,
            )
            self.results.diameters_by_id = valid_diameters_by_id
            self.results.D_by_id = valid_diffusion_by_id
            self.results.r2_threshold = float(self.r2_threshold)

            self.stats_dict = StatisticsUtils.from_results(self.results)

            self.ui._force_render_all_tabs(self.parent.analysis_tabWidget_results)
            self._create_and_display_plots()
            self.ui.update_results_display(stats_dict=self.stats_dict, file_path=self.helper.get_csv_path())

            if self.parent.analysis_checkBox_csv.isChecked():
                csv_path = self.helper.get_csv_path()
                if csv_path and self.results.df is not None:
                    pixel_size = self.config_manager.get_config().pixel_size
                    self.helper.export_corrected_track(self.results.df, csv_path, pixel_size)

            corrector = self.worker.pre_configured_corrector if self.worker else None
            if corrector is not None and getattr(corrector, "export_field_enabled", False):
                csv_path = self.helper.get_csv_path()
                if csv_path:
                    spot_name = os.path.splitext(os.path.basename(csv_path))[0]
                    mode_key = getattr(corrector, "mode_key", "") or "field"
                    field_name = f"{spot_name}_{mode_key}_uv field.csv"
                    field_path = os.path.join(ConfigManager.desktop_path(), field_name)
                    try:
                        export_field_csv(corrector, field_path)
                    except Exception:
                        logger.exception("Field CSV export failed: %s", field_path)

            if hasattr(self.results, 'last_insufficient_tracks'):
                insufficient = self.results.last_insufficient_tracks
                if insufficient:
                    self.ui.show_insufficient_tracks_warning(insufficient)

        except Exception as e:
            self.ui.show_error(str(e))
        finally:
            self.ui.set_processing_ui_enabled(True)
            if self.worker:
                self.worker.deleteLater()
                self.worker = None

    def _create_and_display_plots(self):
        results = self.results
        r2_by_id = self.results.msd_r2_by_id

        if results.drift_speeds is not None:
            fig_drift = self.output.create_drift_plot(
                drift_speeds=results.drift_speeds,
                mean_vx=results.mean_vx,
                mean_vy=results.mean_vy,
                drift_by_id=results.drift_by_id,
                r2_by_id=r2_by_id,
                r2_threshold=self.r2_threshold,
            )
            try:
                self.ui.display_matplotlib_figure(self.parent.analysis_graphicsView_drift, fig_drift)
            finally:
                plt.close(fig_drift)

        brownian_data = self.output.sample_brownian_tracks(results.df, n_samples=10)
        if brownian_data["tracks"]:
            fig_brownian = self.output.create_brownian_plot(
                brownian_data, r2_by_id=r2_by_id, r2_threshold=self.r2_threshold,
            )
            try:
                self.ui.display_matplotlib_figure(self.parent.analysis_graphicsView_brownian, fig_brownian)
            finally:
                plt.close(fig_brownian)

        config = self.config_manager.get_config()
        msd_data = self.output.sample_msd_curves(
            results.msd_by_id, lag_frame=self.lag_frame, n_samples=100,
            fps=config.fps, dt=config.dt,
        )
        if msd_data["msd_data"]:
            fig_msd = self.output.create_msd_plot(
                msd_data, r2_by_id=r2_by_id, r2_threshold=self.r2_threshold,
            )
            try:
                self.ui.display_matplotlib_figure(self.parent.analysis_graphicsView_MSD, fig_msd)
            finally:
                plt.close(fig_msd)

        self._plot_size_histogram()

    def _plot_size_histogram(self):
        if not self.results or self.results.diameters_by_id is None:
            return

        results = self.results
        min_d = float(self.parent.analysis_lineEdit_minD.text())
        max_d = float(self.parent.analysis_lineEdit_maxD.text())
        num_bins = int(self.parent.analysis_lineEdit_binNum.text())
        max_c = float(self.parent.analysis_lineEdit_maxC.text())
        tick_c = float(self.parent.analysis_lineEdit_tickC.text())
        special_ticks = self.ui.parse_custom_axis_ticks()
        log_scale = self.parent.analysis_checkBox_logarithmPlot.isChecked()

        distributor = get_size_distributor(self.ui.get_distribution_mode())
        hist_data = distributor.compute(
            results.diameters_by_id,
            min_d=min_d, max_d=max_d, num_bins=num_bins,
            msd_by_id=results.msd_by_id,
            df=results.df,
            config=self.config_manager.get_config(),
            log_scale=log_scale,
        )

        if hist_data.get("diameters") is not None or hist_data.get("counts") is not None:
            fig_size = self.output.create_size_plot(
                hist_data, min_d=min_d, max_d=max_d, num_bins=num_bins,
                max_c=max_c, tick_c=tick_c, special_ticks=special_ticks,
                log_scale=log_scale,
            )
            if fig_size:
                try:
                    self.ui.display_matplotlib_figure(self.parent.analysis_graphicsView_size, fig_size)
                finally:
                    plt.close(fig_size)

        self._update_diameter_peaks_from_histogram(hist_data)

    def _update_diameter_peaks_from_histogram(self, hist_data: dict):
        if not self.stats_dict or "diameter" not in self.stats_dict:
            return

        bin_centers = hist_data.get("bin_centers")
        bin_edges = hist_data.get("bin_edges")
        counts = hist_data.get("counts")
        if bin_centers is None or bin_edges is None or counts is None:
            return

        peaks = StatisticsUtils.detect_peaks_from_histogram(
            bin_centers=bin_centers,
            bin_edges=bin_edges,
            counts=counts,
        )
        self.stats_dict["diameter"]["peaks"] = peaks
        self.ui.update_results_display(stats_dict=self.stats_dict, file_path=self.helper.get_csv_path())

    def _run_replot(self):
        if not self.results or self.results.diameters_by_id is None:
            self.ui.show_warning("No analysis data available. Run analysis first.")
            return
        try:
            self._plot_size_histogram()
        except Exception as e:
            self.ui.show_error(str(e))
