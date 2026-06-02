import logging

from PyQt5.QtCore import Qt

from ...analysis.data_preprocessor import DataPreprocessor
from ...analysis.drift_corrector import get_drift_corrector
from .adaptor import BatchAnalysisWorker
from .helper import TabBatchAnalysisHelper
from .ui import SpotFileTableModel, TabBatchAnalysisUi

logger = logging.getLogger(__name__)

# Batch tab reuses tab_analysis' dialog dispatcher. If that sub-module is
# unavailable, batch still loads — configuration just fails with a clear
# runtime error so the user can pick a no-config corrector and retry.
try:
    from ..tab_analysis.dialogs import configure_corrector
except Exception as _exc:  # pragma: no cover
    logger.warning("configure_corrector unavailable: %s", _exc)

    def configure_corrector(*_args, **_kwargs):
        raise RuntimeError(
            "Corrector configuration dialog is not available. "
            "Pick a corrector that does not require configuration."
        )


class TabBatchAnalysis:

    def __init__(self, parent):
        self.parent = parent
        self.table_model = SpotFileTableModel()
        self.config_manager = parent.config_manager
        self.ui = TabBatchAnalysisUi(parent, self.table_model)
        self.helper = TabBatchAnalysisHelper(parent)

        self.worker = None
        self._updating_tree = False

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        self.ui.setup_table_view()

    def _setup_connections(self):
        self.parent.batch_pushButton_browse.clicked.connect(self._on_browse_clicked)
        self.parent.batch_pushButton_runAnalysis.clicked.connect(self._on_analysis_clicked)
        self.parent.batch_treeWidget_folders.itemChanged.connect(self._on_tree_item_changed)


    def _on_browse_clicked(self):
        folder = self.helper.browse_result_folder()
        if folder:
            self._populate_tree(folder)

    def _on_analysis_clicked(self):
        self._run_batch_analysis()

    def _on_tree_item_changed(self, item, column):
        if column != 0 or self._updating_tree:
            return

        self._updating_tree = True
        try:
            if item.childCount() > 0:
                new_state = item.checkState(0)
                if new_state in (Qt.Checked, Qt.Unchecked):
                    for i in range(item.childCount()):
                        child = item.child(i)
                        if child.checkState(0) != new_state:
                            child.setCheckState(0, new_state)
                        self._sync_table_row_for_item(child, new_state)
            else:
                self._sync_table_row_for_item(item, item.checkState(0))
                parent = item.parent()
                if parent is not None:
                    self._recompute_parent_check_state(parent)
        finally:
            self._updating_tree = False

    def _sync_table_row_for_item(self, file_item, check_state):
        file_path = file_item.data(0, Qt.UserRole)
        if not file_path:
            return
        row = self.helper.find_row_index_by_path(
            self.table_model.spot_files, file_path,
        )
        if row >= 0:
            self.ui.enable_table_row_widgets(row, check_state == Qt.Checked)

    def _recompute_parent_check_state(self, parent_item):
        total = parent_item.childCount()
        checked = sum(
            1 for i in range(total)
            if parent_item.child(i).checkState(0) == Qt.Checked
        )
        if checked == 0:
            parent_item.setCheckState(0, Qt.Unchecked)
        elif checked == total:
            parent_item.setCheckState(0, Qt.Checked)
        else:
            parent_item.setCheckState(0, Qt.PartiallyChecked)

    def _populate_tree(self, folder: str | None):
        self.table_model.clear_files()

        nta_folders_with_files, spot_files = self.helper.discover_nta_spot_files(folder)

        if not spot_files:
            self.ui.show_no_files_warning()
            return

        self.table_model.spot_files = spot_files

        self.ui.populate_tree(nta_folders_with_files, self.table_model.spot_files)
        self.table_model.layoutChanged.emit()

        self.ui.update_table_widgets(
            lambda: self.helper.get_available_group_numbers(self.table_model.spot_files),
            self._on_group_changed
        )

    def _on_group_changed(self, row: int, text: str):
        try:
            group_num = int(text)
            self.helper.update_file_group(self.table_model.spot_files, row, group_num)

            group_options = self.helper.get_available_group_numbers(self.table_model.spot_files)
            self.ui.update_group_comboboxes(group_options)
        except ValueError:
            pass

    def _run_batch_analysis(self):
        index_checked_files = self.helper.get_checked_files(self.table_model.spot_files)

        if not index_checked_files:
            self.ui.show_no_files_warning()
            return

        if self.worker and self.worker.isRunning():
            self.ui.show_error("Batch analysis is already running!")
            return

        self.ui.set_processing_ui_enabled(False)
        self.ui.enable_all_table_widgets(False)
        started = False

        try:
            config = self.config_manager.get_config()

            lag_frame = self.ui.read_lag_frame_and_refresh_seconds(
                self.parent.batch_lineEdit_lagFrame,
                self.parent.batch_lineEdit_lagSec,
                config.fps,
            )

            base_path = self.helper.get_result_folder()
            if not base_path:
                raise RuntimeError("Please choose an output base folder first.")

            mode = self.ui.get_correction_mode()
            r2_threshold = float(self.parent.batch_lineEdit_r2Threshold.text())
            export_corrected = self.parent.batch_checkBox_csv.isChecked()

            pre_configured_corrector = None
            corrector = get_drift_corrector(config, mode)
            if corrector.requires_configuration():
                first_file_info = index_checked_files[0][1]
                first_csv_path = first_file_info['path']
                try:
                    df_sample, _ = DataPreprocessor(config).load_and_compute_displacements(first_csv_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to preprocess first file for configuration:\n{e}"
                    ) from e

                result = configure_corrector(
                    corrector,
                    df=df_sample,
                    config=config,
                    parent=self.parent,
                )
                if result is None:
                    self.ui.set_processing_ui_enabled(True)
                    self.ui.enable_all_table_widgets(True)
                    return
                result.apply_to(corrector)
                pre_configured_corrector = corrector

            self.ui.reset_all_progress_bars()

            self.worker = BatchAnalysisWorker(
                index_checked_files,
                config,
                lag_frame,
                base_path,
                mode,
                r2_threshold,
                pre_configured_corrector=pre_configured_corrector,
                export_corrected=export_corrected,
            )

            self.worker.progress.connect(self._on_progress)
            self.worker.file_completed.connect(self._on_file_completed)
            self.worker.group_completed.connect(self._on_group_completed)
            self.worker.batch_completed.connect(self._on_batch_completed)

            self.worker.start()
            started = True

        except Exception as e:
            error_message = str(e)
            self._on_batch_completed(False, error_message)
            self.ui.show_error(error_message)
        finally:
            if not started:
                self.ui.set_processing_ui_enabled(True)
                self.ui.enable_all_table_widgets(True)

    def _on_progress(self, file_idx: int, file_progress: int, overall_progress: int):
        self.ui.update_file_progress(file_idx, file_progress)
        self.ui.update_overall_progress(overall_progress)

    def _on_file_completed(self, file_idx: int):
        self.ui.update_file_progress(file_idx, 100)

    def _on_group_completed(self, group_num: int, message: str):
        # Worker already logs group summaries via log_message; logging here
        # would duplicate the same line.
        return

    def _on_batch_completed(self, success: bool, message: str):
        if success:
            logger.info("Batch analysis: %s", message)
        else:
            logger.error("Batch analysis failed: %s", message)

        self.ui.set_processing_ui_enabled(True)
        self.ui.enable_all_table_widgets(True)
        self.ui.update_overall_progress(100)
        self.ui.show_completion_message(success, message)

        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def shutdown_active_worker(self, timeout_ms: int = 8000) -> None:
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(timeout_ms)
