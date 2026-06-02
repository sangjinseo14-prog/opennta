import logging
import os

from PyQt5.QtCore import Qt, QTimer

from ...common.progress import ProgressEmitter
from ...tracking.tracking_method import get_tracking_method
from .adaptor import TrackingWorker, read_tracking_modes_from_ui
from .helper import TabTrackingHelper, detect_fiji_versions
from .ui import TabTrackingUi

logger = logging.getLogger(__name__)

# `dialogs` is an optional sub-module (PyQt dialogs for tracking methods). If
# it is unavailable, the tab still loads — method configuration just fails
# with a clear runtime error so the user can pick a no-config method and
# retry without the app crashing.
try:
    from .dialogs import configure_method
except Exception as _exc:  # pragma: no cover
    logger.warning("configure_method unavailable: %s", _exc)

    def configure_method(*_args, **_kwargs):
        raise RuntimeError(
            "Tracking method configuration dialog is not available. "
            "Pick a method that does not require configuration."
        )


class TabTracking:

    def __init__(self, parent):
        self.parent = parent
        self.current_frame_index = 0
        self.selected_folders = []
        self.tracking_worker = None
        self._updating_tree = False

        self.ui = TabTrackingUi(parent)
        self.helper = TabTrackingHelper(parent)

        self._setup_connections()

        QTimer.singleShot(500, self._check_fiji_path)

    def _check_fiji_path(self):
        fiji_path = None
        try:
            saved_path = self.helper.read_fiji_record()
            if saved_path and os.path.exists(saved_path):
                fiji_path = saved_path
            else:
                guessed_path = self.helper.find_fiji_in_common_places()
                if guessed_path:
                    self.helper.write_fiji_record(guessed_path)
                    logger.info("Fiji auto-detected: %s", guessed_path)
                    fiji_path = guessed_path
        except Exception as e:
            logger.error("Fiji path check error: %s", e)
            return

        if fiji_path:
            versions = detect_fiji_versions(fiji_path)
            logger.info(
                "Fiji integration: path=%s, ImageJ=%s, TrackMate=%s",
                fiji_path,
                versions.get("imagej", "unknown"),
                versions.get("trackmate", "unknown"),
            )
        else:
            logger.warning("Fiji not configured; tracking will be unavailable until set.")

    def _setup_connections(self):
        self.parent.tracking_pushButton_browse.clicked.connect(self._on_browse_clicked)
        self.parent.tracking_pushButton_runTracking.clicked.connect(self._on_run_tracking_clicked)
        self.parent.tracking_pushButton_save.clicked.connect(self._on_save_clicked)
        self.parent.tracking_treeWidget_folders.itemChanged.connect(self._on_tree_item_changed)

        if hasattr(self.parent, "tracking_pushButton_FIJI"):
            self.parent.tracking_pushButton_FIJI.clicked.connect(self._on_fiji_clicked)

    def _on_browse_clicked(self):
        folder_path = self.helper.browse_folder()
        if folder_path:
            self._populate_tree(folder_path)

    def _on_save_clicked(self):
        self.helper.browse_save_folder()

    def _on_fiji_clicked(self):
        self.helper.prompt_user_select_fiji()

    def _on_run_tracking_clicked(self):
        self._run_tracking()

    def _populate_tree(self, root_dir: str):
        self._updating_tree = True
        try:
            self.selected_folders = []
            nta_items_data = self.helper.scan_nta_folders(root_dir)
            self.ui.setup_tree_widget(nta_items_data)
        finally:
            self._updating_tree = False

    def _on_tree_item_changed(self, item, column):
        if column != 0 or self._updating_tree:
            return

        self._updating_tree = True

        try:
            if item.childCount() > 0:
                new_state = item.checkState(0)
                if new_state in (Qt.Checked, Qt.Unchecked):
                    self.ui.update_children_check_state(item, new_state)
            else:
                parent = item.parent()
                if parent:
                    self.ui.update_parent_check_state(parent)
        finally:
            self._updating_tree = False

    def _run_tracking(self):
        selected_folders = self.helper.get_selected_folders()

        if not selected_folders:
            self.ui.show_no_selection_warning()
            return

        fiji_path = self.helper.get_fiji_path()
        if not fiji_path or not os.path.exists(fiji_path):
            self.ui.show_fiji_not_found_error()
            return

        params = read_tracking_modes_from_ui(parent=self.parent)

        if self.tracking_worker and self.tracking_worker.isRunning():
            self.ui.show_already_running_info()
            return

        try:
            preview_emitter = ProgressEmitter()
            method = get_tracking_method(
                mode=params.method,
                params=params,
                emitter=preview_emitter,
                save_path=None,
                fiji_path=fiji_path,
            )
        except Exception as e:
            self.ui.show_invalid_params_error(str(e))
            return

        if method.requires_configuration():
            try:
                config_result = configure_method(
                    method,
                    folder_items=selected_folders,
                    default_index=0,
                    parent=self.parent,
                )
            except Exception as e:
                self.ui.show_invalid_params_error(str(e))
                return
            if config_result is None:
                return
            params = config_result.params

        save_path = self.helper.get_save_path()
        root_folder = self.helper.get_root_folder(selected_folders)

        self.ui.clear_log()

        if hasattr(self, "log_filename"):
            delattr(self, "log_filename")

        self.tracking_worker = TrackingWorker(
            folder_items=selected_folders,
            fiji_path=fiji_path,
            params=params,
            save_path=save_path or None,
            root_folder=root_folder or None,
        )

        self.tracking_worker.log.connect(self._on_tracking_log)
        self.tracking_worker.progress.connect(self._on_tracking_progress)
        self.tracking_worker.finished.connect(self._on_tracking_finished)
        self.tracking_worker.save_path_created.connect(self._on_tracking_save_path_created)

        self.ui.set_processing_ui_enabled(False)
        self.ui.reset_progress_bar()

        self.tracking_worker.start()

    def _on_tracking_log(self, message: str):
        self.ui.append_log(message)

    def _on_tracking_progress(self, progress: int):
        self.ui.update_progress_bar(progress)

    def _on_tracking_save_path_created(self, path: str):
        self.ui.update_save_path(path)

    def _on_tracking_finished(self, success: bool, message: str):
        self.ui.set_processing_ui_enabled(True)
        self.ui.update_progress_bar(100 if success else 0)

        if message:
            self.ui.append_log(message)

        if self.tracking_worker:
            self.tracking_worker.deleteLater()
            self.tracking_worker = None

    def shutdown_active_worker(self, timeout_ms: int = 8000) -> None:
        worker = self.tracking_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(timeout_ms)
