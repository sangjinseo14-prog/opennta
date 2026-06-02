import os
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QTreeWidgetItem

from .adaptor import (
    get_detector_ui_labels,
    get_linker_ui_labels,
    get_tracking_ui_labels,
)


class TabTrackingUi:

    def __init__(self, parent):
        self.parent = parent
        self._populate_tracking_method_comboboxes()
        self.parent.tracking_radioButton_combined.toggled.connect(self._on_track_mode_toggled)
        self._on_track_mode_toggled(self.parent.tracking_radioButton_combined.isChecked())

    def _populate_tracking_method_comboboxes(self) -> None:
        for combo_name, labels in (
            ("tracking_comboBox_method", get_tracking_ui_labels()),
            ("tracking_comboBox_detector", get_detector_ui_labels()),
            ("tracking_comboBox_linker", get_linker_ui_labels()),
        ):
            combo = getattr(self.parent, combo_name)
            combo.clear()
            combo.addItems(labels)

    def _on_track_mode_toggled(self, combined_checked: bool) -> None:
        self.parent.tracking_comboBox_method.setEnabled(combined_checked)
        self.parent.tracking_comboBox_detector.setEnabled(not combined_checked)
        self.parent.tracking_comboBox_linker.setEnabled(not combined_checked)

    def setup_tree_widget(self, nta_items_data: list[dict[str,Any]]):
        tree = self.parent.tracking_treeWidget_folders
        tree.clear()

        for item_data in nta_items_data:
            nta_item = QTreeWidgetItem(tree)
            nta_item.setText(0, item_data["name"])
            nta_item.setToolTip(0, item_data["path"])
            nta_item.setFlags(nta_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            nta_item.setCheckState(0, Qt.Checked)

            for sub in item_data.get("subs", []):
                sub_item = QTreeWidgetItem(nta_item)
                sub_item.setText(0, f"{sub['name']} ({sub['tif_count']} TIFFs)")
                sub_item.setToolTip(0, sub["path"])
                sub_item.setFlags(sub_item.flags() | Qt.ItemIsUserCheckable)
                sub_item.setCheckState(0, Qt.Checked)
                # UserRole tuple is the contract consumed by TrackMateProcessor:
                # (sub_path, sorted_tiff_paths, nta_folder, sub_num).
                sub_item.setData(
                    0,
                    Qt.UserRole,
                    (
                        sub["path"],
                        sorted([os.path.join(sub["path"], f) for f in sub["tiffs"]]),
                        sub["nta_folder"],
                        sub["name"],
                    ),
                )

        tree.expandAll()

    def update_children_check_state(self, parent_item, new_state):
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            child.setCheckState(0, new_state)

    def update_parent_check_state(self, parent_item):
        checked_count = 0
        total_count = parent_item.childCount()

        for i in range(total_count):
            if parent_item.child(i).checkState(0) == Qt.Checked:
                checked_count += 1

        if checked_count == 0:
            parent_item.setCheckState(0, Qt.Unchecked)
        elif checked_count == total_count:
            parent_item.setCheckState(0, Qt.Checked)
        else:
            parent_item.setCheckState(0, Qt.PartiallyChecked)

    def clear_log(self):
        if hasattr(self.parent, 'tracking_textBrowser_logs'):
            self.parent.tracking_textBrowser_logs.clear()

    def append_log(self, message: str):
        if hasattr(self.parent, 'tracking_textBrowser_logs'):
            self.parent.tracking_textBrowser_logs.append(message)

    def update_save_path(self, path: str):
        if hasattr(self.parent, 'tracking_lineEdit_save'):
            self.parent.tracking_lineEdit_save.setText(path)

    def update_progress_bar(self, progress: int | None = None, msg: str | None = None):
        bar = self.parent.main_progressBar_progress
        if bar is None:
            return
        if progress is None:
            progress = bar.value()
        progress = max(0, min(100, int(progress)))
        bar.setValue(progress)
        bar.setFormat(f"{msg} %p%" if msg is not None else "%p%")

    def reset_progress_bar(self):
        bar = self.parent.main_progressBar_progress
        if bar is not None:
            bar.setValue(0)
            bar.setFormat("%p%")

    def set_processing_ui_enabled(self, enabled: bool):
        if hasattr(self.parent, 'tracking_pushButton_browse'):
            self.parent.tracking_pushButton_browse.setEnabled(enabled)

        if hasattr(self.parent, 'tracking_pushButton_runTracking'):
            self.parent.tracking_pushButton_runTracking.setEnabled(enabled)

        if hasattr(self.parent, 'tracking_pushButton_save'):
            self.parent.tracking_pushButton_save.setEnabled(enabled)

        if hasattr(self.parent, 'tracking_pushButton_FIJI'):
            self.parent.tracking_pushButton_FIJI.setEnabled(enabled)

        if hasattr(self.parent, 'tracking_treeWidget_folders'):
            self.parent.tracking_treeWidget_folders.setEnabled(enabled)

        if hasattr(self.parent, 'doubleSpinBox_Track_Radius'):
            self.parent.doubleSpinBox_Track_Radius.setEnabled(enabled)

        if hasattr(self.parent, 'doubleSpinBox_Track_Threshold'):
            self.parent.doubleSpinBox_Track_Threshold.setEnabled(enabled)

        if hasattr(self.parent, 'doubleSpinBox_Track_MaxLinkDistance'):
            self.parent.doubleSpinBox_Track_MaxLinkDistance.setEnabled(enabled)

        if hasattr(self.parent, 'doubleSpinBox_Track_MaxGapDistance'):
            self.parent.doubleSpinBox_Track_MaxGapDistance.setEnabled(enabled)

        if hasattr(self.parent, 'spinBox_Track_MaxFrameGap'):
            self.parent.spinBox_Track_MaxFrameGap.setEnabled(enabled)

        if hasattr(self.parent, 'checkBox_Track_FilterShortTracks'):
            self.parent.checkBox_Track_FilterShortTracks.setEnabled(enabled)

        if hasattr(self.parent, 'spinBox_Track_MinTrackDuration'):
            self.parent.spinBox_Track_MinTrackDuration.setEnabled(enabled)

    def show_no_selection_warning(self):
        QMessageBox.warning(
            self.parent,
            "No Selection",
            "Please select at least one NTA folder to process."
        )

    def show_fiji_not_found_error(self):
        QMessageBox.critical(
            self.parent,
            "FIJI Not Found",
            "FIJI executable not found. Please set the FIJI path first."
        )

    def show_invalid_params_error(self, error_msg: str):
        QMessageBox.critical(
            self.parent,
            "Invalid Parameters",
            f"Invalid parameter values:\n{error_msg}"
        )

    def show_already_running_info(self):
        QMessageBox.information(
            self.parent,
            "Already Running",
            "TrackMate processing is already in progress."
        )

    def show_completion_message(self, success, message: str):
        if success:
            QMessageBox.information(
                self.parent,
                "Processing Complete",
                message or "TrackMate processing completed successfully."
            )
        else:
            QMessageBox.warning(
                self.parent,
                "Processing Error",
                message or "TrackMate processing encountered errors."
            )
