from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHeaderView,
    QMessageBox,
    QProgressBar,
    QTreeWidgetItem,
    QWidget,
)

from ...analysis.drift_corrector import (
    drift_correction_ui_label_to_mode,
    get_drift_correction_ui_labels,
)
from ...analysis.size_distributor import (
    get_size_distribution_ui_labels,
    size_distribution_ui_label_to_mode,
)
from .types import FileInfo

DEFAULT_LAG_FRAME = 2


def _update_progress_bar(bar, progress: int | None = None, msg: str | None = None) -> None:
    if bar is None:
        return
    if progress is None:
        progress = bar.value()
    progress = max(0, min(100, int(progress)))
    bar.setValue(progress)
    bar.setFormat(f"{msg} %p%" if msg is not None else "%p%")


def _reset_progress_bar(bar) -> None:
    if bar is not None:
        bar.setValue(0)
        bar.setFormat("%p%")


def _set_widgets_enabled(parent, widget_names: list, enabled: bool) -> None:
    for name in widget_names:
        w = getattr(parent, name, None)
        if w is None:
            w = parent.findChild(QWidget, name)
        if w is not None:
            w.setEnabled(enabled)


def _get_lag_frame(line_edit) -> int:
    try:
        return int(line_edit.text().strip())
    except (ValueError, TypeError, AttributeError):
        return DEFAULT_LAG_FRAME


def _set_lag_seconds(lag_frame_widget, lag_sec_widget, fps: float) -> None:
    try:
        lag_frames = int(lag_frame_widget.text())
        lag_sec_widget.setText(str(round(lag_frames / fps, 3)))
    except (ValueError, TypeError, AttributeError):
        pass


class SpotFileTableModel(QAbstractTableModel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.headers = ["File Name", "Group", "Progress"]
        self.spot_files: list[FileInfo] = []

    def rowCount(self, parent=QModelIndex()):
        return len(self.spot_files)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return self.spot_files[row]['name']
        return None

    def clear_files(self):
        self.spot_files = []
        self.layoutChanged.emit()


class NoScrollCombo(QComboBox):
    def wheelEvent(self, e):
        e.ignore()


class TabBatchAnalysisUi:

    def __init__(self, parent, table_model):
        self.parent = parent
        self.table_model = table_model
        self._populate_comboboxes()

    def _populate_comboboxes(self) -> None:
        combo = self.parent.batch_comboBox_distributionMode
        combo.clear()
        combo.addItems(get_size_distribution_ui_labels())

        combo = self.parent.batch_comboBox_correctionMode
        combo.clear()
        combo.addItems(get_drift_correction_ui_labels())

    def setup_table_view(self):
        self.parent.batch_tableView_progress.setModel(self.table_model)

        header = self.parent.batch_tableView_progress.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)

        self.parent.batch_tableView_progress.setColumnWidth(1, 80)
        self.parent.batch_tableView_progress.setColumnWidth(2, 150)

    def populate_tree(self, nta_folders, all_files):
        self.parent.batch_treeWidget_folders.clear()

        self.parent.batch_treeWidget_folders.blockSignals(True)
        try:
            for nta_folder, spot_files in nta_folders:
                folder_item = QTreeWidgetItem(self.parent.batch_treeWidget_folders)
                folder_item.setText(0, nta_folder.name)
                folder_item.setFlags(
                    folder_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate
                )
                folder_item.setCheckState(0, Qt.Checked)

                for spot_file in spot_files:
                    file_item = QTreeWidgetItem(folder_item)
                    file_item.setText(0, spot_file.name)
                    file_item.setFlags(file_item.flags() | Qt.ItemIsUserCheckable)
                    file_item.setCheckState(0, Qt.Checked)
                    file_item.setData(0, Qt.UserRole, str(spot_file))

                    for file_info in all_files:
                        if file_info['path'] == str(spot_file):
                            file_info['tree_item'] = file_item
                            break
        finally:
            self.parent.batch_treeWidget_folders.blockSignals(False)

        self.parent.batch_treeWidget_folders.expandAll()

    def update_table_widgets(self, group_options_callback, group_callback):
        for row, file_info in enumerate(self.table_model.spot_files):
            group_cb = NoScrollCombo()
            opts = group_options_callback()
            group_cb.addItems(opts)
            group_cb.setCurrentText(str(file_info.get('group', 1)))
            group_cb.currentTextChanged.connect(lambda text, r=row: group_callback(r, text))
            self.parent.batch_tableView_progress.setIndexWidget(self.table_model.index(row, 1), group_cb)

            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            self.parent.batch_tableView_progress.setIndexWidget(
                self.table_model.index(row, 2), progress
            )

    def update_group_comboboxes(self, group_options:list[str]):
        for r in range(len(self.table_model.spot_files)):
            widget = self.parent.batch_tableView_progress.indexWidget(
                self.table_model.index(r, 1)
            )
            if isinstance(widget, QComboBox):
                old_text = widget.currentText()
                widget.blockSignals(True)
                widget.clear()
                widget.addItems(group_options)
                widget.setCurrentText(old_text)
                widget.blockSignals(False)

    def enable_table_row_widgets(self, row: int, enabled: bool):

        group_widget = self.parent.batch_tableView_progress.indexWidget(
            self.table_model.index(row, 1)
        )
        if group_widget:
            group_widget.setEnabled(enabled)

    def enable_all_table_widgets(self, enabled: bool):
        if hasattr(self.parent, 'batch_tableView_progress'):
            self.parent.batch_tableView_progress.setEnabled(True)

            for row in range(len(self.table_model.spot_files)):
                self.enable_table_row_widgets(row, enabled)

    def update_file_progress(self, file_idx: int, progress: int):
        widget = self.parent.batch_tableView_progress.indexWidget(
            self.table_model.index(file_idx, 2)
        )
        if isinstance(widget, QProgressBar):
            _update_progress_bar(widget, progress)

    def update_overall_progress(self, progress: int):
        _update_progress_bar(self.parent.main_progressBar_progress, progress)

    def reset_all_progress_bars(self):
        _reset_progress_bar(self.parent.main_progressBar_progress)

        for row in range(len(self.table_model.spot_files)):
            widget = self.parent.batch_tableView_progress.indexWidget(
                self.table_model.index(row, 2)
            )
            if isinstance(widget, QProgressBar):
                widget.setValue(0)

    def show_completion_message(self, success: bool, message: str):
        if success:
            QMessageBox.information(self.parent, "Complete", message)
        else:
            QMessageBox.critical(self.parent, "Error", message)

    def show_no_files_warning(self):
        QMessageBox.warning(
            self.parent,
            "No Files Selected",
            "Please select at least one file to analyze"
        )

    def show_error(self, error_message: str):
        QMessageBox.critical(self.parent, "Error", error_message)

    def set_processing_ui_enabled(self, enabled: bool):
        widget_names = [
            "batch_pushButton_browse",
            "batch_pushButton_config",
            "batch_pushButton_runAnalysis",
            "batch_lineEdit_lagFrame",
            "batch_lineEdit_r2Threshold",
            "batch_comboBox_correctionMode",
            "batch_comboBox_distributionMode",
            "batch_checkBox_csv",
            "batch_treeWidget_folders",
        ]
        _set_widgets_enabled(self.parent, widget_names, enabled)

    def read_lag_frame_and_refresh_seconds(self, lag_frame_widget, lag_sec_widget, fps: float):
        lag_frame = _get_lag_frame(lag_frame_widget)
        _set_lag_seconds(lag_frame_widget, lag_sec_widget, fps)
        return lag_frame

    def get_correction_mode(self) -> str:
        return drift_correction_ui_label_to_mode(
            self.parent.batch_comboBox_correctionMode.currentText()
        )

    def get_distribution_mode(self) -> str:
        return size_distribution_ui_label_to_mode(
            self.parent.batch_comboBox_distributionMode.currentText()
        )
