from pathlib import Path

from PyQt5.QtCore import Qt

from ..common.path_selector import PathSelector
from .types import FileInfo


class TabBatchAnalysisHelper:

    def __init__(self, parent):
        self.parent = parent
        self.result_folder: str = None

    def browse_result_folder(self):
        folder = PathSelector.select_folder(
            self.parent,
            self.result_folder,
            "Select Result Folder"
        )

        if folder:
            self.result_folder = folder
            self.parent.batch_lineEdit_browse.setText(folder)

        return folder

    def discover_nta_spot_files(
        self,
        folder: str | None
    ) -> tuple[list[tuple[Path, list[Path]]], list[FileInfo]]:
        if not folder:
            return [], []

        folder_path = Path(folder)

        if not folder_path.exists():
            return [], []

        # Search selected folder and all subfolders regardless of folder name.
        all_spot_file_paths: list[Path] = sorted(folder_path.rglob("*_spots.csv"))

        folder_group = 1
        spot_files: list[FileInfo] = []
        folders_with_files: list[tuple[Path, list[Path]]] = []

        grouped: dict[Path, list[Path]] = {}

        for spot_file_path in all_spot_file_paths:
            parent_folder = spot_file_path.parent
            grouped.setdefault(parent_folder, []).append(spot_file_path)

        for target_folder in sorted(grouped.keys(), key=lambda p: str(p).lower()):
            spot_file_paths = sorted(grouped[target_folder])

            folders_with_files.append((target_folder, spot_file_paths))

            for spot_file_path in spot_file_paths:
                file_info = {
                    'name': spot_file_path.name,
                    'path': str(spot_file_path),
                    'folder': target_folder.name,
                    'group': folder_group,
                    'tree_item': None
                }
                spot_files.append(file_info)

            folder_group += 1

        return folders_with_files, spot_files

    def get_available_group_numbers(self, spot_files: list[FileInfo]):
        used = set(f['group'] for f in spot_files)
        max_used = max(used) if used else 1
        available = list(range(1, max_used + 2))
        return [str(g) for g in available]

    def get_checked_files(self, spot_files: list[FileInfo]) -> list[tuple[int, FileInfo]]:
        index_checked_files: list[tuple[int, FileInfo]] = []
        for idx, file_info in enumerate(spot_files):
            tree_item = file_info.get("tree_item")
            if tree_item and tree_item.checkState(0) == Qt.Checked:
                index_checked_files.append((idx, file_info))
        return index_checked_files

    def update_file_group(self, spot_files: list[FileInfo], row: int, group_num: int):
        if 0 <= row < len(spot_files):
            spot_files[row]['group'] = group_num

    def find_row_index_by_path(self, spot_files: list[FileInfo], file_path: str) -> int:
        for row, file_info in enumerate(spot_files):
            if file_info['path'] == file_path:
                return row
        return -1

    def get_result_folder(self) -> str:
        return self.result_folder

