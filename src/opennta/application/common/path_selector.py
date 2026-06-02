import os
from pathlib import Path

from PyQt5.QtCore import QStandardPaths
from PyQt5.QtWidgets import QFileDialog


class PathSelector:

    @staticmethod
    def select_csv_file(parent, last_dir: str = None,
                       title: str = "Select CSV File") -> tuple[str | None, str | None]:
        start_dir = PathSelector._get_start_directory(last_dir)

        path, _ = QFileDialog.getOpenFileName(
            parent,
            title,
            start_dir,
            "CSV Files (*.csv)"
        )

        if path:
            return path, os.path.dirname(path)
        return None, last_dir

    @staticmethod
    def save_csv_file(parent, last_dir: str = None,
                     default_name: str = "output.csv",
                     title: str = "Save CSV File") -> tuple[str | None, str | None]:
        start_dir = PathSelector._get_start_directory(last_dir)
        default_path = os.path.join(start_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            parent,
            title,
            default_path,
            "CSV Files (*.csv)"
        )

        if path:
            return path, os.path.dirname(path)
        return None, last_dir

    @staticmethod
    def save_html_file(parent, last_dir: str = None,
                      default_name: str = "report.html",
                      title: str = "Save HTML File") -> tuple[str | None, str | None]:
        start_dir = PathSelector._get_start_directory(last_dir)
        default_path = os.path.join(start_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            parent,
            title,
            default_path,
            "HTML Files (*.html)"
        )

        if path:
            return path, os.path.dirname(path)
        return None, last_dir

    @staticmethod
    def select_folder(parent, last_dir: str = None,
                     title: str = "Select Folder") -> str | None:
        start_dir = PathSelector._get_start_directory(last_dir)

        folder = QFileDialog.getExistingDirectory(
            parent,
            title,
            start_dir
        )

        return folder if folder else None

    @staticmethod
    def load_config_file(parent, last_dir: str = None) -> tuple[str | None, str | None]:
        start_dir = PathSelector._get_start_directory(last_dir)

        path, _ = QFileDialog.getOpenFileName(
            parent,
            "Load Config File",
            start_dir,
            "JSON Files (*.json)"
        )

        if path:
            return path, os.path.dirname(path)
        return None, last_dir

    @staticmethod
    def save_config_file(parent, last_dir: str = None,
                        default_name: str = "config.json") -> tuple[str | None, str | None]:
        start_dir = PathSelector._get_start_directory(last_dir)
        default_path = os.path.join(start_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            parent,
            "Save Config File",
            default_path,
            "JSON Files (*.json)"
        )

        if path:
            return path, os.path.dirname(path)
        return None, last_dir

    @staticmethod
    def _get_start_directory(last_dir: str = None) -> str:
        if last_dir and os.path.isdir(last_dir):
            return last_dir

        desktop = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
        if desktop and os.path.isdir(desktop):
            return desktop

        home = str(Path.home())
        desktop_fallback = os.path.join(home, "Desktop")

        return desktop_fallback if os.path.isdir(desktop_fallback) else home
