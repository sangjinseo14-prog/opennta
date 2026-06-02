import logging
import os
import platform
import re
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QStandardPaths, Qt
from PyQt5.QtWidgets import QFileDialog

from ..config import ensure_app_temp_dir

logger = logging.getLogger(__name__)

_FIJI_JAR_VER_RE = re.compile(r"-(?P<ver>[\w.\-]+)\.jar$")


def detect_fiji_versions(fiji_executable: str) -> dict[str, str]:
    p = Path(fiji_executable)
    root = p.parent.parent.parent if p.parent.name == "MacOS" else p.parent
    versions: dict[str, str] = {}
    for key, sub, prefix in (
        ("imagej",    "jars",    "ij-"),
        ("trackmate", "jars", "TrackMate-"),
    ):
        for jar in sorted((root / sub).glob(f"{prefix}*.jar")):
            m = _FIJI_JAR_VER_RE.search(jar.name)
            if m:
                versions[key] = m.group("ver")
                break
    return versions


class TabTrackingHelper:

    def __init__(self, parent):
        self.parent = parent

    def get_fiji_record_file(self) -> str:
        temp_dir = ensure_app_temp_dir()
        return str(temp_dir / "fiji_path.txt")

    def read_fiji_record(self) -> str | None:
        record_file = self.get_fiji_record_file()
        try:
            if os.path.isfile(record_file):
                with open(record_file, encoding="utf-8") as f:
                    path = Path(f.read().strip())
                if path.exists():
                    return str(path)
        except OSError as e:
            logger.warning("Could not read Fiji record: %s", e)
            return None

    def write_fiji_record(self, fiji_path) -> bool:
        record_file = self.get_fiji_record_file()
        try:
            os.makedirs(os.path.dirname(record_file), exist_ok=True)
            with open(record_file, "w", encoding="utf-8") as f:
                f.write(os.path.normpath(fiji_path))
            return True
        except OSError as e:
            logger.error("Error writing Fiji record: %s", e)
            return False

    def find_fiji_in_common_places(self) -> str | None:
        system = platform.system()
        common_paths = []
        home = Path.home()

        if system == "Windows":
            common_paths = [
                home / "Desktop" / "Fiji.app" / "ImageJ-win64.exe",
                home / "Fiji.app" / "ImageJ-win64.exe",
                Path("C:/Fiji.app/ImageJ-win64.exe"),
                Path("C:/Program Files/Fiji.app/ImageJ-win64.exe"),
                Path("C:/Program Files (x86)/Fiji.app/ImageJ-win64.exe"),
            ]
        elif system == "Darwin":
            common_paths = [
                Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
                home / "Applications" / "Fiji.app" / "Contents" / "MacOS" / "ImageJ-macosx",
            ]
        else:
            common_paths = [
                home / "Fiji.app" / "ImageJ-linux64",
                Path("/opt/Fiji.app/ImageJ-linux64"),
                Path("/usr/local/Fiji.app/ImageJ-linux64"),
            ]

        for path in common_paths:
            if path.exists():
                return str(path)

        return None

    def _fiji_dialog_start_dir(self):
        desktop = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation) or str(
            Path.home() / "Desktop"
        )
        return desktop if os.path.isdir(desktop) else str(Path.home())

    def _fiji_dialog_filter(self):
        system = platform.system()
        if system == "Windows":
            return "ImageJ executable (ImageJ-win64.exe);;Executable Files (*.exe);;All Files (*)"
        elif system == "Darwin":
            return "ImageJ executable (ImageJ-macosx);;All Files (*)"
        else:
            return "ImageJ executable (ImageJ-linux64);;All Files (*)"

    def get_fiji_path(self, force_prompt: bool = False, remember: bool = True) -> str | None:
        if not force_prompt:
            saved_path = self.read_fiji_record()
            if saved_path and os.path.exists(saved_path):
                return saved_path

            auto_path = self.find_fiji_in_common_places()
            if auto_path:
                if remember:
                    self.write_fiji_record(auto_path)
                return auto_path

        return self.prompt_user_select_fiji(write_record=remember)

    def prompt_user_select_fiji(self, write_record: bool = True, start_dir: str | None = None) -> str | None:
        if not start_dir:
            start_dir = self.read_fiji_record()

        if not start_dir:
            start_dir = self._fiji_dialog_start_dir()

        fiji_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Select Fiji/ImageJ executable",
            start_dir,
            self._fiji_dialog_filter(),
        )

        if fiji_path and os.path.exists(fiji_path):
            if write_record:
                self.write_fiji_record(fiji_path)
            return fiji_path

        return None

    def _cleanup_temp_files(self, nta_folder: str, sub_path: str, sub: str):
        sample = f"{nta_folder}_{sub}"
        candidates = [
            os.path.join(sub_path, f"{sample}.tif"),
            os.path.join(sub_path, f"{sample}.tiff"),
            os.path.join(sub_path, f"{sample}_first_frame.tif"),
            os.path.join(sub_path, f"{sample}_first_frame.tiff"),
        ]
        for fp in candidates:
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except Exception:
                logger.exception("Failed to remove temporary file %s", fp)

    def browse_folder(self):
        last = self.parent.tracking_lineEdit_browse.text().strip()
        if last and os.path.isdir(last):
            start_dir = last
        else:
            start_dir = self._fiji_dialog_start_dir()

        folder_path = QFileDialog.getExistingDirectory(
            self.parent,
            "Select the parent folder of NTA samples",
            start_dir,
        )

        if folder_path:
            self.parent.tracking_lineEdit_browse.setText(folder_path)
            return folder_path
        return None

    def browse_save_folder(self):
        save_path = QFileDialog.getExistingDirectory(
            self.parent,
            "Select folder to save results",
            "",
        )

        if save_path:
            if hasattr(self.parent, "tracking_lineEdit_save"):
                self.parent.tracking_lineEdit_save.setText(save_path)
            return save_path
        return None

    def get_save_path(self):
        if hasattr(self.parent, "tracking_lineEdit_save"):
            return self.parent.tracking_lineEdit_save.text().strip()
        return ""

    def scan_nta_folders(self, root_dir: str) -> list[dict[str, Any]]:
        nta_items_data = []
        root_path = Path(root_dir)

        # Accept either the parent directory that contains NTA_* folders or a
        # single NTA_* folder; both shapes occur in saved experiment layouts.
        candidate_roots = [root_path]

        nta_dirs = []
        for base_root in candidate_roots:
            for dirpath, dirnames, _ in os.walk(base_root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]

                base = os.path.basename(dirpath)
                if base.startswith("NTA") and os.path.isdir(dirpath):
                    nta_dirs.append(dirpath)
                    dirnames[:] = []  # stop os.walk from descending into the NTA tree

        for nta_dir in sorted(nta_dirs):
            nta_folder = os.path.basename(nta_dir)
            subs_data = []

            try:
                subs = [s for s in os.listdir(nta_dir)]
            except (OSError, PermissionError) as e:
                logger.warning("Cannot read NTA directory %s: %s", nta_dir, e)
                subs = []

            for sub in subs:
                sub_path = os.path.join(nta_dir, sub)
                if os.path.isdir(sub_path) and sub.isdigit():
                    self._cleanup_temp_files(nta_folder, sub_path, sub)

                    try:
                        tiffs = [
                            f for f in os.listdir(sub_path)
                            if f.lower().endswith((".tif", ".tiff"))
                        ]
                    except (OSError, PermissionError):
                        tiffs = []

                    if len(tiffs) > 1:
                        logger.error(
                            "Multiple TIFF files found in %s (%d); skipping folder.",
                            sub_path, len(tiffs),
                        )
                        continue

                    if tiffs:
                        subs_data.append(
                            {
                                "name": sub,
                                "path": sub_path,
                                "tiffs": tiffs,
                                "tif_count": len(tiffs),
                                "nta_folder": nta_folder,
                            }
                        )

            if subs_data:
                nta_items_data.append({"name": nta_folder, "path": nta_dir, "subs": subs_data})

        return nta_items_data

    def get_selected_folders(self) -> list[Any]:
        selected = []
        root = self.parent.tracking_treeWidget_folders.invisibleRootItem()

        for i in range(root.childCount()):
            nta_item = root.child(i)

            for j in range(nta_item.childCount()):
                sub_item = nta_item.child(j)
                if sub_item.checkState(0) == Qt.Checked:
                    data = sub_item.data(0, Qt.UserRole)
                    if data:
                        selected.append(data)

        return selected

    def get_root_folder(self, selected_folders) -> str:
        browsed = self.parent.tracking_lineEdit_browse.text().strip()
        if browsed and os.path.isdir(browsed):
            return browsed
        return None

