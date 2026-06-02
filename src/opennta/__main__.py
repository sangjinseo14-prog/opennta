import logging
import os
import platform
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "opennta.OpenNTA"
        )
    except Exception:
        pass

from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from . import __version__  # noqa: E402
from .application import OpenNtaMainWindow  # noqa: E402

logger = logging.getLogger(__name__)


def _get_user_log_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "opennta" / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "opennta"
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / "opennta" / "logs"


def _configure_logging() -> Path | None:
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(logging.INFO)

    logging.getLogger("qdarkstyle").setLevel(logging.WARNING)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    try:
        log_dir = _get_user_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"opennta_{ts}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))
        root.addHandler(file_handler)
        return log_path
    except OSError as e:
        root.warning("Could not enable file logging: %s", e)
        return None


def main():
    log_path = _configure_logging()
    logger.info(
        "opennta %s starting (Python %s, %s)",
        __version__,
        platform.python_version(),
        sys.platform,
    )
    if log_path is not None:
        logger.info("logging to %s", log_path)

    try:
        app = QApplication(sys.argv)

        window = OpenNtaMainWindow()

        window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)
        window.show()
        window.raise_()
        window.activateWindow()
        QTimer.singleShot(0, lambda: (
            window.setWindowFlags(window.windowFlags() & ~Qt.WindowStaysOnTopHint),
            window.show(),
        ))

        logger.info("UI startup: success")
    except Exception:
        logger.exception("UI startup: failed")
        raise

    app.aboutToQuit.connect(lambda: logger.info("opennta shutting down"))

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
