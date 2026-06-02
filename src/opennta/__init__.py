"""Top-level package for the OpenNTA application suite."""

from importlib.metadata import PackageNotFoundError, version as _metadata_version

try:
    __version__ = _metadata_version("opennta")
except PackageNotFoundError:
    try:
        from ._version import version as __version__
    except Exception:
        __version__ = "1.0.0 (or higher)"

# On Windows, if PyQt5 (the Qt5 runtime) is loaded before TensorFlow,
# TensorFlow's _pywrap_tensorflow_common.dll may fail in DllMain with
# 0x45A (DLL_INIT_FAILED). Because the submodule import chain brings in
# PyQt5 (through application -> main_window -> _ui_builder -> PyQt5),
# this check must run before the `from . import ...` statements below.
#
# On macOS / Linux this DLL ordering issue does not apply, so we skip the
# preload — TensorFlow stays a fully optional, lazy import (~hundreds of MB
# and 1–2 s startup) and users who never touch the U-Net feature don't pay
# the cost.

import os as _os
import sys as _sys

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR

if _sys.platform == "win32":
    try:
        import tensorflow as _tf  # noqa: F401  (Windows DLL preload only)
    except Exception as _tf_preload_exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "TensorFlow preload failed (U-Net correction will be unavailable): %s",
            _tf_preload_exc,
        )

from . import analysis, application, tracking

__all__ = [
    "__version__",
    "analysis",
    "application",
    "tracking",
]
