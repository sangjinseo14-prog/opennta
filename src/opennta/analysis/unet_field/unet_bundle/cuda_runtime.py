from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

REQUIRED_CUDA_LIBS = [
    "libcuda.so.1",
    "libcudart.so.12",
    "libcudnn.so.9",
    "libcublas.so.12",
    "libcusolver.so.11",
    "libcusparse.so.12",
    "libcufft.so.11",
]


def _dlopen_mode() -> int:
    mode = 0
    mode |= getattr(ctypes, "RTLD_NOW", 0)
    mode |= getattr(ctypes, "RTLD_GLOBAL", 0)
    return mode


def preload_cuda_shared_libs():
    loaded = []
    missing = []
    mode = _dlopen_mode()
    for name in REQUIRED_CUDA_LIBS:
        try:
            ctypes.CDLL(name, mode=mode)
            loaded.append(name)
        except OSError as exc:
            missing.append((name, str(exc)))
    return loaded, missing


def run_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return 1, "", "nvidia-smi: not found"

    result = subprocess.run([exe], capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def runtime_snapshot():
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "<unset>"),
    }
