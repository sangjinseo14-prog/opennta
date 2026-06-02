"""U-Net inference runner: runtime bootstrap, model loading, and orchestration."""

import json
import logging
import warnings
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np

from .config import (
    DT_OBS,
    MA_WINDOW,
    NORM_PATH,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REFERENCE_FIELD_PATH,
    WEIGHTS_PATH,
)
from .cuda_runtime import preload_cuda_shared_libs, run_nvidia_smi, runtime_snapshot

logger = logging.getLogger(__name__)


_BUNDLE_API: SimpleNamespace | None = None


def _load_bundle_api() -> SimpleNamespace:
    global _BUNDLE_API
    if _BUNDLE_API is not None:
        return _BUNDLE_API

    # Delay heavy project imports so TensorFlow can probe GPU devices
    # before matplotlib/cv2/pandas native libraries are loaded.
    from .field_builder import (
        build_actual_sample_from_flow_df,
        build_dual_coordinate_prediction,
        denormalize_uv,
    )
    from .io import (
        load_norm_json,
        load_reference_field_npz,
        make_sample_output_dir,
        save_pred_field_csv,
    )
    from .model import build_unet, input_channels
    from .preprocessing import preprocess_actual_spots_to_flow
    from .visualization import save_sc_ac_prediction_panel

    _BUNDLE_API = SimpleNamespace(
        build_unet=build_unet,
        input_channels=input_channels,
        build_dual_coordinate_prediction=build_dual_coordinate_prediction,
        build_actual_sample_from_flow_df=build_actual_sample_from_flow_df,
        denormalize_uv=denormalize_uv,
        load_norm_json=load_norm_json,
        load_reference_field_npz=load_reference_field_npz,
        make_sample_output_dir=make_sample_output_dir,
        preprocess_actual_spots_to_flow=preprocess_actual_spots_to_flow,
        save_pred_field_csv=save_pred_field_csv,
        save_sc_ac_prediction_panel=save_sc_ac_prediction_panel,
    )
    return _BUNDLE_API


def load_weights_compat(model, weights_path) -> None:
    weights_path = str(weights_path)
    primary_exc: Exception | None = None
    try:
        model.load_weights(weights_path)
        return
    except (ValueError, OSError, KeyError) as exc:
        if not weights_path.endswith(".h5"):
            raise
        # Rebind because Python clears the as-name when the except block exits;
        # the fallback's error message below still needs the original exception.
        primary_exc = exc

    # Keras 3 can miss legacy HDF5 weights from older TF/Keras releases.
    # Fall back to the legacy loader so the original bundle stays usable.
    import h5py
    from keras.src.legacy.saving import legacy_h5_format

    try:
        with h5py.File(weights_path, "r") as f:
            legacy_h5_format.load_weights_from_hdf5_group(f, model)
    except Exception as fallback_exc:
        raise RuntimeError(
            f"Failed to load weights from {weights_path}: "
            f"primary loader raised {type(primary_exc).__name__}: {primary_exc}; "
            f"legacy fallback raised {type(fallback_exc).__name__}: {fallback_exc}"
        ) from fallback_exc


_RUNTIME_DEVICE_WARNED = False


def ensure_runtime_ready() -> None:
    global _RUNTIME_DEVICE_WARNED
    if _RUNTIME_DEVICE_WARNED:
        return

    import tensorflow as tf

    if tf.test.is_built_with_cuda():
        loaded, missing = preload_cuda_shared_libs()
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            _RUNTIME_DEVICE_WARNED = True
            return

        snapshot = runtime_snapshot()
        diag_bits = []
        if missing:
            missing_names = ", ".join(name for name, _ in missing)
            diag_bits.append(f"missing CUDA libs: {missing_names}")
        if loaded:
            diag_bits.append(f"preloaded CUDA libs: {len(loaded)}")
        smi_rc, _, smi_stderr = run_nvidia_smi()
        diag_bits.append(f"nvidia-smi exit_code={smi_rc}")
        if smi_stderr:
            diag_bits.append(f"nvidia-smi stderr={smi_stderr}")
        diag_bits.append(f"CUDA_VISIBLE_DEVICES={snapshot['cuda_visible_devices']}")
        diag_bits.append(f"LD_LIBRARY_PATH={snapshot['ld_library_path']}")

        warnings.warn(
            "TensorFlow is GPU-enabled but no GPU was detected. "
            "Inference will continue on CPU. "
            "Run `python gpu_check.py` and make sure `nvidia-smi` works if GPU use is expected. "
            + " | ".join(diag_bits),
            RuntimeWarning,
            stacklevel=2,
        )
        _RUNTIME_DEVICE_WARNED = True
        return

    warnings.warn(
        "GPU-enabled TensorFlow was not detected. Inference will continue on CPU.",
        RuntimeWarning,
        stacklevel=2,
    )
    _RUNTIME_DEVICE_WARNED = True


class UNetInferenceRunner:
    def __init__(self, weights_path=WEIGHTS_PATH, norm_path=NORM_PATH, reference_field_path=REFERENCE_FIELD_PATH):
        ensure_runtime_ready()
        self.api = _load_bundle_api()
        self.weights_path = Path(weights_path)
        self.norm_path = Path(norm_path)
        self.reference_field_path = Path(reference_field_path)
        self.norm = self.api.load_norm_json(self.norm_path)
        self.xs_ref, self.ys_ref, self.U_ref, self.V_ref = self.api.load_reference_field_npz(self.reference_field_path)
        H, W = self.U_ref.shape
        self.model = self.api.build_unet((H, W, self.api.input_channels()))
        load_weights_compat(self.model, self.weights_path)

    def prepare_sample_from_df(self, spots_df=None, csv_path=None, ma_window=None, dt_obs=None,
                               image_width_px=None, image_height_px=None):
        # Returns {"flow_df", "meta", "sample"}; accepts either an in-memory
        # DataFrame or a CSV path (CLI uses the latter).
        ma_window = MA_WINDOW if ma_window is None else ma_window
        dt_obs = DT_OBS if dt_obs is None else dt_obs

        flow_df, meta = self.api.preprocess_actual_spots_to_flow(
            xs_ref=self.xs_ref,
            ys_ref=self.ys_ref,
            ma_window=ma_window,
            dt_obs=dt_obs,
            spots_df=spots_df,
            csv_path=csv_path,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )
        meta["xs_ref"] = self.xs_ref
        meta["ys_ref"] = self.ys_ref

        sample = self.api.build_actual_sample_from_flow_df(
            flow_df=flow_df,
            xs=self.xs_ref,
            ys=self.ys_ref,
            norm=self.norm,
            seed=RANDOM_SEED,
        )
        return {"flow_df": flow_df, "meta": meta, "sample": sample}

    def infer_from_sample(self, sample, meta):
        ensure_runtime_ready()
        pred = self.model.predict(sample["X"][None, ...], batch_size=1, verbose=0)[0]
        pred_u, pred_v = self.api.denormalize_uv(pred[..., 0], pred[..., 1], self.norm)
        dual = self.api.build_dual_coordinate_prediction(
            pred_u, pred_v, self.xs_ref, self.ys_ref, meta["position_transform"],
        )
        dual["position_transform"] = meta["position_transform"]
        dual["sample"] = sample
        dual["meta"] = meta
        return dual

    def infer_from_df(self, spots_df, ma_window=None, dt_obs=None,
                      image_width_px=None, image_height_px=None):
        prep = self.prepare_sample_from_df(
            spots_df=spots_df, ma_window=ma_window, dt_obs=dt_obs,
            image_width_px=image_width_px, image_height_px=image_height_px,
        )
        result = self.infer_from_sample(prep["sample"], prep["meta"])
        result["flow_df"] = prep["flow_df"]
        return result

    def run_and_save_outputs(self, spots_input, output_dir=None, prefix=None):
        spots_input = Path(spots_input)
        output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = prefix or spots_input.stem
        sample_output_dir = self.api.make_sample_output_dir(output_dir, stem)

        prep = self.prepare_sample_from_df(csv_path=spots_input)
        result = self.infer_from_sample(prep["sample"], prep["meta"])

        npz_out = sample_output_dir / f"{stem}_pred_maps.npz"
        pred_field_sc_csv = sample_output_dir / f"{stem}_pred_field_sc.csv"
        pred_field_ac_csv = sample_output_dir / f"{stem}_pred_field_ac.csv"
        coord_panel_png = sample_output_dir / f"{stem}_pred_acsc_panel.png"
        summary_json = sample_output_dir / f"{stem}_summary.json"

        np.savez_compressed(
            npz_out,
            pred_u=result["pred_u_sc"],
            pred_v=result["pred_v_sc"],
            pred_speed=result["pred_speed_sc"],
            pred_u_sc=result["pred_u_sc"],
            pred_v_sc=result["pred_v_sc"],
            pred_speed_sc=result["pred_speed_sc"],
            pred_u_ac=result["pred_u_ac"],
            pred_v_ac=result["pred_v_ac"],
            pred_speed_ac=result["pred_speed_ac"],
            xs=self.xs_ref.astype(np.float32),
            ys=self.ys_ref.astype(np.float32),
            xs_sc=result["xs_sc"],
            ys_sc=result["ys_sc"],
            xs_ac=result["xs_ac"],
            ys_ac=result["ys_ac"],
            xs_ac_px=result["xs_ac_px"],
            ys_ac_px=result["ys_ac_px"],
        )
        self.api.save_pred_field_csv(result["pred_u_sc"], result["pred_v_sc"], result["xs_sc"], result["ys_sc"], pred_field_sc_csv)
        self.api.save_pred_field_csv(result["pred_u_ac"], result["pred_v_ac"], result["xs_ac"], result["ys_ac"], pred_field_ac_csv)
        self.api.save_sc_ac_prediction_panel(stem, result, coord_panel_png)

        meta = prep["meta"]
        summary = {
            "mode": "fast",
            "spots_input": str(spots_input),
            "sample_output_dir": str(sample_output_dir),
            "pred_npz": str(npz_out),
            "pred_field_sc_csv": str(pred_field_sc_csv),
            "pred_field_ac_csv": str(pred_field_ac_csv),
            "coord_panel_png": str(coord_panel_png),
            "n_raw_rows": int(meta["n_raw_rows"]),
            "n_raw_particles": int(meta["n_raw_particles"]),
            "n_flow_rows": int(meta["n_flow_rows"]),
            "n_flow_particles": int(meta["n_flow_particles"]),
            "frame_min": int(meta["frame_min"]),
            "frame_max": int(meta["frame_max"]),
            "pred_u_mean": float(np.mean(result["pred_u_sc"])),
            "pred_v_mean": float(np.mean(result["pred_v_sc"])),
            "pred_speed_mean": float(np.mean(result["pred_speed_sc"])),
            "pred_speed_max": float(np.max(result["pred_speed_sc"])),
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "summary": summary,
            "pred_u": result["pred_u_sc"],
            "pred_v": result["pred_v_sc"],
            "pred_u_sc": result["pred_u_sc"],
            "pred_v_sc": result["pred_v_sc"],
            "pred_u_ac": result["pred_u_ac"],
            "pred_v_ac": result["pred_v_ac"],
            "sample": prep["sample"],
            "flow_df": prep["flow_df"],
        }


_RUNNER: UNetInferenceRunner | None = None
_RUNNER_LOCK = Lock()


def get_shared_runner() -> UNetInferenceRunner:
    # First call loads Keras weights + reference field + norm stats and is
    # slow; subsequent calls are free.
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            logger.info("Loading U-Net inference runner (one-time)")
            _RUNNER = UNetInferenceRunner()
        return _RUNNER


def clear_shared_runner() -> None:
    global _RUNNER
    with _RUNNER_LOCK:
        _RUNNER = None
