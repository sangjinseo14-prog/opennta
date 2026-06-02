import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SIM_META_PATH


def load_norm_json(norm_path):
    with open(norm_path, encoding="utf-8") as f:
        return json.load(f)


def load_reference_field_npz(npz_path):
    z = np.load(npz_path)
    return z["xs"].astype(np.float32), z["ys"].astype(np.float32), z["U"].astype(np.float32), z["V"].astype(np.float32)


def load_simulation_meta_csv(meta_csv_path=SIM_META_PATH):
    df = pd.read_csv(meta_csv_path)
    meta = {}
    for _, row in df.iterrows():
        key = str(row["key"])
        value = row["value"]
        try:
            value_num = float(value)
            if value_num.is_integer():
                value = int(value_num)
            else:
                value = value_num
        except Exception:
            pass
        meta[key] = value
    return meta


def make_sample_output_dir(output_dir, sample_name):
    sample_dir = Path(output_dir) / str(sample_name)
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def save_pred_field_csv(pred_u, pred_v, xs, ys, out_csv):
    pred_u = np.asarray(pred_u, dtype=np.float32)
    pred_v = np.asarray(pred_v, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    df = pd.DataFrame(
        {
            "x": xx.reshape(-1),
            "y": yy.reshape(-1),
            "u": pred_u.reshape(-1),
            "v": pred_v.reshape(-1),
        }
    )
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def maybe_read_headerless_spots(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    header_is_weird = False
    if df.shape[1] >= 4:
        cols = list(df.columns)
        if all(str(c).replace(".", "", 1).replace("-", "", 1).isdigit() for c in cols[:4]):
            header_is_weird = True
    if header_is_weird:
        df = pd.read_csv(csv_path, header=None)
    return df


def load_actual_spots_raw(csv_path: Path) -> pd.DataFrame:
    df = maybe_read_headerless_spots(csv_path)
    if df.shape[1] < 4:
        raise ValueError(f"{csv_path} must have at least 4 columns. Got shape={df.shape}")

    df = df.iloc[:, :5].copy()
    rename_map = {
        df.columns[0]: "particle_id",
        df.columns[1]: "x",
        df.columns[2]: "y",
        df.columns[3]: "frame_idx",
    }
    if df.shape[1] >= 5:
        rename_map[df.columns[4]] = "extra"
    df = df.rename(columns=rename_map)

    for c in ["particle_id", "x", "y", "frame_idx"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["particle_id", "x", "y", "frame_idx"]).copy()
    df["particle_id"] = df["particle_id"].astype(np.int64)
    df["frame_idx"] = df["frame_idx"].astype(np.int64)
    df["x"] = df["x"].astype(np.float32)
    df["y"] = df["y"].astype(np.float32)
    df = df.drop_duplicates(subset=["particle_id", "frame_idx"])
    return df.sort_values(["particle_id", "frame_idx"]).reset_index(drop=True)
