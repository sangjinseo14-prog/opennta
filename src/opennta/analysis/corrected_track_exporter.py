"""Qt-free exporter that writes drift-corrected tracks back to the spot-CSV
layout.

Shared by the single-file Analysis tab and the Batch tab, so it lives in the
domain layer rather than inside either tab's UI helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def write_corrected_track_csv(
    df: pd.DataFrame, original_csv_path: str, output_path: str, pixel_size: float
) -> int | None:

    df_raw = pd.read_csv(original_csv_path, header=None)
    n_cols = df_raw.shape[1]
    has_quality = n_cols >= 5

    if n_cols == 4:
        df_raw.columns = ["ID", "X", "Y", "FRAME"]
    elif n_cols >= 5:
        df_raw = df_raw.iloc[:, :5]
        df_raw.columns = ["ID", "X", "Y", "FRAME", "QUALITY"]
    else:
        return None

    df_raw = df_raw.sort_values(by=["ID", "FRAME"]).reset_index(drop=True)

    first_pos = df_raw.groupby("ID").first()[["X", "Y"]].rename(
        columns={"X": "X_first", "Y": "Y_first"}
    )

    df_corr = df[["ID", "FRAME", "X_diff_corr", "Y_diff_corr"]].copy()
    df_merged = df_raw.merge(df_corr, on=["ID", "FRAME"], how="left")
    df_merged = df_merged.merge(first_pos, on="ID", how="left")

    X_mask = df_merged["X_diff_corr"].isna()
    df_merged["X"] = np.where(X_mask, df_merged["X"],
        df_merged["X_first"] + df_merged["X_diff_corr"] / pixel_size)
    Y_mask = df_merged["Y_diff_corr"].isna()
    df_merged["Y"] = np.where(Y_mask, df_merged["Y"],
        df_merged["Y_first"] + df_merged["Y_diff_corr"] / pixel_size)

    export_cols = ["ID", "X", "Y", "FRAME"]
    if has_quality and "QUALITY" in df_merged.columns:
        export_cols.append("QUALITY")

    df_merged[export_cols].to_csv(output_path, index=False, header=False)
    return len(df_merged)
