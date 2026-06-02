from __future__ import annotations

import pandas as pd

from .types import AnalysisConfig


class DataPreprocessor:

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def load_and_compute_displacements(self, csv_path: str) -> tuple[pd.DataFrame, int]:

        df_header = pd.read_csv(csv_path)
        cols_upper = {str(col).strip().upper(): col for col in df_header.columns}
        required = ["ID", "X", "Y", "FRAME"]
        has_required_header = all(col in cols_upper for col in required)

        if has_required_header:
            keep_cols = [cols_upper["ID"], cols_upper["X"], cols_upper["Y"], cols_upper["FRAME"]]
            if "QUALITY" in cols_upper:
                keep_cols.append(cols_upper["QUALITY"])
            df_raw = df_header.loc[:, keep_cols].copy()
            rename_map = {v: k for k, v in zip(["ID", "X", "Y", "FRAME"], keep_cols[:4], strict=False)}
            if len(keep_cols) == 5:
                rename_map[keep_cols[4]] = "QUALITY"
            df_raw = df_raw.rename(columns=rename_map)
        else:
            # Backward compatibility with header-less legacy TrackMate CSVs.
            df_raw = pd.read_csv(csv_path, header=None)
            n_cols = df_raw.shape[1]
            if n_cols == 4:
                df_raw.columns = ["ID", "X", "Y", "FRAME"]
            elif n_cols >= 5:
                df_raw = df_raw.iloc[:, :5]
                df_raw.columns = ["ID", "X", "Y", "FRAME", "QUALITY"]
            else:
                raise ValueError(f"Unsupported CSV format: {n_cols} columns (expected 4 or 5).")

        df = df_raw.sort_values(by=["ID", "FRAME"])

        df[["dX", "dY"]] = df.groupby("ID")[["X", "Y"]].diff()
        df = df.dropna()

        df["dX_um"] = df["dX"] * self.config.pixel_size
        df["dY_um"] = df["dY"] * self.config.pixel_size

        df[["X_diff", "Y_diff"]] = df.groupby("ID")[["dX_um", "dY_um"]].cumsum()

        n_ids = df["ID"].nunique()

        return df, n_ids
