from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from opennta.analysis.data_preprocessor import DataPreprocessor


def test_loads_5col_spots_csv(analysis_config, synthetic_spots_csv):
    pre = DataPreprocessor(analysis_config)
    df, n_ids = pre.load_and_compute_displacements(str(synthetic_spots_csv))

    assert n_ids > 0
    for col in ("ID", "X", "Y", "FRAME", "QUALITY",
                "dX", "dY", "dX_um", "dY_um", "X_diff", "Y_diff"):
        assert col in df.columns
    assert not df.isna().any().any()
    assert n_ids == df["ID"].nunique()


def test_loads_4col_csv(tmp_path, analysis_config):
    csv = tmp_path / "synthetic_4col.csv"
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(1, 6):
        for frame in range(10):
            rows.append([pid, rng.normal(100, 1), rng.normal(100, 1), frame])
    pd.DataFrame(rows).to_csv(csv, header=False, index=False)

    pre = DataPreprocessor(analysis_config)
    df, n_ids = pre.load_and_compute_displacements(str(csv))

    assert n_ids == 5
    assert "QUALITY" not in df.columns
    assert "X_diff" in df.columns


def test_rejects_unsupported_column_count(tmp_path, analysis_config):
    csv = tmp_path / "bad.csv"
    pd.DataFrame([[1, 2, 3]]).to_csv(csv, header=False, index=False)

    with pytest.raises(ValueError, match="Unsupported CSV format"):
        DataPreprocessor(analysis_config).load_and_compute_displacements(str(csv))


def test_dx_um_uses_pixel_size(analysis_config, tmp_path):
    df = pd.DataFrame({
        "ID": [1, 1, 1],
        "X": [0.0, 1.0, 2.0],
        "Y": [0.0, 0.0, 0.0],
        "FRAME": [0, 1, 2],
        "QUALITY": [1.0, 1.0, 1.0],
    })
    csv = tmp_path / "unit_step.csv"
    df.to_csv(csv, header=False, index=False)

    pre = DataPreprocessor(analysis_config)
    out, _ = pre.load_and_compute_displacements(str(csv))
    expected_step = analysis_config.pixel_size
    assert np.allclose(out["dX_um"].to_numpy(), expected_step)


def test_loads_5col_csv_with_uppercase_header(tmp_path, analysis_config):
    """Headered CSV with the canonical 5-column names is parsed via the
    header path (not the headerless fallback)."""
    csv = tmp_path / "headered.csv"
    rows = []
    for pid in range(1, 4):
        for frame in range(8):
            rows.append([pid, 100.0 + frame, 50.0 - frame, frame, 7.5])
    pd.DataFrame(rows, columns=["ID", "X", "Y", "FRAME", "QUALITY"]).to_csv(
        csv, index=False
    )

    df, n_ids = DataPreprocessor(analysis_config).load_and_compute_displacements(str(csv))
    assert n_ids == 3
    assert set(["QUALITY", "dX_um", "dY_um"]).issubset(df.columns)
    # X step of +1 px per frame -> dX_um == pixel_size
    assert np.allclose(df["dX_um"].to_numpy(), analysis_config.pixel_size)


def test_truncates_extra_columns_to_first_five(tmp_path, analysis_config):
    """A header-less CSV with >5 columns is truncated to the first 5."""
    csv = tmp_path / "wide.csv"
    rows = []
    for pid in range(1, 3):
        for frame in range(6):
            rows.append([pid, float(frame), float(frame), frame, 1.0, "extra", 9.0])
    pd.DataFrame(rows).to_csv(csv, header=False, index=False)

    df, n_ids = DataPreprocessor(analysis_config).load_and_compute_displacements(str(csv))
    assert n_ids == 2
    assert "QUALITY" in df.columns
    # 7th column "extra" must not have leaked into the frame.
    assert df.shape[1] <= 11
