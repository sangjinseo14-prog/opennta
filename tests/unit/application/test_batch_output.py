"""Smoke tests for ``opennta.application.tab_batch_analysis.output``.

``TabBatchAnalysisOutput`` is functionally Qt-free (its only transitive Qt import,
``ReportTemplates``, only uses ``QTreeWidgetItem`` as a TypedDict
annotation, satisfied by the conftest PyQt5 stub). These tests cover the
two filesystem-touching helpers that drive the Batch tab output layout.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from opennta.application.tab_batch_analysis.output import TabBatchAnalysisOutput


def test_create_output_directory_produces_unique_sibling_dirs(tmp_path):
    first = TabBatchAnalysisOutput(str(tmp_path))
    second = TabBatchAnalysisOutput(str(tmp_path))

    first_path = first.create_output_directory()
    second_path = second.create_output_directory()

    assert first_path != second_path
    for path in (first_path, second_path):
        assert (tmp_path / path).exists()
        assert (tmp_path / path / "individual_results").is_dir()
        assert (tmp_path / path / "merged_results").is_dir()


def test_save_individual_result_writes_diameter_csv(tmp_path):
    output = TabBatchAnalysisOutput(str(tmp_path))
    output.create_output_directory()

    result = {
        "folder": "sample_A",
        "file": "movie01_spots.csv",
        "diameters": [80.0, 95.0, 110.0],
        # ``stats`` is None so the HTML report branch is skipped — the
        # csv-writing path is the one this smoke test covers.
        "stats": None,
        "path": "/tmp/movie01.tif",
    }

    csv_path = output.save_individual_result(result)

    assert csv_path.endswith("movie01_diameter.csv")
    with open(csv_path, encoding="utf-8") as f:
        rows = [row[0] for row in csv.reader(f) if row]
    assert [float(r) for r in rows] == result["diameters"]


def test_calculate_merged_statistics_returns_diameter_section():
    diameters = [50.0, 75.0, 100.0, 125.0, 150.0]

    stats = TabBatchAnalysisOutput(base_path="/unused")._calculate_merged_statistics(diameters)

    assert "diameter" in stats
    diameter = stats["diameter"]
    # Pin the actual moments, not just ">0": mean/median are 100, std matches
    # numpy's population std, so a regression in the aggregation surfaces.
    assert diameter["mean"] == pytest.approx(100.0)
    assert diameter["median"] == pytest.approx(100.0)
    assert diameter["std"] == pytest.approx(float(np.std(diameters)))
    assert diameter["unit"] == "nm"
