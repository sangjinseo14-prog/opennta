"""Unit tests for ``opennta.tracking.tracking_processor``.

The orchestrator is exercised end-to-end with a stub ``TrackingMethod``
injected through the ``get_tracking_method`` lookup, so the batch loop,
nta_folder grouping, save-path bootstrap, success/failure aggregation
and exception handling can be verified without Fiji or any I/O beyond
``tmp_path``. The Fiji-bound TrackMate plugin is covered separately: its
Fiji-independent helpers by the ``unit/tracking/trackmate/`` unit tests,
and the full Fiji + TrackMate run by ``integration/test_trackmate_tracking.py``
(which skips when no Fiji binary is configured).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from opennta.tracking import tracking_processor as tp_mod
from opennta.tracking.status_codes import TrackingStatus
from opennta.tracking.tracking_method import TrackingMethod
from opennta.tracking.tracking_processor import TrackingProcessor
from opennta.tracking.types import FolderItem


class _RecordingMethod(TrackingMethod):
    """In-memory tracking-method stub that records every track_subfolder
    call and reports success/failure from a pre-baked list."""

    mode_key = "_recording"
    ui_label = "Recording Stub"

    def __init__(
        self,
        params,
        emitter,
        save_path=None,
        fiji_path=None,
        outcomes: Sequence[bool] | None = None,
    ):
        super().__init__(params, emitter, save_path, fiji_path)
        self._outcomes = list(outcomes or [])
        self.calls: list[tuple[str, tuple[str, ...], str, str]] = []

    def track_subfolder(self, sub_path, tiff_paths, nta_folder, sub) -> bool:
        self.calls.append((sub_path, tuple(tiff_paths), nta_folder, sub))
        if not self._outcomes:
            return True
        return self._outcomes.pop(0)

    def get_parameter_log_lines(self) -> list[str]:
        return ["[ Recording Stub ]", "  marker: true", ""]


@pytest.fixture(autouse=True)
def _noop_sleep(monkeypatch):
    """``TrackingProcessor.run_tracking_batch`` sleeps 0.1s several times
    per item to flush log output; in tests we skip the wall-clock cost."""
    monkeypatch.setattr(tp_mod.time, "sleep", lambda _s: None)


def _make_items(tmp_path: Path) -> list[FolderItem]:
    """Three items spread over two nta_folders ('A' x2, 'B' x1)."""
    items: list[FolderItem] = []
    for nta_folder, subs in (("A", ("s1", "s2")), ("B", ("s1",))):
        for sub in subs:
            sub_dir = tmp_path / nta_folder / sub
            sub_dir.mkdir(parents=True, exist_ok=True)
            tiff = sub_dir / "in.tif"
            tiff.write_bytes(b"")
            items.append((str(sub_dir), [str(tiff)], nta_folder, sub))
    return items


def _install_recording_method(monkeypatch, outcomes: Sequence[bool] | None = None):
    holder: dict[str, _RecordingMethod] = {}

    def _factory(mode, params, emitter, save_path=None, fiji_path=None):
        method = _RecordingMethod(
            params, emitter, save_path=save_path, fiji_path=fiji_path,
            outcomes=outcomes,
        )
        holder["instance"] = method
        return method

    monkeypatch.setattr(tp_mod, "get_tracking_method", _factory)
    return holder


def test_batch_iterates_every_item_grouped_by_nta_folder(monkeypatch, tmp_path):
    items = _make_items(tmp_path)
    holder = _install_recording_method(monkeypatch)

    processor = TrackingProcessor(
        folder_items=items, fiji_path="/fake/fiji",
        save_path=str(tmp_path / "out"), root_folder=str(tmp_path),
    )
    Path(processor.save_path).mkdir(parents=True, exist_ok=True)
    assert processor.run_tracking_batch() is True

    method = holder["instance"]
    assert len(method.calls) == len(items)
    # nta_folder grouping must be preserved: all 'A' items appear before 'B'.
    seen_groups = [call[2] for call in method.calls]
    assert seen_groups == ["A", "A", "B"]


def test_batch_returns_false_when_any_item_fails(monkeypatch, tmp_path):
    items = _make_items(tmp_path)
    holder = _install_recording_method(monkeypatch, outcomes=[True, False, True])

    processor = TrackingProcessor(
        folder_items=items, fiji_path="/fake/fiji",
        save_path=str(tmp_path / "out"), root_folder=str(tmp_path),
    )
    Path(processor.save_path).mkdir(parents=True, exist_ok=True)
    assert processor.run_tracking_batch() is False
    # The remaining items are still processed — failure aggregation is at
    # the batch level, not a short-circuit.
    assert len(holder["instance"].calls) == 3


def test_batch_catches_method_construction_errors(monkeypatch, tmp_path):
    """If ``get_tracking_method`` raises, the batch must surface the error
    via the emitter and return False rather than propagating."""
    items = _make_items(tmp_path)
    log_seen: list[str] = []
    progress_seen: list[int] = []

    def _boom(*a, **kw):
        raise RuntimeError("plugin import failed")

    monkeypatch.setattr(tp_mod, "get_tracking_method", _boom)

    processor = TrackingProcessor(
        folder_items=items, fiji_path="/fake/fiji",
        save_path=str(tmp_path / "out"), root_folder=str(tmp_path),
        progress_callback=progress_seen.append, log_callback=log_seen.append,
    )
    Path(processor.save_path).mkdir(parents=True, exist_ok=True)
    assert processor.run_tracking_batch() is False
    assert any("plugin import failed" in m for m in log_seen)
    assert TrackingStatus.ERROR_GENERAL in progress_seen


def test_get_or_create_save_path_creates_timestamped_dir_when_missing(
    monkeypatch, tmp_path,
):
    """Without a save_path the processor must materialise a fresh
    ``OpenNTA_Results_*`` directory under ``root_folder``."""
    items = _make_items(tmp_path)
    _install_recording_method(monkeypatch)

    processor = TrackingProcessor(
        folder_items=items, fiji_path="/fake/fiji",
        save_path=None, root_folder=str(tmp_path),
    )
    out = processor._get_or_create_save_path()
    assert Path(out).is_dir()
    assert Path(out).parent == tmp_path
    assert Path(out).name.startswith("OpenNTA_Results_")


def test_get_or_create_save_path_reuses_existing(monkeypatch, tmp_path):
    existing = tmp_path / "pre-existing"
    existing.mkdir()
    _install_recording_method(monkeypatch)
    processor = TrackingProcessor(
        folder_items=[], fiji_path="/fake/fiji",
        save_path=str(existing), root_folder=str(tmp_path),
    )
    assert processor._get_or_create_save_path() == str(existing)


def test_emit_parameter_summary_includes_method_and_paths(monkeypatch, tmp_path):
    """The parameter summary must reference the method display name and
    the user-visible fiji/save paths so the saved log is self-contained."""
    items = _make_items(tmp_path)
    log_seen: list[str] = []
    _install_recording_method(monkeypatch)

    save_dir = tmp_path / "out"
    save_dir.mkdir()
    processor = TrackingProcessor(
        folder_items=items, fiji_path="/fake/fiji-app",
        save_path=str(save_dir), root_folder=str(tmp_path),
        log_callback=log_seen.append,
    )
    assert processor.run_tracking_batch() is True
    joined = "\n".join(log_seen)
    assert "Recording Stub" in joined
    assert "/fake/fiji-app" in joined
    assert str(save_dir) in joined
    # Method-supplied parameter lines must be forwarded into the summary.
    assert "marker: true" in joined


def test_batch_emits_lifecycle_status_codes(monkeypatch, tmp_path):
    items = _make_items(tmp_path)
    progress_seen: list[int] = []
    _install_recording_method(monkeypatch)

    processor = TrackingProcessor(
        folder_items=items, fiji_path="/fake/fiji",
        save_path=str(tmp_path / "out"), root_folder=str(tmp_path),
        progress_callback=progress_seen.append,
    )
    Path(processor.save_path).mkdir(parents=True, exist_ok=True)
    assert processor.run_tracking_batch() is True
    # STARTED is emitted for the batch, each group, and each item.
    # ALL_COMPLETE is emitted exactly once at the end.
    assert progress_seen.count(TrackingStatus.ALL_COMPLETE) == 1
    assert TrackingStatus.STARTED in progress_seen
    assert TrackingStatus.ITEM_COMPLETE in progress_seen


def test_batch_with_empty_folder_items_still_emits_all_complete(monkeypatch, tmp_path):
    progress_seen: list[int] = []
    _install_recording_method(monkeypatch)
    processor = TrackingProcessor(
        folder_items=[], fiji_path="/fake/fiji",
        save_path=str(tmp_path / "out"), root_folder=str(tmp_path),
        progress_callback=progress_seen.append,
    )
    Path(processor.save_path).mkdir(parents=True, exist_ok=True)
    assert processor.run_tracking_batch() is True
    assert TrackingStatus.ALL_COMPLETE in progress_seen
