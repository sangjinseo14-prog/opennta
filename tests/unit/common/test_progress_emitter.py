"""Tests for the shared ProgressEmitter (status code + log forwarding)."""
from __future__ import annotations

from pathlib import Path

from opennta.common.progress import ProgressEmitter


def test_emit_forwards_progress_and_log_callbacks():
    progress_seen: list[int] = []
    log_seen: list[str] = []

    em = ProgressEmitter(
        progress_callback=progress_seen.append,
        log_callback=log_seen.append,
    )
    em.emit(code=10, msg="started")
    em.emit(code=50)              # no msg → no log
    em.emit(msg="midway")          # no code → no progress event

    assert progress_seen == [10, 50]
    assert log_seen == ["started", "midway"]


def test_emit_buffers_messages_for_save():
    em = ProgressEmitter()
    em.emit(msg="a")
    em.emit(msg="b")
    assert em._log_buffer == ["a", "b"]


def test_emit_no_args_is_noop():
    em = ProgressEmitter()
    em.emit()
    assert em._log_buffer == []


def test_save_log_writes_buffer_to_directory(tmp_path: Path):
    em = ProgressEmitter(save_path=str(tmp_path))
    em.emit(msg="line one")
    em.emit(msg="line two")
    em.save_log()

    logs = list(tmp_path.glob("trackmate_log_*.txt"))
    assert len(logs) == 1
    assert logs[0].read_text(encoding="utf-8") == "line one\nline two"


def test_save_log_empty_buffer_does_nothing(tmp_path: Path):
    em = ProgressEmitter(save_path=str(tmp_path))
    em.save_log()
    assert list(tmp_path.glob("trackmate_log_*.txt")) == []


def test_save_log_reuses_filename_across_calls(tmp_path: Path):
    """Calling save_log twice should overwrite the same file rather than
    creating a second one (filename is cached after the first call)."""
    em = ProgressEmitter(save_path=str(tmp_path))
    em.emit(msg="round one")
    em.save_log()
    em.emit(msg="round two")
    em.save_log()
    logs = list(tmp_path.glob("trackmate_log_*.txt"))
    assert len(logs) == 1
    assert "round one" in logs[0].read_text(encoding="utf-8")
    assert "round two" in logs[0].read_text(encoding="utf-8")


def test_save_log_creates_missing_directory(tmp_path: Path):
    save_dir = tmp_path / "nested" / "result"
    em = ProgressEmitter(save_path=str(save_dir))
    em.emit(msg="hi")
    em.save_log()
    assert save_dir.is_dir()
    assert list(save_dir.glob("trackmate_log_*.txt"))


def test_save_log_io_error_is_recorded_in_buffer(monkeypatch, tmp_path: Path):
    """If writing the log file fails, save_log appends a record rather than
    raising (so a save failure can't mask a successful pipeline)."""
    em = ProgressEmitter(save_path=str(tmp_path))
    em.emit(msg="payload")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", _boom)
    em.save_log()  # must not raise
    assert any("Failed to write log" in m for m in em._log_buffer)
