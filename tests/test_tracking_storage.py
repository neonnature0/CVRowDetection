"""Tests for tracking/storage.py — corruption handling, append, load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tracking.storage import (
    TrackingStorageCorrupted,
    _read_json_array,
    _write_json_array,
    append_run,
    append_block_results,
    load_runs,
    load_block_results,
    generate_run_id,
    RUNS_FILE,
    BLOCK_RESULTS_FILE,
)


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Redirect storage files to temp directory for test isolation."""
    monkeypatch.setattr("tracking.storage.RUNS_FILE", tmp_path / "runs.json")
    monkeypatch.setattr("tracking.storage.BLOCK_RESULTS_FILE", tmp_path / "per_block_results.json")
    yield tmp_path


class TestReadJsonArray:
    def test_missing_file_returns_empty(self, tmp_path):
        result = _read_json_array(tmp_path / "nonexistent.json")
        assert result == []

    def test_valid_array(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('[{"a": 1}, {"b": 2}]')
        result = _read_json_array(path)
        assert len(result) == 2
        assert result[0]["a"] == 1

    def test_empty_array(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text("[]")
        result = _read_json_array(path)
        assert result == []

    def test_corrupted_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json!!!")

        with pytest.raises(TrackingStorageCorrupted, match="Corrupted tracking file"):
            _read_json_array(path)

        # Verify backup was created
        backups = list(tmp_path.glob("bad.json.corrupted.*"))
        assert len(backups) == 1

    def test_non_array_json_raises(self, tmp_path):
        path = tmp_path / "obj.json"
        path.write_text('{"not": "an array"}')

        with pytest.raises(TrackingStorageCorrupted, match="expected array"):
            _read_json_array(path)


class TestAppendAndLoad:
    def test_append_run_and_load(self):
        record = {"run_id": "test-001", "timestamp": "2026-01-01T00:00:00Z", "run_type": "evaluation"}
        append_run(record)

        runs = load_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "test-001"

    def test_append_multiple_runs(self):
        for i in range(3):
            append_run({"run_id": f"test-{i:03d}", "timestamp": f"2026-01-0{i+1}T00:00:00Z"})

        runs = load_runs()
        assert len(runs) == 3
        # Newest first
        assert runs[0]["run_id"] == "test-002"

    def test_append_block_results(self):
        records = [
            {"run_id": "r1", "block_id": "a", "f1_04": 0.9},
            {"run_id": "r1", "block_id": "b", "f1_04": 0.8},
        ]
        append_block_results(records)

        results = load_block_results()
        assert len(results) == 2

    def test_append_empty_list_is_noop(self):
        append_block_results([])
        results = load_block_results()
        assert results == []

    def test_corrupted_file_blocks_append(self, tmp_path):
        """Appending to a corrupted file raises, not silently overwrites."""
        runs_path = tmp_path / "runs.json"
        runs_path.write_text("CORRUPTED!!!")

        with pytest.raises(TrackingStorageCorrupted):
            append_run({"run_id": "should-fail"})


class TestGenerateRunId:
    def test_format(self):
        rid = generate_run_id()
        assert isinstance(rid, str)
        assert len(rid) > 20  # timestamp + random + hash
        # URL-safe: no colons, slashes, spaces
        assert ":" not in rid
        assert "/" not in rid
        assert " " not in rid

    def test_uniqueness(self):
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100  # all unique (random suffix ensures this)
