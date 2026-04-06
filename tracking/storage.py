"""Read/write tracking data (runs + per-block results) with thread safety.

Storage files:
  tracking/runs.json             — array of run records (appended over time)
  tracking/per_block_results.json — array of per-block evaluation records

Both files are JSON arrays of objects. Writes use a threading lock
(same pattern as gui/services/block_registry.py).
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

TRACKING_DIR = Path(__file__).resolve().parent
RUNS_FILE = TRACKING_DIR / "runs.json"
BLOCK_RESULTS_FILE = TRACKING_DIR / "per_block_results.json"

_lock = threading.Lock()
_run_id_lock = threading.Lock()
_last_run_ts_ms = ""
_last_run_seq = -1


# ── Run ID generation ──


def generate_run_id() -> str:
    """Generate a run ID: {ISO timestamp-ms}_{7-char git hash}[-dirty]_s{seq}.

    Example:
      2026-04-06T14-23-11-372_a3f2c1_s0000
      2026-04-06T14-23-11-372_a3f2c1_s0001
      2026-04-06T14-23-11-372_a3f2c1-dirty_s0002
    """
    ts, seq = _next_run_ts_and_seq()
    commit = _git_short_hash()
    dirty = _git_is_dirty()
    suffix = f"{commit}-dirty" if dirty else commit
    return f"{ts}_{suffix}_s{seq:04d}"


def _next_run_ts_and_seq() -> tuple[str, int]:
    """Return millisecond UTC timestamp plus per-ms sequence number."""
    global _last_run_ts_ms, _last_run_seq

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]
    with _run_id_lock:
        if ts == _last_run_ts_ms:
            _last_run_seq += 1
        else:
            _last_run_ts_ms = ts
            _last_run_seq = 0
        return ts, _last_run_seq


def get_git_info() -> tuple[str, bool]:
    """Return (7-char commit hash, is_dirty)."""
    return _git_short_hash(), _git_is_dirty()


def _git_short_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=TRACKING_DIR,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_is_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, timeout=5,
            cwd=TRACKING_DIR,
        )
        return result.returncode != 0
    except Exception:
        return True  # Assume dirty if we can't check


# ── Low-level JSON I/O ──


def _read_json_array(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("Expected JSON array in %s, got %s", path, type(data).__name__)
        return []
    return data


def _write_json_array(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Public API ──


def append_run(record: dict) -> None:
    """Append a run record to runs.json."""
    with _lock:
        runs = _read_json_array(RUNS_FILE)
        runs.append(record)
        _write_json_array(RUNS_FILE, runs)
    logger.info("Appended run %s (type=%s)", record.get("run_id"), record.get("run_type"))


def append_block_results(records: list[dict]) -> None:
    """Append one or more per-block result records to per_block_results.json."""
    if not records:
        return
    with _lock:
        existing = _read_json_array(BLOCK_RESULTS_FILE)
        existing.extend(records)
        _write_json_array(BLOCK_RESULTS_FILE, existing)
    logger.info("Appended %d block results for run %s", len(records), records[0].get("run_id"))


def load_runs() -> list[dict]:
    """Load all run records, newest first."""
    with _lock:
        runs = _read_json_array(RUNS_FILE)
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return runs


def load_block_results() -> list[dict]:
    """Load all per-block result records."""
    with _lock:
        return _read_json_array(BLOCK_RESULTS_FILE)


def get_run(run_id: str) -> dict | None:
    """Load a single run by ID."""
    for r in load_runs():
        if r.get("run_id") == run_id:
            return r
    return None
