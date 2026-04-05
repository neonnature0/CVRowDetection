"""Read/write block registry (test_blocks.json) with thread safety."""

from __future__ import annotations

import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from gui.config import BLOCKS_FILE

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _read_raw() -> dict:
    if not BLOCKS_FILE.exists():
        return {"blocks": []}
    with open(BLOCKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_raw(data: dict) -> None:
    BLOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BLOCKS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _generate_hex_name(existing_names: set[str]) -> str:
    """Generate a unique 6-hex-digit name."""
    for _ in range(100):
        name = secrets.token_hex(3)  # 6 hex chars
        if name not in existing_names:
            return name
    raise RuntimeError("Failed to generate unique block name")


def list_blocks() -> list[dict]:
    """Return all blocks."""
    with _lock:
        data = _read_raw()
    return data.get("blocks", [])


def get_block(name: str) -> dict | None:
    """Get a single block by name."""
    for b in list_blocks():
        if b["name"] == name:
            return b
    return None


def create_block(boundary: dict) -> dict:
    """Create a new block with a random hex name. Returns the new block dict."""
    with _lock:
        data = _read_raw()
        blocks = data.get("blocks", [])
        existing = {b["name"] for b in blocks}
        name = _generate_hex_name(existing)

        block = {
            "name": name,
            "vineyard_name": "",
            "boundary": boundary,
            "row_spacing_m": None,
            "row_count": None,
            "row_angle": None,
            "row_orientation": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "gui",
            "stage": "draft",
            "last_detection_at": None,
            "thumbnail_path": None,
        }
        blocks.append(block)
        data["blocks"] = blocks
        _write_raw(data)

    logger.info("Created block %s", name)
    return block


def delete_block(name: str) -> bool:
    """Delete a block by name. Returns True if found and deleted."""
    with _lock:
        data = _read_raw()
        blocks = data.get("blocks", [])
        original_len = len(blocks)
        blocks = [b for b in blocks if b["name"] != name]
        if len(blocks) == original_len:
            return False
        data["blocks"] = blocks
        _write_raw(data)

    logger.info("Deleted block %s", name)
    return True


def update_block(name: str, updates: dict) -> dict | None:
    """Update fields on a block. Returns updated block or None if not found."""
    with _lock:
        data = _read_raw()
        blocks = data.get("blocks", [])
        for b in blocks:
            if b["name"] == name:
                b.update(updates)
                data["blocks"] = blocks
                _write_raw(data)
                return b
    return None
