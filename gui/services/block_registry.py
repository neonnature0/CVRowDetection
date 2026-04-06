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

        # Auto-detect region from boundary
        region_info = _detect_block_region(boundary)

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
            "difficulty_rating": None,
            "region": region_info["region"],
            "region_auto_detected": True,
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


def backfill_regions() -> list[dict]:
    """Backfill region field for all blocks that don't have one.

    Called on first startup after the region feature lands.
    Returns list of dicts: [{"name": ..., "region": ..., "distance_km": ..., "confidence": ...}]
    """
    assignments = []
    with _lock:
        data = _read_raw()
        blocks = data.get("blocks", [])
        changed = False

        for b in blocks:
            if b.get("region") is not None:
                continue
            info = _detect_block_region(b.get("boundary", {}))
            b["region"] = info["region"]
            b["region_auto_detected"] = True
            changed = True
            assignments.append({
                "name": b["name"],
                "region": info["region"],
                "distance_km": info["distance_km"],
                "confidence": info["confidence"],
            })

        if changed:
            data["blocks"] = blocks
            _write_raw(data)

    for a in assignments:
        logger.info(
            "Backfill: %s → %s (%.1f km, %s)",
            a["name"], a["region"], a["distance_km"], a["confidence"],
        )

    return assignments


def _detect_block_region(boundary: dict) -> dict:
    """Detect region for a boundary, with fallback on error."""
    try:
        from blocks.region_detection import detect_region
        return detect_region(boundary)
    except Exception as e:
        logger.warning("Region detection failed: %s", e)
        return {"region": None, "distance_km": 0.0, "confidence": "low"}


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
