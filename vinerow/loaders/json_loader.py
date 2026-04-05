"""Load block definitions from a JSON file (test_blocks.json format)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = "data/blocks/test_blocks.json"


class JsonLoader:
    """Load blocks from a JSON file with ``{"blocks": [...]}`` schema."""

    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            logger.error("Test blocks file not found: %s", self.path)
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        blocks = data.get("blocks", [])
        logger.info("Loaded %d blocks from %s", len(blocks), self.path)
        return blocks


def load_test_blocks(path: str = DEFAULT_PATH) -> list[dict]:
    """Convenience function matching the legacy API."""
    return JsonLoader(path).load()
