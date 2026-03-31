"""Shared test fixtures for vineyard row detection tests."""

import json
import sys
from pathlib import Path

import pytest

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def test_blocks():
    """Load test blocks from test_blocks.json."""
    path = PROJECT_ROOT / "test_blocks.json"
    if not path.exists():
        pytest.skip("test_blocks.json not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("blocks", [])


@pytest.fixture
def block_c(test_blocks):
    """Block C from Brooklands — well-tested reference block."""
    for b in test_blocks:
        if b["name"] == "Block C":
            return b
    pytest.skip("Block C not found in test blocks")


@pytest.fixture
def default_config():
    """Default pipeline configuration."""
    from vinerow.config import PipelineConfig
    return PipelineConfig()
