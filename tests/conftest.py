"""Shared test fixtures for vineyard row detection tests."""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vinerow.loaders.json_loader import load_test_blocks


@pytest.fixture
def test_blocks():
    """Load test blocks from test_blocks.json."""
    blocks = load_test_blocks(PROJECT_ROOT / "data" / "blocks" / "test_blocks.json")
    if not blocks:
        pytest.skip("test_blocks.json not found or empty")
    return blocks


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
