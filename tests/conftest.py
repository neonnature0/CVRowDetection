"""Shared test fixtures for vineyard row detection tests."""

import sys
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))