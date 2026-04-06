"""GUI configuration."""

from __future__ import annotations

from pathlib import Path

# Project root (CVRowDetection/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
BLOCKS_FILE = PROJECT_ROOT / "data" / "blocks" / "test_blocks.json"
BLOCKS_LOCAL_FILE = PROJECT_ROOT / "data" / "blocks" / "test_blocks.local.json"
ANNOTATIONS_DIR = PROJECT_ROOT / "dataset" / "annotations"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
DETECTIONS_DIR = PROJECT_ROOT / "output" / "detections"
TILE_CACHE_DIR = PROJECT_ROOT / "output" / ".tile_cache"
THUMBNAILS_DIR = PROJECT_ROOT / "output" / "thumbnails"
TRAINING_PROGRESS_FILE = PROJECT_ROOT / "output" / "training_progress.json"

# Server
HOST = "127.0.0.1"
PORT = 8765

# Default map center: Marlborough, NZ
DEFAULT_CENTER_LNG = 173.95
DEFAULT_CENTER_LAT = -41.51
DEFAULT_ZOOM = 14
