"""Shared data types for the detection system."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BlockDetection:
    """A single detected vineyard block."""

    block_id: int
    polygon_px: list[tuple[float, float]]  # pixel coordinates in stitched image
    polygon_lnglat: list[tuple[float, float]] | None = None  # WGS84 if converted
    area_px: float = 0.0
    confidence: float = 0.0


@dataclass
class BlockDetectionResult:
    """Complete result from block detection on a property image."""

    blocks: list[BlockDetection] = field(default_factory=list)
    interior_mask: np.ndarray | None = None  # (H, W) float32 probability map
    boundary_mask: np.ndarray | None = None  # (H, W) float32 probability map
    image_size: tuple[int, int] = (0, 0)  # (height, width)
    meters_per_pixel: float = 0.0
    processing_time_s: float = 0.0
