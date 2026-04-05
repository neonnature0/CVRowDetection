"""Run the row detection pipeline on a block."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from vinerow.acquisition.geo_utils import meters_per_pixel, polygon_bbox
from vinerow.acquisition.tile_fetcher import (
    TILE_SOURCES,
    auto_select_source,
    default_zoom,
    fetch_and_stitch,
)
from vinerow.config import PipelineConfig
from vinerow.pipeline import run_pipeline
from vinerow.types import BlockRowDetectionResult

logger = logging.getLogger(__name__)


def detect_block(block: dict, config: PipelineConfig | None = None):
    """Fetch tiles and run the pipeline on a block.

    Returns:
        Tuple of (image_bgr, mask, result, mpp) or None if detection fails.
    """
    if config is None:
        config = PipelineConfig()

    boundary = block["boundary"]
    coords = boundary["coordinates"][0]
    name = block.get("name", "unknown")

    bbox = polygon_bbox(coords)
    centroid_lng = (bbox[0] + bbox[2]) / 2.0
    centroid_lat = (bbox[1] + bbox[3]) / 2.0
    source_name = auto_select_source(centroid_lng)
    source = TILE_SOURCES[source_name]
    zoom = default_zoom(source_name)

    logger.info("Fetching tiles for %s (source=%s, zoom=%d)", name, source_name, zoom)
    t0 = time.perf_counter()
    image_bgr, mask, tile_origin = fetch_and_stitch(
        source, coords, zoom, source_name, cache_dir=config.tile_cache_dir,
    )
    fetch_time = time.perf_counter() - t0
    logger.info("Tiles fetched in %.1fs (%dx%d)", fetch_time, image_bgr.shape[1], image_bgr.shape[0])

    mpp = meters_per_pixel(centroid_lat, zoom, source.tile_size)

    result = run_pipeline(
        image_bgr=image_bgr,
        mask=mask,
        mpp=mpp,
        lat=centroid_lat,
        zoom=zoom,
        tile_size=source.tile_size,
        tile_origin=tile_origin,
        tile_source=source_name,
        config=config,
        block_name=name,
    )

    if result is None:
        logger.error("Detection failed for %s", name)
        return None

    return image_bgr, mask, result, mpp


def generate_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    result: BlockRowDetectionResult,
    mpp: float,
    block_name: str = "",
) -> np.ndarray:
    """Draw detected rows on the aerial image. Returns BGR overlay."""
    from visual_verify import draw_row_overlay
    return draw_row_overlay(
        image_bgr, result, mask, mpp,
        block_name=block_name, vineyard="",
        gt_spacing=None, gt_rows=None,
    )


def generate_thumbnail(image_bgr: np.ndarray, max_size: int = 256) -> np.ndarray:
    """Downscale an image to thumbnail size."""
    h, w = image_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_bgr
