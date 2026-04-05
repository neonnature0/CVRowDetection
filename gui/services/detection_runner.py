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


def compute_ensemble_confidence(
    block: dict,
    primary_result,
    primary_config: PipelineConfig,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    mpp: float,
):
    """Run a shadow detection with the alternative ridge mode and score agreement.

    For each primary row, assigns ensemble_confidence based on whether a
    matching row exists in the shadow detection:
      - Close match (< 0.2 × spacing): 1.0
      - Weak match (0.2-0.4 × spacing): 0.6
      - No match: 0.2
    """
    import math
    from scipy.optimize import linear_sum_assignment

    # Determine shadow mode
    primary_mode = primary_config.ridge_mode
    shadow_mode = "gabor" if primary_mode in ("ml", "ml_ensemble") else "ml"

    shadow_config = PipelineConfig(
        ridge_mode=shadow_mode,
        save_debug_artifacts=False,
    )

    # Get tile info from primary config
    from vinerow.acquisition.geo_utils import polygon_bbox
    from vinerow.acquisition.tile_fetcher import auto_select_source, default_zoom, TILE_SOURCES

    coords = block["boundary"]["coordinates"][0]
    bbox = polygon_bbox(coords)
    lat = (bbox[1] + bbox[3]) / 2.0
    lng = (bbox[0] + bbox[2]) / 2.0
    source_name = auto_select_source(lng)
    zoom = default_zoom(source_name)

    logger.info("Running shadow detection (mode=%s) for ensemble confidence", shadow_mode)
    shadow_result = run_pipeline(
        image_bgr=image_bgr,
        mask=mask,
        mpp=mpp,
        lat=lat,
        zoom=zoom,
        tile_size=TILE_SOURCES[source_name].tile_size,
        tile_origin=(0, 0),
        tile_source=source_name,
        config=shadow_config,
        block_name=block.get("name", "shadow"),
    )

    if shadow_result is None or not shadow_result.rows:
        logger.warning("Shadow detection failed or found no rows")
        for row in primary_result.rows:
            row.ensemble_confidence = 0.2
        return

    # Compute perpendicular positions for matching
    h, w = mask.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    angle_rad = math.radians(primary_result.dominant_angle_deg)
    pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)

    primary_perps = []
    for row in primary_result.rows:
        perps = [(x - cx) * pdx + (y - cy) * pdy for x, y in row.centerline_px]
        primary_perps.append(float(np.mean(perps)))

    shadow_perps = []
    for row in shadow_result.rows:
        perps = [(x - cx) * pdx + (y - cy) * pdy for x, y in row.centerline_px]
        shadow_perps.append(float(np.mean(perps)))

    spacing_px = primary_result.mean_spacing_m / max(mpp, 1e-6)

    # Hungarian matching
    n_pri = len(primary_perps)
    n_sha = len(shadow_perps)
    cost = np.zeros((n_pri, n_sha), dtype=np.float64)
    for i in range(n_pri):
        for j in range(n_sha):
            cost[i, j] = abs(primary_perps[i] - shadow_perps[j])

    threshold_close = 0.2 * spacing_px
    threshold_match = 0.4 * spacing_px

    pri_idx, sha_idx = linear_sum_assignment(cost)

    matched = {}
    for pi, si in zip(pri_idx, sha_idx):
        dist = cost[pi, si]
        if dist <= threshold_close:
            matched[pi] = 1.0
        elif dist <= threshold_match:
            matched[pi] = 0.6

    for i, row in enumerate(primary_result.rows):
        row.ensemble_confidence = matched.get(i, 0.2)

    n_high = sum(1 for v in matched.values() if v == 1.0)
    n_weak = sum(1 for v in matched.values() if v == 0.6)
    n_none = n_pri - len(matched)
    logger.info(
        "Ensemble confidence: %d high, %d weak, %d unmatched (of %d rows)",
        n_high, n_weak, n_none, n_pri,
    )


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


def generate_lines_only_overlay(
    result,
    image_shape: tuple[int, int],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Create a transparent RGBA image with only row lines drawn (no aerial background).

    Uses magenta (255, 0, 255) with full alpha on line pixels, zero alpha elsewhere.
    Line thickness is 1px (thinner than the default overlay's 2px) so when overlaid
    on the default, the green default lines show through underneath.
    """
    h, w = image_shape[:2]
    canvas = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA, fully transparent

    for row in result.rows:
        # Use ensemble_confidence if available, otherwise regular confidence
        conf = getattr(row, 'ensemble_confidence', None) or row.confidence

        # Magenta with varying alpha based on confidence
        alpha = max(128, int(conf * 255))
        color_rgba = (255, 0, 255, alpha)

        pts = row.centerline_px
        for i in range(len(pts) - 1):
            x1, y1 = int(round(pts[i][0])), int(round(pts[i][1]))
            x2, y2 = int(round(pts[i + 1][0])), int(round(pts[i + 1][1]))
            # Draw on BGR channels + alpha separately
            cv2.line(canvas, (x1, y1), (x2, y2), color_rgba, 1, cv2.LINE_AA)

    return canvas


def generate_thumbnail(image_bgr: np.ndarray, max_size: int = 256) -> np.ndarray:
    """Downscale an image to thumbnail size."""
    h, w = image_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_bgr
