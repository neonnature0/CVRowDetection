#!/usr/bin/env python3
"""
Generate visual verification images for all test blocks using the NEW pipeline.

Draws 2px magenta lines on each detected row with row numbers and spacing labels.
Saves to output/visual_verify/ with vineyard-qualified filenames.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from vinerow.acquisition.geo_utils import meters_per_pixel, polygon_bbox
from vinerow.acquisition.tile_fetcher import (
    TILE_SOURCES,
    auto_select_source,
    default_zoom,
    fetch_and_stitch,
)
from vinerow.config import PipelineConfig
from vinerow.pipeline import run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s » %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join("output", "visual_verify")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _clip_polyline_to_mask(
    pts: list[tuple[float, float]],
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Keep only polyline points that fall within the mask."""
    h, w = mask.shape[:2]
    valid = []
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
            valid.append((ix, iy))
    return valid


def _put_text_outlined(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    font_thick: int,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw text with a dark outline for readability on any background."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, bg, font_thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg, font_thick, cv2.LINE_AA)


def _put_text_rotated(
    canvas: np.ndarray,
    text: str,
    center: tuple[int, int],
    angle_deg: float,
    font_scale: float,
    font_thick: int,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Render text rotated by angle_deg around center onto canvas."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick + 2,
    )
    pad = 4
    tmp_w = tw + pad * 2
    tmp_h = th + baseline + pad * 2

    tmp = np.zeros((tmp_h, tmp_w, 4), dtype=np.uint8)
    text_org = (pad, th + pad)
    cv2.putText(tmp, text, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (*bg, 255), font_thick + 2, cv2.LINE_AA)
    cv2.putText(tmp, text, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (*fg, 255), font_thick, cv2.LINE_AA)

    tmp_cx, tmp_cy = tmp_w / 2, tmp_h / 2
    M = cv2.getRotationMatrix2D((tmp_cx, tmp_cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(tmp_h * sin_a + tmp_w * cos_a) + 2
    new_h = int(tmp_h * cos_a + tmp_w * sin_a) + 2
    M[0, 2] += (new_w - tmp_w) / 2
    M[1, 2] += (new_h - tmp_h) / 2

    rotated = cv2.warpAffine(tmp, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    cx, cy = center
    x0 = cx - new_w // 2
    y0 = cy - new_h // 2
    x1 = x0 + new_w
    y1 = y0 + new_h

    ch, cw = canvas.shape[:2]
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(cw, x1)
    dy1 = min(ch, y1)
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)

    if dx1 <= dx0 or dy1 <= dy0 or sx1 <= sx0 or sy1 <= sy0:
        return

    patch = rotated[sy0:sy1, sx0:sx1]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[dy0:dy1, dx0:dx1]
    blended = (roi * (1 - alpha) + patch[:, :, :3] * alpha).astype(np.uint8)
    canvas[dy0:dy1, dx0:dx1] = blended


# ---------------------------------------------------------------------------
# Main drawing function
# ---------------------------------------------------------------------------


def draw_verification_overlay(
    image_bgr: np.ndarray,
    result,
    mask: np.ndarray,
    mpp: float,
) -> np.ndarray:
    """Draw magenta row lines with numbers and spacing labels."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    line_color = (255, 0, 255)  # Magenta BGR

    rows = result.rows
    n_rows = len(rows)
    if n_rows == 0:
        return canvas

    # Adaptive label interval
    if n_rows > 100:
        label_every = 10
    elif n_rows > 40:
        label_every = 5
    else:
        label_every = 1

    # Font sizing from spacing
    spacing_px = result.mean_spacing_m / max(mpp, 1e-6)
    num_font = max(0.35, spacing_px * label_every / 80.0)
    num_thick = max(1, int(num_font * 1.3 + 0.5))
    spc_font = max(0.30, spacing_px * label_every / 100.0)
    spc_thick = max(1, int(spc_font * 1.3 + 0.5))

    (_, digit_h), _ = cv2.getTextSize("8", cv2.FONT_HERSHEY_SIMPLEX, num_font, num_thick)
    stagger_offset = int(digit_h * 1.6)

    # Clip and draw each row
    clipped_rows: list[list[tuple[int, int]]] = []
    for row in rows:
        clipped = _clip_polyline_to_mask(row.centerline_px, mask)
        clipped_rows.append(clipped)

    # Determine left-to-right display order
    # Check if row 0 is left or right of last row
    first_clipped = clipped_rows[0] if clipped_rows else []
    last_clipped = clipped_rows[-1] if len(clipped_rows) > 1 else []
    reverse = False
    if first_clipped and last_clipped:
        first_top = first_clipped[0]
        last_top = last_clipped[0]
        if first_top[0] > last_top[0]:
            reverse = True

    for idx, (row, clipped) in enumerate(zip(rows, clipped_rows)):
        if len(clipped) < 2:
            continue

        # Draw polyline
        pts_arr = np.array(clipped, dtype=np.int32)
        cv2.polylines(canvas, [pts_arr], isClosed=False, color=line_color,
                      thickness=2, lineType=cv2.LINE_AA)

        # Display number (1-based, left-to-right)
        display_num = (n_rows - idx) if reverse else (idx + 1)

        if display_num % label_every == 0 or display_num == 1 or display_num == n_rows:
            # Row number at top of line
            top_pt = clipped[0]
            stagger = stagger_offset if (display_num % 2 == 0) else 0
            label_y = max(top_pt[1] - 8 - stagger, digit_h + 4)
            _put_text_outlined(
                canvas, str(display_num),
                (top_pt[0] - 4, label_y),
                num_font, num_thick,
            )

            # Spacing label at bottom between this row and next
            if idx < n_rows - 1 and row.spacing_to_prev_m is not None:
                next_clipped = clipped_rows[idx + 1] if idx + 1 < len(clipped_rows) else []
                if clipped and next_clipped:
                    bot_this = clipped[-1]
                    bot_next = next_clipped[-1] if next_clipped else bot_this
                    mid_x = (bot_this[0] + bot_next[0]) // 2
                    mid_y = (bot_this[1] + bot_next[1]) // 2

                    spacing_text = f"{row.spacing_to_prev_m:.2f}"
                    text_angle = result.dominant_angle_deg - 90
                    _put_text_rotated(
                        canvas, spacing_text,
                        (mid_x, mid_y),
                        text_angle,
                        spc_font, spc_thick,
                        fg=(200, 255, 200),
                    )

    # Draw mask boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)

    # Title bar
    title = (
        f"{n_rows} rows | angle={result.dominant_angle_deg:.1f}° | "
        f"spacing={result.mean_spacing_m:.2f}m (std={result.spacing_std_m:.3f}) | "
        f"conf={result.overall_confidence:.2f} | {result.quality_flags}"
    )
    _put_text_outlined(canvas, title, (10, 30), 0.6, 1, fg=(0, 255, 255))

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load test blocks
    blocks_path = Path(__file__).parent / "data" / "blocks" / "test_blocks.json"
    with open(blocks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    blocks = data.get("blocks", [])

    logger.info("Processing %d blocks for visual verification", len(blocks))

    config = PipelineConfig(
        save_debug_artifacts=False,  # We'll save our own overlay
    )

    for block in blocks:
        name = block["name"]
        vineyard = block.get("vineyard_name", "Unknown")
        boundary = block["boundary"]
        coords = boundary["coordinates"][0]

        safe_name = f"{vineyard}_{name}".replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")

        logger.info("=" * 60)
        logger.info("Processing: %s (%s)", name, vineyard)

        # Tile fetch
        bbox = polygon_bbox(coords)
        centroid_lng = (bbox[0] + bbox[2]) / 2.0
        centroid_lat = (bbox[1] + bbox[3]) / 2.0
        source_name = auto_select_source(centroid_lng)
        source = TILE_SOURCES[source_name]
        zoom = default_zoom(source_name)

        t0 = time.perf_counter()
        image_bgr, mask, tile_origin = fetch_and_stitch(
            source, coords, zoom, source_name,
            cache_dir=config.tile_cache_dir,
        )
        fetch_time = time.perf_counter() - t0

        mpp = meters_per_pixel(centroid_lat, zoom, source.tile_size)

        # Run pipeline
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
        )

        if result is None:
            logger.error("Pipeline failed for %s — saving original image", name)
            canvas = image_bgr.copy()
            _put_text_outlined(canvas, f"DETECTION FAILED: {name}", (10, 30), 0.8, 2, fg=(0, 0, 255))
            cv2.imwrite(output_path, canvas)
            continue

        # Draw overlay
        canvas = draw_verification_overlay(image_bgr, result, mask, mpp)

        # Ground truth comparison text at bottom
        gt_spacing = block.get("row_spacing_m")
        gt_count = block.get("row_count")
        gt_text_parts = []
        if gt_spacing:
            err = abs(result.mean_spacing_m - gt_spacing) / gt_spacing * 100
            gt_text_parts.append(f"GT spacing: {gt_spacing:.2f}m (err={err:.1f}%)")
        if gt_count:
            gt_text_parts.append(f"GT rows: {gt_count} (det={result.row_count}, diff={result.row_count - gt_count:+d})")
        if gt_text_parts:
            gt_text = " | ".join(gt_text_parts)
            bh = canvas.shape[0]
            _put_text_outlined(canvas, gt_text, (10, bh - 15), 0.5, 1, fg=(128, 255, 128))

        cv2.imwrite(output_path, canvas)
        logger.info(
            "Saved: %s (%d rows, %.2fm spacing, conf=%.2f, %.1fs)",
            output_path, result.row_count, result.mean_spacing_m,
            result.overall_confidence, result.timings.total + fetch_time,
        )

    logger.info("=" * 60)
    logger.info("All blocks processed. Output in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
