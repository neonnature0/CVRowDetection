"""
Generate visual verification images for all test blocks using the
production vinerow/ pipeline.

Draws detected row centerlines on the aerial image with row numbers,
spacing labels, and a summary header. Also produces a 4-panel overview
(aerial | likelihood | rows | metrics).

Output: output/visual_verify/

Usage:
    python visual_verify.py                  # All blocks
    python visual_verify.py --blocks "B10"   # Single block
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dotenv import load_dotenv

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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s >> %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output") / "visual_verify"

COMPASS_TO_DEGREES: dict[str, float] = {
    "N-S": 90.0, "S-N": 90.0, "E-W": 0.0, "W-E": 0.0,
    "NE-SW": 45.0, "SW-NE": 45.0, "NW-SE": 135.0, "SE-NW": 135.0,
}


def bearing_to_image_angle(bearing_deg: float) -> float:
    rad = math.radians(bearing_deg)
    return math.degrees(math.atan2(-math.cos(rad), math.sin(rad))) % 180.0


def angular_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put_text_outlined(
    canvas: np.ndarray, text: str, org: tuple[int, int],
    font_scale: float, font_thick: int,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw text with a dark outline for readability."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, bg, font_thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg, font_thick, cv2.LINE_AA)


def _clip_line_to_mask(
    pts: list[tuple[float, float]], mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Keep only the portion of a polyline inside the mask."""
    h, w = mask.shape[:2]
    if len(pts) < 2:
        return []
    valid: list[tuple[int, int]] = []
    for px, py in pts:
        x, y = int(round(px)), int(round(py))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            valid.append((x, y))
    return valid


def _downsample(image: np.ndarray, max_dim: int = 2000) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Row overlay drawing (OpenCV, full-res)
# ---------------------------------------------------------------------------

def draw_row_overlay(
    image_bgr: np.ndarray,
    result: BlockRowDetectionResult,
    mask: np.ndarray,
    mpp: float,
    block_name: str,
    vineyard: str,
    gt_spacing: float | None,
    gt_rows: int | None,
) -> np.ndarray:
    """Draw row lines + labels on a copy of the aerial image."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    n_rows = result.row_count
    rows = result.rows

    # Adaptive label interval
    if n_rows > 100:
        label_every = 10
    elif n_rows > 40:
        label_every = 5
    else:
        label_every = 1

    spacing_px = result.mean_spacing_m / max(mpp, 1e-6)
    num_font = max(0.30, spacing_px * label_every / 90.0)
    num_thick = max(1, int(num_font * 1.3 + 0.5))

    # Clip and draw each row
    clipped_rows: list[list[tuple[int, int]]] = []
    for row in rows:
        clipped = _clip_line_to_mask(row.centerline_px, mask)
        clipped_rows.append(clipped)

        if len(clipped) < 2:
            continue

        # Color by confidence
        if row.confidence >= 0.7:
            color = (0, 255, 0)    # green
        elif row.confidence >= 0.4:
            color = (0, 200, 255)  # yellow
        else:
            color = (0, 0, 255)    # red

        for i in range(len(clipped) - 1):
            cv2.line(canvas, clipped[i], clipped[i + 1], color, 2, cv2.LINE_AA)

    # Determine left-to-right ordering for row numbers
    reverse = False
    first_valid = next((c for c in clipped_rows if len(c) >= 2), None)
    last_valid = next((c for c in reversed(clipped_rows) if len(c) >= 2), None)
    if first_valid and last_valid:
        if first_valid[0][0] > last_valid[0][0]:
            reverse = True

    # Row number labels at top endpoint
    (_, digit_h), _ = cv2.getTextSize("8", cv2.FONT_HERSHEY_SIMPLEX, num_font, num_thick)
    stagger = int(digit_h * 1.6)
    label_idx = 0
    for idx, row in enumerate(rows):
        display_n = (n_rows - idx) if reverse else (idx + 1)
        is_endpoint = idx == 0 or idx == n_rows - 1
        if not is_endpoint and display_n % label_every != 0:
            continue

        clipped = clipped_rows[idx]
        if len(clipped) < 2:
            continue

        # Top = min y
        top = clipped[0] if clipped[0][1] <= clipped[-1][1] else clipped[-1]
        label = str(display_n)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, num_font, num_thick)
        y_off = int(digit_h * 0.8) + (stagger if label_idx % 2 == 1 else 0)
        label_idx += 1
        lx = max(2, min(top[0] - tw // 2, w - tw - 2))
        ly = max(digit_h + 4, top[1] - y_off)
        _put_text_outlined(canvas, label, (lx, ly), num_font, num_thick, fg=(0, 255, 255))

    # Block boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)

    # Header
    scale = max(0.5, min(h, w) / 3000.0)
    thick = max(1, int(scale + 0.5))
    y = int(28 * scale)
    gap = int(28 * scale)

    header_lines = [
        f"{vineyard} / {block_name}",
        f"{n_rows} rows   "
        f"spacing={result.mean_spacing_m:.2f}m (std={result.spacing_std_m:.3f}m)   "
        f"angle={result.dominant_angle_deg:.1f} deg   "
        f"conf={result.overall_confidence:.2f}",
    ]
    if gt_rows is not None:
        recall = min(n_rows, gt_rows) / gt_rows * 100
        sp_err = abs(result.mean_spacing_m - gt_spacing) / gt_spacing * 100 if gt_spacing else 0
        header_lines.append(
            f"GT: {gt_rows} rows, {gt_spacing:.2f}m   |   "
            f"Recall: {recall:.1f}%   Spacing err: {sp_err:.1f}%"
        )

    for text in header_lines:
        _put_text_outlined(canvas, text, (10, y), scale, thick)
        y += gap

    return canvas


# ---------------------------------------------------------------------------
# 4-panel matplotlib overview
# ---------------------------------------------------------------------------

def save_overview_panel(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    result: BlockRowDetectionResult,
    config: PipelineConfig,
    mpp: float,
    block: dict,
    output_path: Path,
    elapsed: float,
) -> None:
    """Save a 4-panel overview: aerial | likelihood | rows | metrics."""
    name = block["name"]
    vineyard = block.get("vineyard_name", "Unknown")

    gt_spacing = block.get("row_spacing_m")
    gt_count = block.get("row_count")
    gt_bearing = block.get("row_angle")
    gt_orient = block.get("row_orientation")
    gt_angle_img = None
    if gt_bearing is not None:
        gt_angle_img = bearing_to_image_angle(gt_bearing)
    elif gt_orient and gt_orient in COMPASS_TO_DEGREES:
        gt_angle_img = COMPASS_TO_DEGREES[gt_orient]

    recall = min(result.row_count, gt_count) / gt_count if gt_count else None
    spacing_err = abs(result.mean_spacing_m - gt_spacing) / gt_spacing * 100 if gt_spacing else None
    angle_err = angular_distance(result.dominant_angle_deg, gt_angle_img) if gt_angle_img is not None else None

    # Downsample
    rgb = cv2.cvtColor(_downsample(image_bgr), cv2.COLOR_BGR2RGB)
    mask_ds = _downsample(mask)
    h_orig, w_orig = image_bgr.shape[:2]
    h_ds, w_ds = rgb.shape[:2]
    sx, sy = w_ds / w_orig, h_ds / h_orig

    likelihood = result.likelihood_map
    lk_ds = _downsample(likelihood) if likelihood is not None else np.zeros((h_ds, w_ds), dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor("#1a1a2e")

    # Panel 1: Aerial
    ax = axes[0, 0]
    ax.imshow(rgb)
    contours_ds, _ = cv2.findContours(mask_ds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_ds:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 2:
            poly = np.vstack([pts, pts[:1]])
            ax.plot(poly[:, 0], poly[:, 1], color="white", linewidth=1.5, alpha=0.8)
    ax.set_title("Aerial Image", fontsize=12, color="white", fontweight="bold")
    ax.axis("off")

    # Panel 2: Likelihood
    ax = axes[0, 1]
    ax.imshow(rgb, alpha=0.3)
    im = ax.imshow(lk_ds, cmap="inferno", vmin=0, vmax=1, alpha=0.85)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Likelihood")
    ax.set_title(f"Ridge Likelihood [{config.ridge_mode}]", fontsize=12, color="white", fontweight="bold")
    ax.axis("off")

    # Panel 3: Rows overlay
    ax = axes[1, 0]
    ax.imshow(rgb)
    for cnt in contours_ds:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 2:
            poly = np.vstack([pts, pts[:1]])
            ax.plot(poly[:, 0], poly[:, 1], color="white", linewidth=1, alpha=0.4)

    high_n = med_n = low_n = 0
    for row in result.rows:
        if len(row.centerline_px) < 2:
            continue
        pts = np.array([(x * sx, y * sy) for x, y in row.centerline_px])
        if row.confidence >= 0.7:
            color, high_n = "#00ff88", high_n + 1
        elif row.confidence >= 0.4:
            color, med_n = "#ffcc00", med_n + 1
        else:
            color, low_n = "#ff4444", low_n + 1
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, alpha=0.9)

    legend_items = []
    if high_n:
        legend_items.append(mpatches.Patch(color="#00ff88", label=f"High conf ({high_n})"))
    if med_n:
        legend_items.append(mpatches.Patch(color="#ffcc00", label=f"Med conf ({med_n})"))
    if low_n:
        legend_items.append(mpatches.Patch(color="#ff4444", label=f"Low conf ({low_n})"))
    if legend_items:
        ax.legend(handles=legend_items, loc="lower right", fontsize=8,
                  facecolor="black", edgecolor="gray", labelcolor="white")
    ax.set_title(f"Detected Rows (n={result.row_count})", fontsize=12, color="white", fontweight="bold")
    ax.axis("off")

    # Panel 4: Metrics
    ax = axes[1, 1]
    ax.set_facecolor("#0f0f23")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    flags_str = str(result.quality_flags)
    if flags_str == "QualityFlag.NONE":
        flags_str = "NONE"
    else:
        flags_str = flags_str.replace("QualityFlag.", "")

    def _color_recall(r):
        if r is None: return "#888888"
        return "#00ff88" if r >= 0.8 else "#ffcc00" if r >= 0.5 else "#ff4444"

    def _color_spacing(e):
        if e is None: return "#888888"
        return "#00ff88" if e < 5 else "#ffcc00" if e < 20 else "#ff4444"

    def _color_angle(e):
        if e is None: return "#888888"
        return "#00ff88" if e < 2 else "#ffcc00"

    lines = [
        (f"{name} ({vineyard})", 15, "bold", "#ffffff"),
        ("", 0, "", ""),
        ("DETECTION", 11, "bold", "#6699ff"),
        (f"  Rows detected:   {result.row_count}" + (f"  /  {gt_count} GT" if gt_count else ""), 11, "normal", "#ffffff"),
        (f"  Recall:          {recall*100:.1f}%" if recall is not None else "  Recall:          N/A", 11, "normal", _color_recall(recall)),
        (f"  Confidence:      {result.overall_confidence:.2f}", 11, "normal", "#ffffff"),
        ("", 0, "", ""),
        ("GEOMETRY", 11, "bold", "#6699ff"),
        (f"  Angle:           {result.dominant_angle_deg:.1f} deg" + (f"  (err: {angle_err:.1f} deg)" if angle_err is not None else ""), 11, "normal", _color_angle(angle_err)),
        (f"  Spacing:         {result.mean_spacing_m:.2f} m" + (f"  (err: {spacing_err:.1f}%)" if spacing_err is not None else ""), 11, "normal", _color_spacing(spacing_err)),
        (f"  Spacing std:     {result.spacing_std_m:.3f} m", 11, "normal", "#aaaaaa"),
        ("", 0, "", ""),
        ("QUALITY", 11, "bold", "#6699ff"),
        (f"  Flags:           {flags_str}", 11, "normal", "#00ff88" if flags_str == "NONE" else "#ffcc00"),
        (f"  Ridge mode:      {config.ridge_mode}", 11, "normal", "#aaaaaa"),
        ("", 0, "", ""),
        ("PERFORMANCE", 11, "bold", "#6699ff"),
        (f"  Total time:      {elapsed:.1f}s", 11, "normal", "#aaaaaa"),
        (f"  Image size:      {w_orig} x {h_orig} px", 11, "normal", "#aaaaaa"),
        (f"  Resolution:      {mpp:.3f} m/px", 11, "normal", "#aaaaaa"),
    ]

    y = 9.5
    for text, fontsize, weight, color in lines:
        if text == "":
            y -= 0.15
            continue
        ax.text(0.5, y, text, fontsize=fontsize, fontweight=weight, color=color,
                fontfamily="monospace", va="top")
        y -= 0.45

    # Recall badge
    if recall is not None:
        ax.text(8.5, 9.2, f"{recall*100:.0f}%", fontsize=28, fontweight="bold",
                color=_color_recall(recall), ha="center", va="top", fontfamily="monospace")
        ax.text(8.5, 8.3, "recall", fontsize=9, color="#888888", ha="center", va="top")

    fig.suptitle(
        f"Visual Verification  |  {name} ({vineyard})  |  {config.ridge_mode} mode",
        fontsize=14, color="white", fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_test_blocks(path: str = "test_blocks.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("blocks", [])


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visual verification of row detection")
    parser.add_argument("--blocks", type=str, default=None, help="Comma-separated block names")
    parser.add_argument("--overlay-only", action="store_true", help="Only save row overlay (no 4-panel)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    blocks = load_test_blocks()
    if args.blocks:
        names = [n.strip() for n in args.blocks.split(",")]
        blocks = [b for b in blocks if b["name"] in names]

    if not blocks:
        print("No blocks to process.")
        sys.exit(1)

    config = PipelineConfig(save_debug_artifacts=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visual verification for {len(blocks)} blocks (mode={config.ridge_mode})")
    print(f"Output: {OUTPUT_DIR.resolve()}\n")

    for i, block in enumerate(blocks):
        name = block["name"]
        vineyard = block.get("vineyard_name", "Unknown")
        gt_spacing = block.get("row_spacing_m")
        gt_rows = block.get("row_count")
        coords = block["boundary"]["coordinates"][0]

        safe_name = f"{vineyard.replace(' ', '_')}_{name.replace(' ', '_')}"
        logger.info("=== [%d/%d] %s / %s ===", i + 1, len(blocks), vineyard, name)

        # Fetch tiles
        bbox = polygon_bbox(coords)
        cx_lng = (bbox[0] + bbox[2]) / 2.0
        cx_lat = (bbox[1] + bbox[3]) / 2.0
        source_name = auto_select_source(cx_lng)
        source = TILE_SOURCES[source_name]
        zoom = default_zoom(source_name)

        try:
            image_bgr, mask, tile_origin = fetch_and_stitch(
                source, coords, zoom, source_name, cache_dir=config.tile_cache_dir,
            )
        except Exception as e:
            logger.error("Tile fetch failed: %s", e)
            continue

        mpp = meters_per_pixel(cx_lat, zoom, source.tile_size)

        # Run pipeline
        t0 = time.perf_counter()
        try:
            result = run_pipeline(
                image_bgr=image_bgr, mask=mask, mpp=mpp, lat=cx_lat,
                zoom=zoom, tile_size=source.tile_size, tile_origin=tile_origin,
                tile_source=source_name, config=config,
            )
        except Exception as e:
            logger.error("Pipeline failed: %s", e)
            gc.collect()
            continue
        elapsed = time.perf_counter() - t0

        if result is None:
            logger.error("Pipeline returned None")
            continue

        recall = min(result.row_count, gt_rows) / gt_rows if gt_rows else None
        recall_str = f"{recall*100:.0f}%" if recall else "N/A"

        # 1. Full-res row overlay
        overlay_path = OUTPUT_DIR / f"{safe_name}_rows.png"
        overlay = draw_row_overlay(
            image_bgr, result, mask, mpp, name, vineyard, gt_spacing, gt_rows,
        )
        cv2.imwrite(str(overlay_path), overlay)
        logger.info("Saved overlay: %s  (%d rows, recall=%s)", overlay_path.name, result.row_count, recall_str)

        # 2. 4-panel overview
        if not args.overlay_only:
            panel_path = OUTPUT_DIR / f"{safe_name}_overview.png"
            try:
                save_overview_panel(
                    image_bgr, mask, result, config, mpp, block, panel_path, elapsed,
                )
                logger.info("Saved overview: %s", panel_path.name)
            except Exception as e:
                logger.error("Overview panel failed: %s", e)

        # Free memory between blocks
        del result, overlay, image_bgr, mask
        gc.collect()

    print(f"\nDone. Images saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
