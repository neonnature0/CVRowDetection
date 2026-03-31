"""
Debug artifact generation for the vineyard row detection pipeline.

Produces diagnostic images and JSON summaries for each processed block.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from vinerow.types import BlockRowDetectionResult, QualityFlag

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_all_artifacts(
    output_dir: Path | str,
    result: BlockRowDetectionResult,
    block_name: str = "block",
) -> None:
    """Generate and save all debug artifacts for a detection result.

    Creates the output directory if it doesn't exist.
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    logger.info("Saving debug artifacts to %s", output_dir)

    preprocessed = result.preprocessed
    coarse = result.coarse_orientation
    likelihood = result.likelihood_map

    # 01: Original masked image
    if preprocessed is not None:
        _save_original(output_dir / "01_original.png", preprocessed.image_bgr, preprocessed.mask)

    # 02: Channel overview (4-panel)
    if preprocessed is not None:
        _save_channels(output_dir / "02_channels.png", preprocessed)

    # 03: Channel quality JSON
    if preprocessed is not None:
        _save_channel_quality(output_dir / "03_channel_quality.json", preprocessed)

    # 04: FFT magnitude spectrum
    if coarse is not None and coarse.log_magnitude is not None:
        _save_fft_spectrum(output_dir / "04_fft_magnitude.png", coarse)

    # 05: Coarse orientation overlay
    if preprocessed is not None and coarse is not None:
        _save_coarse_overlay(output_dir / "05_coarse_orientation.png", preprocessed, coarse)

    # 06: Row-likelihood heatmap
    if likelihood is not None:
        _save_likelihood_map(output_dir / "06_likelihood_map.png", likelihood, preprocessed)

    # 07: Candidates overlay
    if result.candidate_points and preprocessed is not None:
        _save_candidates(output_dir / "07_candidates.png", preprocessed, result.candidate_points)

    # 10: Final fitted rows overlay
    if preprocessed is not None:
        _save_fitted_rows(output_dir / "10_fitted_rows.png", preprocessed, result)

    # 11: Spacing histogram
    if result.rows:
        _save_spacing_histogram(output_dir / "11_spacing_histogram.png", result)

    # 13: Quality summary JSON
    _save_quality_summary(output_dir / "13_quality_summary.json", result, block_name)

    # 14: Three-panel comparison
    if preprocessed is not None and likelihood is not None:
        _save_comparison(output_dir / "14_comparison.png", preprocessed, likelihood, result)

    logger.info("Saved %d artifacts to %s", len(list(output_dir.glob("*"))), output_dir)


# ---------------------------------------------------------------------------
# Individual artifact generators
# ---------------------------------------------------------------------------


def _save_original(path: Path, image_bgr: np.ndarray, mask: np.ndarray) -> None:
    """01: Masked aerial image."""
    display = image_bgr.copy()
    # Draw mask boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (255, 255, 255), 2)
    cv2.imwrite(str(path), display)


def _save_channels(path: Path, preprocessed) -> None:
    """02: 4-panel channel overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Preprocessed Channels", fontsize=14)

    channels = [
        ("ExG Vegetation", preprocessed.exg, "Greens"),
        ("Luminance", preprocessed.luminance, "gray"),
        ("Normalized Vegetation", preprocessed.normalized_veg, "YlGn"),
        ("Structure Tensor Magnitude", preprocessed.structure_mag, "hot"),
    ]

    for ax, (title, ch, cmap) in zip(axes.flat, channels):
        if ch is not None:
            ax.imshow(_downsample(ch), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_channel_quality(path: Path, preprocessed) -> None:
    """03: Channel quality scores as JSON."""
    data = []
    for q in preprocessed.channel_qualities:
        data.append({
            "name": q.name,
            "std_dev": round(q.std_dev, 2),
            "contrast": round(q.contrast, 4),
            "weight": round(q.weight, 4),
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_fft_spectrum(path: Path, coarse) -> None:
    """04: FFT magnitude spectrum with peak marked."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(_downsample(coarse.log_magnitude), cmap="inferno")
    if coarse.peak_position:
        px, py = coarse.peak_position
        ax.plot(px, py, "c+", markersize=15, markeredgewidth=2)
        ax.annotate(
            f"angle={coarse.angle_deg:.1f}°\nspacing={coarse.spacing_m:.2f}m",
            (px, py), xytext=(10, 10), textcoords="offset points",
            color="white", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7),
        )
    ax.set_title(f"2D FFT Magnitude (conf={coarse.angle_confidence:.2f})")
    ax.axis("off")
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_coarse_overlay(path: Path, preprocessed, coarse) -> None:
    """05: Original with coarse angle + spacing grid overlaid."""
    import math
    display = preprocessed.image_bgr.copy()
    h, w = display.shape[:2]

    angle_rad = math.radians(coarse.angle_deg)
    spacing_px = coarse.spacing_px
    row_dx = math.cos(angle_rad)
    row_dy = math.sin(angle_rad)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)
    cx, cy = w / 2, h / 2

    # Draw parallel lines at coarse spacing
    max_perp = max(w, h)
    for offset in np.arange(-max_perp, max_perp, spacing_px):
        p1 = (
            int(cx + row_dx * (-max_perp) + perp_x * offset),
            int(cy + row_dy * (-max_perp) + perp_y * offset),
        )
        p2 = (
            int(cx + row_dx * max_perp + perp_x * offset),
            int(cy + row_dy * max_perp + perp_y * offset),
        )
        cv2.line(display, p1, p2, (128, 128, 128), 1, cv2.LINE_AA)

    # Mask boundary
    contours, _ = cv2.findContours(preprocessed.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (255, 255, 255), 2)

    cv2.imwrite(str(path), display)


def _save_likelihood_map(path: Path, likelihood: np.ndarray, preprocessed) -> None:
    """06: Row-likelihood heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(_downsample(likelihood), cmap="jet", vmin=0, vmax=1)
    ax.set_title("Row-Likelihood Map")
    ax.axis("off")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.03)
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_candidates(path: Path, preprocessed, candidates) -> None:
    """07: Likelihood map with candidate points colored by strip."""
    display = preprocessed.image_bgr.copy()

    # Color candidates by strip index
    if candidates:
        max_strip = max(c.strip_index for c in candidates)
        cmap = plt.cm.get_cmap("tab20")
        for c in candidates:
            color_f = cmap(c.strip_index / max(max_strip, 1))
            color_bgr = (
                int(color_f[2] * 255),
                int(color_f[1] * 255),
                int(color_f[0] * 255),
            )
            cv2.circle(display, (int(c.x), int(c.y)), 3, color_bgr, -1)

    cv2.imwrite(str(path), display)


def _save_fitted_rows(path: Path, preprocessed, result: BlockRowDetectionResult) -> None:
    """10: Original with fitted centerlines colored by confidence."""
    display = preprocessed.image_bgr.copy()

    for row in result.rows:
        if len(row.centerline_px) < 2:
            continue

        # Color by confidence: green=high, yellow=medium, red=low
        if row.confidence >= 0.7:
            color = (0, 255, 0)  # green
        elif row.confidence >= 0.4:
            color = (0, 200, 255)  # yellow
        else:
            color = (0, 0, 255)  # red

        pts = [(int(x), int(y)) for x, y in row.centerline_px]
        for i in range(len(pts) - 1):
            cv2.line(display, pts[i], pts[i+1], color, 2, cv2.LINE_AA)

        # Row number label at midpoint
        mid = pts[len(pts) // 2]
        cv2.putText(
            display, str(row.row_index),
            (mid[0] + 5, mid[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Mask boundary
    contours, _ = cv2.findContours(preprocessed.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (255, 255, 255), 2)

    cv2.imwrite(str(path), display)


def _save_spacing_histogram(path: Path, result: BlockRowDetectionResult) -> None:
    """11: Distribution of per-row spacings."""
    spacings = [r.spacing_to_prev_m for r in result.rows if r.spacing_to_prev_m is not None]
    if not spacings:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(spacings, bins=max(10, len(spacings) // 3), edgecolor="black", alpha=0.7)
    ax.axvline(result.mean_spacing_m, color="red", linestyle="--", label=f"Mean: {result.mean_spacing_m:.2f}m")
    ax.axvline(result.median_spacing_m, color="blue", linestyle="--", label=f"Median: {result.median_spacing_m:.2f}m")
    ax.set_xlabel("Row Spacing (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Row Spacing Distribution (n={len(spacings)}, std={result.spacing_std_m:.3f}m)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_quality_summary(path: Path, result: BlockRowDetectionResult, block_name: str) -> None:
    """13: Complete quality summary as JSON."""
    data = {
        "block_name": block_name,
        "row_count": result.row_count,
        "dominant_angle_deg": result.dominant_angle_deg,
        "dominant_angle_bearing": result.dominant_angle_bearing,
        "angle_confidence": result.angle_confidence,
        "mean_spacing_m": result.mean_spacing_m,
        "median_spacing_m": result.median_spacing_m,
        "spacing_std_m": result.spacing_std_m,
        "spacing_range_m": list(result.spacing_range_m),
        "overall_confidence": result.overall_confidence,
        "quality_flags": str(result.quality_flags),
        "image_size": list(result.image_size),
        "meters_per_pixel": result.meters_per_pixel,
        "tile_source": result.tile_source,
        "zoom_level": result.zoom_level,
        "timings": {
            "acquisition_s": round(result.timings.acquisition, 3),
            "preprocessing_s": round(result.timings.preprocessing, 3),
            "orientation_s": round(result.timings.orientation, 3),
            "ridge_s": round(result.timings.ridge, 3),
            "candidates_s": round(result.timings.candidates, 3),
            "tracking_s": round(result.timings.tracking, 3),
            "fitting_s": round(result.timings.fitting, 3),
            "postprocessing_s": round(result.timings.postprocessing, 3),
            "total_s": round(result.timings.total, 3),
        },
        "per_row": [
            {
                "index": r.row_index,
                "confidence": r.confidence,
                "length_m": r.length_m,
                "spacing_to_prev_m": r.spacing_to_prev_m,
                "curvature_max": r.curvature_max_deg_per_m,
            }
            for r in result.rows
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _downsample(image: np.ndarray, max_dim: int = 2000) -> np.ndarray:
    """Downsample an image if either dimension exceeds max_dim."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _save_comparison(path: Path, preprocessed, likelihood, result: BlockRowDetectionResult) -> None:
    """14: Three-panel comparison (original / likelihood / fitted rows)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Downsample for matplotlib (avoid OOM on large images)
    rgb = cv2.cvtColor(_downsample(preprocessed.image_bgr), cv2.COLOR_BGR2RGB)
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Panel 2: Likelihood (downsample)
    lk_small = _downsample(likelihood)
    axes[1].imshow(lk_small, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Row Likelihood")
    axes[1].axis("off")

    # Panel 3: Fitted rows on original (use downsampled)
    h_orig, w_orig = preprocessed.image_bgr.shape[:2]
    h_small, w_small = rgb.shape[:2]
    scale_x = w_small / w_orig
    scale_y = h_small / h_orig
    overlay = rgb.copy()
    for row in result.rows:
        if len(row.centerline_px) < 2:
            continue
        pts = np.array([(int(x * scale_x), int(y * scale_y)) for x, y in row.centerline_px])
        if row.confidence >= 0.7:
            color = (0, 255, 0)
        elif row.confidence >= 0.4:
            color = (255, 200, 0)
        else:
            color = (255, 0, 0)
        for i in range(len(pts) - 1):
            cv2.line(overlay, tuple(pts[i]), tuple(pts[i+1]), color, 2, cv2.LINE_AA)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Detected Rows (n={result.row_count})")
    axes[2].axis("off")

    fig.suptitle(
        f"Angle={result.dominant_angle_deg:.1f}° | "
        f"Spacing={result.mean_spacing_m:.2f}m | "
        f"Conf={result.overall_confidence:.2f} | "
        f"Flags={result.quality_flags}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
