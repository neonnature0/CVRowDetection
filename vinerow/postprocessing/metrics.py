"""
Post-processing: compute block-level metrics, quality flags, and assemble
the final BlockRowDetectionResult.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from vinerow.config import PipelineConfig
from vinerow.types import (
    BlockRowDetectionResult,
    CoarseOrientation,
    FittedRow,
    OcclusionGap,
    PreprocessedChannels,
    QualityFlag,
    RowCandidate,
    StageTimings,
)

logger = logging.getLogger(__name__)


def _image_angle_to_bearing(image_angle_deg: float) -> float:
    """Convert image-space angle (0-180) to geographic bearing (0-360)."""
    rad = math.radians(image_angle_deg)
    bearing = math.degrees(math.atan2(math.cos(rad), -math.sin(rad))) % 360.0
    return round(bearing, 2)


def compute_block_metrics(
    fitted_rows: list[FittedRow],
    coarse: CoarseOrientation,
    preprocessed: PreprocessedChannels,
    likelihood_map: np.ndarray,
    candidates: list[RowCandidate],
    mpp: float,
    tile_source: str,
    zoom: int,
    image_size: tuple[int, int],
    timings: StageTimings,
    config: PipelineConfig,
    occlusion_gaps: list[OcclusionGap] | None = None,
) -> BlockRowDetectionResult:
    """Compute all block-level metrics and quality flags.

    Args:
        fitted_rows: Fitted centerlines from Stage 6.
        coarse: Coarse orientation from Stage 2.
        preprocessed: Preprocessed channels from Stage 1.
        likelihood_map: Ridge likelihood map from Stage 3.
        candidates: All candidates from Stage 4.
        mpp: Meters per pixel.
        tile_source: Tile source name.
        zoom: Zoom level.
        image_size: (width, height).
        timings: Stage timing accumulator.
        config: Pipeline configuration.

    Returns:
        Complete BlockRowDetectionResult.
    """
    # Filter rows by confidence and length
    # Use median length as reference (robust to outliers), not max
    row_lengths = [r.length_m for r in fitted_rows if r.length_m > 0]
    if row_lengths:
        median_length = float(np.median(row_lengths))
    else:
        median_length = 0.0
    min_length = median_length * config.min_row_length_fraction

    valid_rows = [
        r for r in fitted_rows
        if r.confidence >= config.min_row_confidence and r.length_m >= min_length
    ]

    logger.info(
        "Post-processing: %d/%d rows pass filters (conf>=%.2f, len>=%.1fm)",
        len(valid_rows), len(fitted_rows), config.min_row_confidence, min_length,
    )

    # Spacing statistics from fitted rows
    spacings = [r.spacing_to_prev_m for r in valid_rows if r.spacing_to_prev_m is not None]

    if spacings:
        mean_spacing = float(np.mean(spacings))
        median_spacing = float(np.median(spacings))
        std_spacing = float(np.std(spacings))
        min_spacing = float(np.min(spacings))
        max_spacing = float(np.max(spacings))
    else:
        mean_spacing = coarse.spacing_m
        median_spacing = coarse.spacing_m
        std_spacing = 0.0
        min_spacing = coarse.spacing_m
        max_spacing = coarse.spacing_m

    # Dominant angle: from fitted rows (mean of row angles at midpoint)
    # For now, use the coarse FFT angle since individual row angles aren't
    # computed separately. The FFT angle is very accurate (<1 degree).
    dominant_angle = coarse.angle_deg
    dominant_bearing = _image_angle_to_bearing(dominant_angle)

    # Overall confidence: combine multiple signals
    if valid_rows:
        row_confidences = [r.confidence for r in valid_rows]
        median_row_conf = float(np.median(row_confidences))
    else:
        median_row_conf = 0.0

    # Spacing regularity
    spacing_cv = std_spacing / mean_spacing if mean_spacing > 0 else 0.0
    spacing_regularity = max(0.0, 1.0 - spacing_cv)

    # Completeness: detected rows vs expected
    expected_rows = coarse.row_count_estimate
    if expected_rows > 0:
        completeness = min(len(valid_rows) / expected_rows, 1.0)
    else:
        completeness = 0.5

    overall_confidence = (
        0.4 * coarse.angle_confidence
        + 0.3 * median_row_conf
        + 0.2 * spacing_regularity
        + 0.1 * completeness
    )
    overall_confidence = round(min(overall_confidence, 1.0), 3)

    # Quality flags
    flags = QualityFlag.NONE

    if overall_confidence < 0.3:
        flags |= QualityFlag.LOW_CONFIDENCE

    if spacing_cv > config.spacing_cv_warning:
        flags |= QualityFlag.SPACING_IRREGULAR

    # Check for missing rows (gaps > 1.5x median)
    if spacings and median_spacing > 0:
        for sp in spacings:
            if sp > config.missing_row_factor * median_spacing:
                flags |= QualityFlag.MISSING_ROWS
                break

    # Weak signal: low likelihood map contrast
    mask_pixels = preprocessed.mask > 0
    if mask_pixels.any():
        lm_mean = float(likelihood_map[mask_pixels].mean())
        if lm_mean < 0.15:
            flags |= QualityFlag.WEAK_SIGNAL

    if coarse.angle_confidence < 0.3:
        flags |= QualityFlag.ORIENTATION_UNCERTAIN

    if len(valid_rows) < 5:
        flags |= QualityFlag.FEW_ROWS

    # Harmonic spacing: flag if median spacing is implausibly large
    if median_spacing > 4.5:
        flags |= QualityFlag.HARMONIC_SPACING
        logger.warning(
            "Possible harmonic: median_spacing=%.2fm > 4.5m", median_spacing,
        )

    # Internal occlusion: flag if any occlusion gaps were detected
    if occlusion_gaps:
        flags |= QualityFlag.INTERNAL_OCCLUSION

    return BlockRowDetectionResult(
        rows=valid_rows,
        row_count=len(valid_rows),
        dominant_angle_deg=round(dominant_angle, 2),
        dominant_angle_bearing=dominant_bearing,
        angle_confidence=round(coarse.angle_confidence, 3),
        mean_spacing_m=round(mean_spacing, 3),
        median_spacing_m=round(median_spacing, 3),
        spacing_std_m=round(std_spacing, 4),
        spacing_range_m=(round(min_spacing, 3), round(max_spacing, 3)),
        overall_confidence=overall_confidence,
        quality_flags=flags,
        timings=timings,
        image_size=image_size,
        meters_per_pixel=round(mpp, 4),
        tile_source=tile_source,
        zoom_level=zoom,
        occlusion_gaps=occlusion_gaps or [],
        coarse_orientation=coarse,
        likelihood_map=likelihood_map,
        candidate_points=candidates,
        preprocessed=preprocessed,
    )
