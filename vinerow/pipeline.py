"""
Top-level pipeline orchestrator for vineyard row detection.

Runs all stages in sequence, collects timing, and assembles the final
BlockRowDetectionResult.
"""

from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np

from vinerow.config import PipelineConfig
from vinerow.types import (
    BlockRowDetectionResult,
    CoarseOrientation,
    FittedRow,
    PreprocessedChannels,
    QualityFlag,
    RowCandidate,
    RowTrajectory,
    StageTimings,
)

logger = logging.getLogger(__name__)


def _image_angle_to_bearing(image_angle_deg: float) -> float:
    """Convert image-space angle (0-180) to geographic bearing (0-360).

    Image coords: 0 = E-W horizontal, 90 = N-S vertical.
    Bearing: 0 = North, 90 = East.
    """
    rad = math.radians(image_angle_deg)
    bearing = math.degrees(math.atan2(math.cos(rad), -math.sin(rad))) % 360.0
    return round(bearing, 2)


def run_pipeline(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    mpp: float,
    lat: float,
    zoom: int,
    tile_size: int = 256,
    tile_origin: tuple[int, int] = (0, 0),
    tile_source: str = "unknown",
    config: PipelineConfig | None = None,
    block_name: str = "",
    vineyard_name: str = "",
) -> BlockRowDetectionResult | None:
    """Execute the full row detection pipeline.

    Args:
        image_bgr: Stitched BGR image (masked to block polygon).
        mask: Binary polygon mask (uint8, 0 or 255).
        mpp: Meters per pixel.
        lat: Block centroid latitude.
        zoom: Tile zoom level.
        tile_size: Tile pixel size (256 or 512).
        tile_origin: (min_tx, min_ty) for geo-coordinate conversion.
        tile_source: Source name for metadata.
        config: Pipeline configuration (uses defaults if None).

    Returns:
        BlockRowDetectionResult or None if detection fails entirely.
    """
    if config is None:
        config = PipelineConfig()

    warnings = config.validate()
    for w in warnings:
        logger.warning("Config warning: %s", w)

    h, w_img = image_bgr.shape[:2]
    timings = StageTimings()

    logger.info(
        "Pipeline start: image=%dx%d, mpp=%.3f, source=%s, zoom=%d",
        w_img, h, mpp, tile_source, zoom,
    )

    # Start diagnostics collection
    from vinerow.debug.row_diagnostics import start_block, current as diag_current, finish_block
    diag = start_block(block_name=block_name, vineyard_name=vineyard_name)

    # ------------------------------------------------------------------
    # Stage 1: Preprocessing
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from vinerow.preprocessing.channels import preprocess_channels
    preprocessed = preprocess_channels(image_bgr, mask, mpp, config)
    timings.preprocessing = time.perf_counter() - t0
    logger.info("Stage 1 (preprocessing): %.2fs", timings.preprocessing)

    # ------------------------------------------------------------------
    # Stage 2: Coarse Orientation (2D FFT)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from vinerow.orientation.fft2d import detect as detect_fft2d
    coarse = detect_fft2d(
        image=preprocessed.fused if preprocessed.fused is not None
              else preprocessed.exg,
        mask=preprocessed.mask,
        mpp=mpp,
        lat=lat,
        zoom=zoom,
        tile_size=tile_size,
        plausible_min_m=config.min_spacing_m,
        plausible_max_m=config.max_spacing_m,
    )
    timings.orientation = time.perf_counter() - t0
    logger.info("Stage 2 (orientation): %.2fs", timings.orientation)

    if coarse is None:
        logger.error("Pipeline failed: no orientation detected")
        return None

    # ------------------------------------------------------------------
    # Stage 3: Row-Likelihood Map
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    if config.ridge_mode == "ml":
        from vinerow.ridge.ml_likelihood import compute_ml_likelihood
        likelihood_map = compute_ml_likelihood(
            preprocessed, preprocessed.mask, config,
            coarse_angle_deg=coarse.angle_deg,
        )
    elif config.ridge_mode == "ml_ensemble":
        from vinerow.ridge.likelihood import compute_row_likelihood
        from vinerow.ridge.ml_likelihood import compute_ml_likelihood
        gabor_cfg = PipelineConfig(**{**config.__dict__, "ridge_mode": "gabor"})
        gabor_map = compute_row_likelihood(preprocessed, coarse, mpp, gabor_cfg)
        ml_map = compute_ml_likelihood(
            preprocessed, preprocessed.mask, config,
            coarse_angle_deg=coarse.angle_deg,
        )
        likelihood_map = np.maximum(gabor_map, ml_map)
    else:
        from vinerow.ridge.likelihood import compute_row_likelihood
        likelihood_map = compute_row_likelihood(
            preprocessed, coarse, mpp, config,
        )
    timings.ridge = time.perf_counter() - t0
    logger.info("Stage 3 (ridge likelihood): %.2fs", timings.ridge)

    # ------------------------------------------------------------------
    # Stage 4: Local Candidate Extraction
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from vinerow.candidates.extraction import extract_candidates
    candidates, strip_centers = extract_candidates(
        likelihood_map, preprocessed.mask, coarse, mpp, config,
        luminance=preprocessed.luminance,
    )
    timings.candidates = time.perf_counter() - t0
    logger.info(
        "Stage 4 (candidates): %.2fs, %d candidates in %d strips",
        timings.candidates, len(candidates),
        len(set(c.strip_index for c in candidates)) if candidates else 0,
    )

    if len(candidates) < 2:
        logger.error("Pipeline failed: fewer than 2 candidates found")
        return None

    # ------------------------------------------------------------------
    # Stage 5: Row Tracking
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from vinerow.tracking.assignment import track_rows
    trajectories = track_rows(candidates, strip_centers, coarse, config)
    timings.tracking = time.perf_counter() - t0
    logger.info(
        "Stage 5 (tracking): %.2fs, %d tracks",
        timings.tracking, len(trajectories),
    )

    # Record tracking diagnostics
    diag = diag_current()
    if diag:
        diag.n_candidates = len(candidates)
        diag.n_strips = len(strip_centers)
        diag.n_tracks_after_tracking = len(trajectories)

    if len(trajectories) < 1:
        logger.error("Pipeline failed: no row tracks formed")
        return None

    # ------------------------------------------------------------------
    # Stage 5b: Post-Tracking Stitching
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from vinerow.tracking.stitching import stitch_trajectories
    n_strips = len(strip_centers)
    trajectories, occlusion_gaps = stitch_trajectories(
        trajectories, n_strips, coarse.spacing_px, config,
    )
    stitch_time = time.perf_counter() - t0
    timings.tracking += stitch_time  # fold into tracking time
    logger.info(
        "Stage 5b (stitching): %.2fs, %d tracks after stitch, %d occlusion gaps",
        stitch_time, len(trajectories), len(occlusion_gaps),
    )

    # Record stitching diagnostics
    diag = diag_current()
    if diag:
        diag.n_tracks_after_stitching = len(trajectories)
        diag.n_occlusion_gaps = len(occlusion_gaps)

    # ------------------------------------------------------------------
    # Stage 6: Centerline Fitting
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from vinerow.fitting.splines import fit_centerlines
    fitted_rows = fit_centerlines(
        trajectories, coarse, mpp, preprocessed.mask, tile_origin, zoom, tile_size, config,
        likelihood_map=likelihood_map,
        strip_centers=strip_centers,
        exg=preprocessed.exg,
    )
    timings.fitting = time.perf_counter() - t0
    logger.info(
        "Stage 6 (fitting): %.2fs, %d rows fitted",
        timings.fitting, len(fitted_rows),
    )

    # ------------------------------------------------------------------
    # Stage 7: Post-Processing
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    from vinerow.postprocessing.metrics import compute_block_metrics
    result = compute_block_metrics(
        fitted_rows=fitted_rows,
        coarse=coarse,
        preprocessed=preprocessed,
        likelihood_map=likelihood_map,
        candidates=candidates,
        mpp=mpp,
        tile_source=tile_source,
        zoom=zoom,
        image_size=(w_img, h),
        timings=timings,
        config=config,
        occlusion_gaps=occlusion_gaps,
    )
    timings.postprocessing = time.perf_counter() - t0
    logger.info(
        "Stage 7 (postprocessing): %.2fs",
        timings.postprocessing,
    )

    logger.info(
        "Pipeline complete: %d rows, angle=%.1f deg, spacing=%.2f m, "
        "confidence=%.2f, total=%.2fs",
        result.row_count,
        result.dominant_angle_deg,
        result.mean_spacing_m,
        result.overall_confidence,
        timings.total,
    )

    # Populate per-trajectory diagnostics (always — counters are cheap)
    diag = diag_current()
    if diag:
        from vinerow.debug.row_diagnostics import StripEvent, TrajectoryDiagnostic
        diag.n_tracks_after_fitting = len(fitted_rows)
        diag.n_rows_after_filtering = result.row_count

        # Retrieve per-strip events stashed by tracker
        strip_events: dict[int, list[StripEvent]] = getattr(diag, "_strip_events", {}) or {}

        valid_ids = {r.row_index for r in result.rows}
        for traj in trajectories:
            td = TrajectoryDiagnostic(
                track_id=traj.track_id,
                mean_perp=round(traj.mean_perp, 1),
                n_matched=traj.n_matched,
                n_strips=len(traj.candidates),
                birth_strip=traj.birth_strip,
                death_strip=traj.death_strip,
                is_stitched=traj.source_trajectory_ids is not None,
                stitch_source_ids=traj.source_trajectory_ids or [],
                is_recovered=False,
            )

            # Attach per-strip events for this trajectory
            events = strip_events.get(traj.track_id, [])
            if events:
                td.strip_events = events
                # Derive birth/death info from events
                births = [e for e in events if e.event == "birth"]
                if births:
                    td.birth_strength = births[0].strength or 0.0
                    td.birth_perp = births[0].perp_actual or 0.0
                    td.birth_source = births[0].reason or "seed"
                deaths = [e for e in events if e.event == "death"]
                if deaths:
                    td.death_strip = deaths[-1].strip_index
                    td.death_reason = deaths[-1].reason or "skip_limit"

            # Find corresponding fitted row
            fitted = next((r for r in fitted_rows if r.row_index == traj.track_id), None)
            if fitted:
                td.spline_max_curvature = fitted.curvature_max_deg_per_m
                td.confidence_final = fitted.confidence
                td.length_m = fitted.length_m
                td.passed_filters = fitted.row_index in valid_ids
                if not td.passed_filters:
                    if fitted.confidence < config.min_row_confidence:
                        td.filter_reason = "low_confidence"
                    else:
                        td.filter_reason = "too_short"
            else:
                td.passed_filters = False
                td.filter_reason = "fitting_failed"

            diag.trajectories.append(td)

        # Save diagnostics JSON only when debug artifacts are enabled
        if config.save_debug_artifacts:
            finish_block()
        else:
            # Reset without saving
            from vinerow.debug.row_diagnostics import _current
            import vinerow.debug.row_diagnostics as _diag_mod
            _diag_mod._current = None

    return result
