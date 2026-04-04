"""
Centerline fitting via smoothing splines.

Converts noisy row trajectories (sequences of candidate points) into
smooth geometric centerlines. Straight rows get straight lines; curved
rows get smooth curves. Smoothing controls noise suppression.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.interpolate import UnivariateSpline

from vinerow.acquisition.geo_utils import pixel_to_lnglat
from vinerow.config import PipelineConfig
from vinerow.types import CoarseOrientation, FittedRow, RowSegment, RowTrajectory

logger = logging.getLogger(__name__)


def _trim_endpoints(
    centerline_px: list[tuple[float, float]],
    lk_profile: list[float],
    mpp: float,
    block_median_lk: float,
    config: PipelineConfig,
) -> tuple[list[tuple[float, float]], list[float], float]:
    """Trim centerline endpoints where likelihood evidence is weak.

    Walks inward from each end. Trims points where likelihood < threshold
    for a sustained run. Uses a relative threshold based on the block's
    median likelihood (protects faint blocks).
    """
    if len(centerline_px) < 4 or lk_profile is None:
        length = sum(
            math.hypot(centerline_px[i][0] - centerline_px[i-1][0],
                       centerline_px[i][1] - centerline_px[i-1][1])
            for i in range(1, len(centerline_px))
        ) * mpp
        return centerline_px, lk_profile, length

    threshold = block_median_lk * config.endpoint_trim_likelihood_ratio
    min_run = config.endpoint_trim_min_run

    # Trim from the start
    start = 0
    run = 0
    for i, lk in enumerate(lk_profile):
        if lk < threshold:
            run += 1
        else:
            if run >= min_run:
                start = i
            break
    else:
        start = 0

    # Trim from the end
    end = len(lk_profile)
    run = 0
    for i in range(len(lk_profile) - 1, -1, -1):
        if lk_profile[i] < threshold:
            run += 1
        else:
            if run >= min_run:
                end = i + 1
            break

    # Safety: keep at least 50% of original
    min_keep = max(4, len(centerline_px) // 2)
    if end - start < min_keep:
        start = 0
        end = len(centerline_px)

    trimmed_cl = centerline_px[start:end]
    trimmed_lk = lk_profile[start:end]

    length = sum(
        math.hypot(trimmed_cl[i][0] - trimmed_cl[i-1][0],
                   trimmed_cl[i][1] - trimmed_cl[i-1][1])
        for i in range(1, len(trimmed_cl))
    ) * mpp

    return trimmed_cl, trimmed_lk, length


def _map_strips_to_centerline(
    strip_centers: list[float],
    cl_along: list[float],
) -> dict[int, int]:
    """Map strip indices to nearest centerline point indices.

    Uses the fitting stage's own along-axis positions (cl_along), not a
    reconstructed projection. Handles reversed centerline direction.
    """
    mapping: dict[int, int] = {}
    n_cl = len(cl_along)
    if n_cl == 0:
        return mapping
    for s, sc in enumerate(strip_centers):
        best_idx = 0
        best_dist = abs(cl_along[0] - sc)
        for ci in range(1, n_cl):
            d = abs(cl_along[ci] - sc)
            if d < best_dist:
                best_dist = d
                best_idx = ci
        mapping[s] = best_idx
    return mapping


def _hysteresis_segments(
    supported: list[bool],
    first_strip: int,
    last_strip: int,
    min_unsup: int,
    min_sup: int,
) -> list[RowSegment]:
    """Convert a boolean support array into segments with hysteresis debouncing."""
    segments: list[RowSegment] = []
    in_visible = supported[first_strip]
    seg_start = first_strip
    run = 0

    for s in range(first_strip, last_strip + 1):
        if in_visible:
            if not supported[s]:
                run += 1
                if run >= min_unsup:
                    gap_start = s - run + 1
                    if gap_start > seg_start:
                        segments.append(RowSegment(
                            start_strip=seg_start, end_strip=gap_start - 1, is_visible=True))
                    seg_start = gap_start
                    in_visible = False
                    run = 0
            else:
                run = 0
        else:
            if supported[s]:
                run += 1
                if run >= min_sup:
                    vis_start = s - run + 1
                    if vis_start > seg_start:
                        segments.append(RowSegment(
                            start_strip=seg_start, end_strip=vis_start - 1, is_visible=False))
                    seg_start = vis_start
                    in_visible = True
                    run = 0
            else:
                run = 0

    if seg_start <= last_strip:
        segments.append(RowSegment(
            start_strip=seg_start, end_strip=last_strip, is_visible=in_visible))

    return segments


def _detect_support_gaps(
    centerline_px: list[tuple[float, float]],
    cl_along: list[float],
    lk_profile: list[float] | None,
    trajectory: RowTrajectory,
    strip_centers: list[float],
    coarse: CoarseOrientation,
    mpp: float,
    block_median_lk: float,
    ref_x: float,
    ref_y: float,
    perp_x: float,
    perp_y: float,
    config: PipelineConfig,
) -> list[RowSegment]:
    """Detect unsupported spans using strip-level candidate support + hysteresis."""
    n_strips = len(trajectory.candidates)
    n_cl = len(centerline_px)

    if n_cl < 4 or n_strips < 4:
        return [RowSegment(start_strip=0, end_strip=n_strips - 1, is_visible=True,
                           start_point_idx=0, end_point_idx=n_cl - 1)]

    # Strip-to-centerline mapping (uses fitting's own along-axis frame)
    strip_to_cl = _map_strips_to_centerline(strip_centers, cl_along)

    # Row-relative strength threshold
    matched_strengths = [c.strength for c in trajectory.candidates if c is not None]
    row_median_strength = float(np.median(matched_strengths)) if matched_strengths else 0.5
    strength_threshold = max(
        config.gap_min_candidate_strength,
        row_median_strength * config.gap_strength_ratio,
    )

    # Per-strip support: candidate present + strong enough + close to fitted centerline
    strip_supported = [False] * n_strips
    n_demoted_strength = 0
    n_demoted_residual = 0
    for s in range(n_strips):
        c = trajectory.candidates[s]
        if c is None:
            continue
        if c.strength < strength_threshold:
            n_demoted_strength += 1
            continue
        # Residual check: candidate perp vs fitted centerline perp at this strip
        cl_idx = strip_to_cl.get(s, -1)
        if cl_idx >= 0 and cl_idx < n_cl:
            fitted_perp = (centerline_px[cl_idx][0] - ref_x) * perp_x + \
                          (centerline_px[cl_idx][1] - ref_y) * perp_y
            cand_perp = (c.x - ref_x) * perp_x + (c.y - ref_y) * perp_y
            residual = abs(fitted_perp - cand_perp)
            if residual > config.gap_max_candidate_residual_factor * coarse.spacing_px:
                n_demoted_residual += 1
                continue
        strip_supported[s] = True

    logger.debug(
        "Gap detect row %d: median_str=%.3f, thresh=%.3f, "
        "demoted_str=%d, demoted_res=%d, supported=%d/%d",
        trajectory.track_id, row_median_strength, strength_threshold,
        n_demoted_strength, n_demoted_residual,
        sum(strip_supported), n_strips,
    )

    # Active strip range
    first_sup = next((s for s in range(n_strips) if strip_supported[s]), -1)
    last_sup = next((s for s in range(n_strips - 1, -1, -1) if strip_supported[s]), -1)
    if first_sup < 0:
        return [RowSegment(start_strip=0, end_strip=n_strips - 1, is_visible=False,
                           start_point_idx=0, end_point_idx=n_cl - 1)]

    # Likelihood dilation: extend supported edges by up to N strips
    if lk_profile is not None and config.gap_likelihood_dilation_strips > 0:
        lk_thresh = config.gap_likelihood_threshold * block_median_lk
        dil = config.gap_likelihood_dilation_strips
        dilated = list(strip_supported)
        for s in range(first_sup, last_sup + 1):
            if dilated[s]:
                continue
            near_supported = any(
                dilated[ns]
                for ns in range(max(first_sup, s - dil), min(last_sup + 1, s + dil + 1))
                if ns != s
            )
            if not near_supported:
                continue
            cl_idx = strip_to_cl.get(s, -1)
            if 0 <= cl_idx < len(lk_profile) and lk_profile[cl_idx] >= lk_thresh:
                dilated[s] = True
        strip_supported = dilated

    # Hysteresis segmentation
    segments = _hysteresis_segments(
        strip_supported, first_sup, last_sup,
        config.gap_min_consecutive_unsupported,
        config.gap_min_consecutive_supported,
    )

    # Map strip ranges to centerline point indices
    for seg in segments:
        seg.start_point_idx = strip_to_cl.get(seg.start_strip, 0)
        seg.end_point_idx = strip_to_cl.get(seg.end_strip, n_cl - 1)
        # Ensure point indices don't invert
        if seg.start_point_idx > seg.end_point_idx:
            seg.start_point_idx, seg.end_point_idx = seg.end_point_idx, seg.start_point_idx

    # Discard very short visible segments
    min_vis_m = config.gap_min_visible_segment_length_m
    for seg in segments:
        if seg.is_visible and seg.start_point_idx >= 0 and seg.end_point_idx > seg.start_point_idx:
            seg_len = sum(
                math.hypot(centerline_px[j][0] - centerline_px[j-1][0],
                           centerline_px[j][1] - centerline_px[j-1][1])
                for j in range(seg.start_point_idx + 1, seg.end_point_idx + 1)
            ) * mpp
            if seg_len < min_vis_m:
                seg.is_visible = False

    # Merge adjacent same-type segments
    if len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.is_visible == merged[-1].is_visible:
                merged[-1] = RowSegment(
                    start_strip=merged[-1].start_strip, end_strip=seg.end_strip,
                    is_visible=seg.is_visible,
                    start_point_idx=merged[-1].start_point_idx, end_point_idx=seg.end_point_idx)
            else:
                merged.append(seg)
        segments = merged

    return segments


def _fit_single_row(
    trajectory: RowTrajectory,
    coarse: CoarseOrientation,
    mpp: float,
    mask: np.ndarray,
    config: PipelineConfig,
    likelihood_map: np.ndarray | None = None,
    block_median_lk: float = 0.0,
    strip_centers: list[float] | None = None,
) -> FittedRow | None:
    """Fit a smooth centerline to a single row trajectory."""
    matched = [(c.strip_index, c) for c in trajectory.candidates if c is not None]
    if not matched:
        return None

    angle_rad = math.radians(coarse.angle_deg)
    row_dx = math.cos(angle_rad)
    row_dy = math.sin(angle_rad)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)

    # Sort by strip index (along-row order)
    matched.sort(key=lambda m: m[0])

    if len(matched) == 1:
        # Single point: create a short line segment
        c = matched[0][1]
        # Extend in both directions by a few spacing widths
        extent = coarse.spacing_px * 3
        p1 = (c.x - row_dx * extent, c.y - row_dy * extent)
        p2 = (c.x + row_dx * extent, c.y + row_dy * extent)
        return FittedRow(
            row_index=trajectory.track_id,
            centerline_px=[p1, p2],
            confidence=round(c.strength, 3),
            length_m=round(2 * extent * mpp, 2),
        )

    if len(matched) == 2:
        # Two points: straight line between them
        c1, c2 = matched[0][1], matched[1][1]
        return FittedRow(
            row_index=trajectory.track_id,
            centerline_px=[(c1.x, c1.y), (c2.x, c2.y)],
            confidence=round((c1.strength + c2.strength) / 2, 3),
            length_m=round(math.hypot(c2.x - c1.x, c2.y - c1.y) * mpp, 2),
        )

    # 3+ points: fit a smoothing spline to the (along_position, perp_position) track
    # Extract along-row and perpendicular coordinates relative to first point
    ref_x, ref_y = matched[0][1].x, matched[0][1].y

    along_positions = []
    perp_positions = []
    strengths = []
    for _, c in matched:
        dx = c.x - ref_x
        dy = c.y - ref_y
        along = dx * row_dx + dy * row_dy
        perp = dx * perp_x + dy * perp_y
        along_positions.append(along)
        perp_positions.append(perp)
        strengths.append(c.strength)

    along_arr = np.array(along_positions)
    perp_arr = np.array(perp_positions)
    strength_arr = np.array(strengths)

    # Ensure along positions are monotonically increasing (required by UnivariateSpline)
    sort_idx = np.argsort(along_arr)
    along_arr = along_arr[sort_idx]
    perp_arr = perp_arr[sort_idx]
    strength_arr = strength_arr[sort_idx]

    # Remove duplicates in along position
    unique_mask = np.diff(along_arr, prepend=-1e9) > 0.5
    along_arr = along_arr[unique_mask]
    perp_arr = perp_arr[unique_mask]
    strength_arr = strength_arr[unique_mask]

    if len(along_arr) < 3:
        # Degenerated after cleanup — straight line
        p1 = (matched[0][1].x, matched[0][1].y)
        p2 = (matched[-1][1].x, matched[-1][1].y)
        conf = float(np.mean(strength_arr)) if len(strength_arr) > 0 else 0.5
        return FittedRow(
            row_index=trajectory.track_id,
            centerline_px=[p1, p2],
            confidence=round(conf, 3),
            length_m=round(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) * mpp, 2),
        )

    # Fit spline with smoothing
    # s = n * (allowed_deviation_px)^2 — smoothing factor
    allowed_dev_px = config.spline_smoothing_m / mpp
    s_factor = len(along_arr) * (allowed_dev_px ** 2)

    try:
        # Use weights proportional to candidate strength
        weights = np.maximum(strength_arr, 0.1)  # don't let weights go to zero
        spline = UnivariateSpline(along_arr, perp_arr, w=weights, s=s_factor, k=3)
    except Exception:
        # Spline fitting failed — fall back to linear
        logger.debug("Spline fitting failed for track %d, using linear", trajectory.track_id)
        p1 = (matched[0][1].x, matched[0][1].y)
        p2 = (matched[-1][1].x, matched[-1][1].y)
        return FittedRow(
            row_index=trajectory.track_id,
            centerline_px=[p1, p2],
            confidence=round(float(np.mean(strength_arr)), 3),
            length_m=round(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) * mpp, 2),
        )

    # Sample the spline at regular intervals
    sample_interval = config.centerline_sample_interval_px
    along_min, along_max = float(along_arr[0]), float(along_arr[-1])
    n_samples = max(2, int((along_max - along_min) / sample_interval) + 1)
    sample_along = np.linspace(along_min, along_max, n_samples)
    sample_perp = spline(sample_along)

    # Clamp spline output to prevent wild extrapolation
    perp_range = float(np.ptp(perp_arr))
    perp_center = float(np.mean(perp_arr))
    max_deviation = max(perp_range * 3.0, coarse.spacing_px)
    sample_perp = np.clip(sample_perp, perp_center - max_deviation, perp_center + max_deviation)

    # Convert back to image coordinates
    h_img, w_img = mask.shape[:2]
    centerline_px = []
    for a, p in zip(sample_along, sample_perp):
        x = ref_x + row_dx * a + perp_x * p
        y = ref_y + row_dy * a + perp_y * p
        # Clamp to image bounds with generous margin
        x = max(-w_img, min(2 * w_img, float(x)))
        y = max(-h_img, min(2 * h_img, float(y)))
        centerline_px.append((round(x, 1), round(y, 1)))

    # Compute curvature: max angular change per meter
    curvature_max = 0.0
    if len(centerline_px) >= 3:
        for i in range(1, len(centerline_px) - 1):
            dx1 = centerline_px[i][0] - centerline_px[i-1][0]
            dy1 = centerline_px[i][1] - centerline_px[i-1][1]
            dx2 = centerline_px[i+1][0] - centerline_px[i][0]
            dy2 = centerline_px[i+1][1] - centerline_px[i][1]
            a1 = math.atan2(dy1, dx1)
            a2 = math.atan2(dy2, dx2)
            angle_change = abs(a2 - a1)
            if angle_change > math.pi:
                angle_change = 2 * math.pi - angle_change
            segment_len = math.hypot(dx1, dy1) * mpp
            if segment_len > 0.1:
                curvature = math.degrees(angle_change) / segment_len
                curvature_max = max(curvature_max, curvature)

    # Length
    total_length_px = 0.0
    for i in range(1, len(centerline_px)):
        total_length_px += math.hypot(
            centerline_px[i][0] - centerline_px[i-1][0],
            centerline_px[i][1] - centerline_px[i-1][1],
        )
    length_m = total_length_px * mpp

    confidence = float(np.mean(strength_arr))
    # Penalize confidence for short tracks (few matched strips)
    # Use total strip count as denominator — a row doesn't need to span
    # the whole block to be valid (block shape may be narrow at ends)
    total_strips = len([c for c in trajectory.candidates])  # total strip slots
    completeness = trajectory.n_matched / max(total_strips * 0.5, 1)
    confidence *= min(completeness, 1.0)

    # Penalize confidence for high curvature (soft limit, not hard rejection)
    if curvature_max > 0 and config.curvature_soft_limit > 0:
        curvature_penalty = max(0.3, 1.0 - curvature_max / config.curvature_soft_limit)
        confidence *= curvature_penalty

    # Build segment metadata from trajectory (if available from stitching)
    segments = None
    if trajectory.segments:
        segments = list(trajectory.segments)
        # Add inferred gap segments between visible segments
        all_segs: list[RowSegment] = []
        for idx, seg in enumerate(segments):
            all_segs.append(seg)
            if idx < len(segments) - 1:
                next_seg = segments[idx + 1]
                gap_start = seg.end_strip + 1
                gap_end = next_seg.start_strip - 1
                if gap_end >= gap_start:
                    all_segs.append(RowSegment(
                        start_strip=gap_start,
                        end_strip=gap_end,
                        is_visible=False,
                    ))
        segments = all_segs

    # Sample likelihood along centerline for evidence profile
    lk_profile = None
    if likelihood_map is not None and len(centerline_px) >= 2:
        lh, lw = likelihood_map.shape[:2]
        lk_profile = []
        for px, py in centerline_px:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < lw and 0 <= iy < lh:
                lk_profile.append(round(float(likelihood_map[iy, ix]), 3))
            else:
                lk_profile.append(0.0)

    # Trim endpoints where likelihood evidence is weak
    if lk_profile is not None and block_median_lk > 0:
        centerline_px, lk_profile, length_m = _trim_endpoints(
            centerline_px, lk_profile, mpp, block_median_lk, config,
        )

    # Post-fit gap detection: identify unsupported spans via strip-level support
    # Always run when strip_centers available — ensures point indices are set
    if strip_centers is not None and len(centerline_px) >= 4:
        # Build cl_along from the same frame used in fitting
        cl_along = [
            ((px - ref_x) * row_dx + (py - ref_y) * row_dy)
            for px, py in centerline_px
        ]
        support_segments = _detect_support_gaps(
            centerline_px, cl_along, lk_profile, trajectory, strip_centers,
            coarse, mpp, block_median_lk,
            ref_x, ref_y, perp_x, perp_y, config,
        )
        segments = support_segments

    return FittedRow(
        row_index=trajectory.track_id,
        centerline_px=centerline_px,
        confidence=round(confidence, 3),
        length_m=round(length_m, 2),
        curvature_max_deg_per_m=round(curvature_max, 4),
        segments=segments,
        likelihood_profile=lk_profile,
    )


def fit_centerlines(
    trajectories: list[RowTrajectory],
    coarse: CoarseOrientation,
    mpp: float,
    mask: np.ndarray,
    tile_origin: tuple[int, int],
    zoom: int,
    tile_size: int,
    config: PipelineConfig,
    likelihood_map: np.ndarray | None = None,
    strip_centers: list[float] | None = None,
) -> list[FittedRow]:
    """Fit smooth centerlines to all row trajectories.

    Args:
        trajectories: Row trajectories from tracking stage.
        coarse: Coarse orientation estimate.
        mpp: Meters per pixel.
        mask: Binary polygon mask.
        tile_origin: For geographic coordinate conversion.
        zoom: Tile zoom level.
        tile_size: Tile pixel size.
        config: Pipeline configuration.

    Returns:
        List of FittedRow, sorted by perpendicular position (row order).
    """
    # Compute block-level median likelihood for relative endpoint trimming
    block_median_lk = 0.0
    if likelihood_map is not None:
        mask_pixels = mask > 0
        if mask_pixels.any():
            block_median_lk = float(np.median(likelihood_map[mask_pixels]))

    fitted: list[FittedRow] = []

    for traj in trajectories:
        row = _fit_single_row(traj, coarse, mpp, mask, config, likelihood_map, block_median_lk, strip_centers)
        if row is None:
            continue
        fitted.append(row)

    # Sort by the y-intercept-like metric (mean perpendicular position)
    # Use the midpoint of each centerline projected onto the perpendicular axis
    angle_rad = math.radians(coarse.angle_deg)
    perp_x = -math.sin(angle_rad)
    perp_y = math.cos(angle_rad)

    def _mean_perp(row: FittedRow) -> float:
        if not row.centerline_px:
            return 0.0
        mid_idx = len(row.centerline_px) // 2
        mx, my = row.centerline_px[mid_idx]
        return mx * perp_x + my * perp_y

    fitted.sort(key=_mean_perp)

    # Re-index and compute spacing to previous
    for i, row in enumerate(fitted):
        row.row_index = i
        if i > 0:
            prev = fitted[i - 1]
            # Spacing: perpendicular distance between centerlines at their midpoints
            perp_curr = _mean_perp(row)
            perp_prev = _mean_perp(prev)
            spacing = abs(perp_curr - perp_prev) * mpp
            row.spacing_to_prev_m = round(spacing, 3)

    # Convert to geographic coordinates
    for row in fitted:
        geo_line = []
        for px, py in row.centerline_px:
            lng, lat = pixel_to_lnglat(px, py, tile_origin, zoom, tile_size)
            geo_line.append((round(lng, 7), round(lat, 7)))
        row.centerline_geo = geo_line

    logger.info("Fitted %d centerlines (from %d trajectories)", len(fitted), len(trajectories))

    return fitted
