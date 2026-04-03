"""
Post-tracking segment stitching.

Reconnects broken row trajectories that were split by signal dropout or
internal occlusions (buildings, frost fans, service pads). Operates on
finished trajectories without modifying the core tracker.

Two stitching strategies:
1. Group-aware: for occlusion bands where multiple adjacent rows disappear
   together, solved as a group correspondence problem.
2. Pairwise: for isolated single-row dropouts, solved greedily with
   strict consistency checks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from vinerow.config import PipelineConfig
from vinerow.types import OcclusionGap, RowCandidate, RowSegment, RowTrajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matched_strip_range(traj: RowTrajectory) -> tuple[int, int]:
    """Return (first_matched_strip, last_matched_strip) for a trajectory."""
    first = last = -1
    for i, c in enumerate(traj.candidates):
        if c is not None:
            if first == -1:
                first = i
            last = i
    return first, last


def _end_slope(traj: RowTrajectory, side: str, n_points: int = 3) -> float | None:
    """Estimate local slope (perp change per strip) at one end of a trajectory.

    Args:
        side: "left" (low strip indices) or "right" (high strip indices)
        n_points: number of matched candidates to use

    Returns:
        Slope in perp_position units per strip, or None if insufficient data.
    """
    matched = [(i, c) for i, c in enumerate(traj.candidates) if c is not None]
    if len(matched) < 2:
        return None

    if side == "right":
        pts = matched[-n_points:]
    else:
        pts = matched[:n_points]

    if len(pts) < 2:
        return None

    strips = [p[0] for p in pts]
    perps = [p[1].perp_position for p in pts]
    strip_span = strips[-1] - strips[0]
    if strip_span == 0:
        return 0.0
    return (perps[-1] - perps[0]) / strip_span


def _extrapolate_perp(traj: RowTrajectory, target_strip: int, side: str) -> float | None:
    """Extrapolate perpendicular position to a target strip.

    Uses the last few matched candidates on the given side to estimate
    where the row would be at target_strip.
    """
    slope = _end_slope(traj, side)
    matched = [(i, c) for i, c in enumerate(traj.candidates) if c is not None]
    if not matched:
        return None

    if side == "right":
        anchor_strip, anchor_cand = matched[-1]
    else:
        anchor_strip, anchor_cand = matched[0]

    if slope is None:
        return anchor_cand.perp_position

    return anchor_cand.perp_position + slope * (target_strip - anchor_strip)


def _compute_segments(traj: RowTrajectory) -> list[RowSegment]:
    """Compute visible segments from a trajectory's candidate array."""
    segments: list[RowSegment] = []
    n = len(traj.candidates)
    in_visible = False
    seg_start = 0

    for i in range(n):
        if traj.candidates[i] is not None:
            if not in_visible:
                seg_start = i
                in_visible = True
        else:
            if in_visible:
                segments.append(RowSegment(start_strip=seg_start, end_strip=i - 1, is_visible=True))
                in_visible = False

    if in_visible:
        segments.append(RowSegment(start_strip=seg_start, end_strip=n - 1, is_visible=True))

    return segments


# ---------------------------------------------------------------------------
# Presence matrix and occlusion detection
# ---------------------------------------------------------------------------


def _build_presence_matrix(
    trajectories: list[RowTrajectory],
    n_strips: int,
) -> np.ndarray:
    """Build a boolean matrix: rows × strips, True where candidate is matched."""
    n_rows = len(trajectories)
    matrix = np.zeros((n_rows, n_strips), dtype=bool)
    for r, traj in enumerate(trajectories):
        for s, c in enumerate(traj.candidates):
            if c is not None:
                matrix[r, s] = True
    return matrix


@dataclass
class _OcclusionBand:
    """A detected band of missing rows across a strip range."""
    row_start: int      # first row index in the band
    row_end: int        # last row index (inclusive)
    strip_start: int    # first strip where the band is missing
    strip_end: int      # last strip where the band is missing (inclusive)


def _detect_occlusion_bands(
    presence: np.ndarray,
    min_rows: int,
) -> list[_OcclusionBand]:
    """Detect rectangular-ish bands of missing values in the presence matrix.

    Looks for groups of ≥min_rows adjacent rows that are all missing over
    roughly the same strip range, with valid data on both sides.
    """
    n_rows, n_strips = presence.shape
    bands: list[_OcclusionBand] = []

    # For each row, find contiguous interior gaps (not at edges)
    row_gaps: list[list[tuple[int, int]]] = []  # per row: list of (gap_start, gap_end)
    for r in range(n_rows):
        gaps = []
        first_match = -1
        last_match = -1
        for s in range(n_strips):
            if presence[r, s]:
                if first_match == -1:
                    first_match = s
                last_match = s

        if first_match == -1:
            row_gaps.append([])
            continue

        # Find interior gaps (between first and last match)
        in_gap = False
        gap_start = 0
        for s in range(first_match, last_match + 1):
            if not presence[r, s]:
                if not in_gap:
                    gap_start = s
                    in_gap = True
            else:
                if in_gap:
                    gaps.append((gap_start, s - 1))
                    in_gap = False
        if in_gap:
            gaps.append((gap_start, last_match))

        row_gaps.append(gaps)

    # Note: we intentionally do NOT check for "edge gaps" (where a trajectory
    # doesn't extend as far as its neighbour). Edge gaps create false groupings
    # when the tracker splits a row into two separate trajectories. Edge gaps
    # are better handled by pairwise stitching.

    # Cluster overlapping gaps across adjacent rows into bands
    # Sort all gaps by (row, strip_start)
    all_gaps: list[tuple[int, int, int]] = []  # (row_idx, gap_start, gap_end)
    for r, gaps in enumerate(row_gaps):
        for gs, ge in gaps:
            all_gaps.append((r, gs, ge))

    if not all_gaps:
        return bands

    all_gaps.sort(key=lambda g: (g[1], g[0]))  # sort by strip_start, then row

    # Group gaps that overlap in strip range and are in adjacent rows
    used = set()
    for i, (r1, gs1, ge1) in enumerate(all_gaps):
        if i in used:
            continue

        group_rows = {r1}
        group_strip_start = gs1
        group_strip_end = ge1

        # Expand: find adjacent rows with overlapping gaps
        changed = True
        while changed:
            changed = False
            for j, (r2, gs2, ge2) in enumerate(all_gaps):
                if j in used or j == i:
                    continue
                if r2 in group_rows:
                    continue

                # Check adjacency: row must be within 1 of an existing group row
                adjacent = any(abs(r2 - gr) <= 1 for gr in group_rows)
                if not adjacent:
                    continue

                # Check strip overlap
                overlap_start = max(group_strip_start, gs2)
                overlap_end = min(group_strip_end, ge2)
                # Require significant overlap (>50% of smaller gap)
                smaller_len = min(ge2 - gs2, group_strip_end - group_strip_start) + 1
                overlap_len = max(0, overlap_end - overlap_start + 1)
                if overlap_len >= smaller_len * 0.5:
                    group_rows.add(r2)
                    group_strip_start = min(group_strip_start, gs2)
                    group_strip_end = max(group_strip_end, ge2)
                    used.add(j)
                    changed = True

        used.add(i)

        if len(group_rows) >= min_rows:
            sorted_rows = sorted(group_rows)
            bands.append(_OcclusionBand(
                row_start=sorted_rows[0],
                row_end=sorted_rows[-1],
                strip_start=group_strip_start,
                strip_end=group_strip_end,
            ))

    return bands


# ---------------------------------------------------------------------------
# Stitching logic
# ---------------------------------------------------------------------------


def _stitch_pair(
    left: RowTrajectory,
    right: RowTrajectory,
    n_strips: int,
) -> RowTrajectory:
    """Merge two non-overlapping trajectories into one."""
    merged_candidates: list[RowCandidate | None] = [None] * n_strips
    source_ids = []

    for s, c in enumerate(left.candidates):
        if c is not None:
            merged_candidates[s] = c
    for s, c in enumerate(right.candidates):
        if c is not None:
            merged_candidates[s] = c

    if left.source_trajectory_ids:
        source_ids.extend(left.source_trajectory_ids)
    else:
        source_ids.append(left.track_id)

    if right.source_trajectory_ids:
        source_ids.extend(right.source_trajectory_ids)
    else:
        source_ids.append(right.track_id)

    first_strip, last_strip = _matched_strip_range(
        RowTrajectory(track_id=0, candidates=merged_candidates, birth_strip=0, death_strip=0)
    )

    merged = RowTrajectory(
        track_id=left.track_id,
        candidates=merged_candidates,
        birth_strip=first_strip,
        death_strip=last_strip,
        source_trajectory_ids=source_ids,
    )
    return merged


def _join_angle_deg(left: RowTrajectory, right: RowTrajectory) -> float | None:
    """Compute direction change (degrees) at the join between two trajectories.

    Uses last 2 matched candidates of left and first 2 of right to estimate
    the direction vectors meeting at the gap.
    """
    left_pts = [(c.x, c.y) for c in left.candidates if c is not None]
    right_pts = [(c.x, c.y) for c in right.candidates if c is not None]
    if len(left_pts) < 2 or len(right_pts) < 2:
        return None
    ldx = left_pts[-1][0] - left_pts[-2][0]
    ldy = left_pts[-1][1] - left_pts[-2][1]
    rdx = right_pts[1][0] - right_pts[0][0]
    rdy = right_pts[1][1] - right_pts[0][1]
    dot = ldx * rdx + ldy * rdy
    cross = ldx * rdy - ldy * rdx
    return abs(math.degrees(math.atan2(cross, dot)))


def _stitch_score(
    left: RowTrajectory,
    right: RowTrajectory,
    spacing_px: float,
    config: PipelineConfig,
) -> float:
    """Score a potential stitch between two trajectories. Lower = better.

    Returns float('inf') if the pair should not be stitched.
    """
    _, left_last = _matched_strip_range(left)
    right_first, _ = _matched_strip_range(right)

    # Must not overlap
    if left_last >= right_first:
        return float("inf")

    gap = right_first - left_last - 1
    if gap > config.max_stitch_gap_strips:
        return float("inf")

    # Perpendicular offset at gap midpoint
    gap_mid = (left_last + right_first) / 2.0
    left_perp = _extrapolate_perp(left, int(gap_mid), "right")
    right_perp = _extrapolate_perp(right, int(gap_mid), "left")
    if left_perp is None or right_perp is None:
        return float("inf")

    perp_diff = abs(left_perp - right_perp)
    if perp_diff > config.stitch_perp_tolerance * spacing_px:
        return float("inf")

    # Slope consistency
    left_slope = _end_slope(left, "right")
    right_slope = _end_slope(right, "left")
    if left_slope is not None and right_slope is not None:
        slope_diff = abs(left_slope - right_slope)
        if slope_diff > config.stitch_slope_tolerance:
            return float("inf")
    else:
        slope_diff = 0.0

    # Strength penalty: penalize weak-to-weak stitches
    left_strength = left.mean_strength
    right_strength = right.mean_strength
    strength_penalty = 0.3 * (2.0 - left_strength - right_strength)

    # Combined score: weighted perp offset + slope difference + gap + strength
    score = (
        perp_diff / spacing_px          # normalized perp offset (0-0.5)
        + 0.5 * slope_diff              # slope inconsistency
        + 0.1 * gap / config.max_stitch_gap_strips  # small penalty for larger gaps
        + strength_penalty              # weak segments penalized
    )

    logger.debug(
        "Stitch score: traj %d→%d: perp=%.3f slope=%.3f gap=%.3f strength=%.3f total=%.3f",
        left.track_id, right.track_id,
        perp_diff / spacing_px, 0.5 * slope_diff,
        0.1 * gap / config.max_stitch_gap_strips,
        strength_penalty, score,
    )
    return score


def _group_stitch(
    band: _OcclusionBand,
    trajectories: list[RowTrajectory],
    spacing_px: float,
    config: PipelineConfig,
) -> list[tuple[int, int]]:
    """Solve group correspondence for an occlusion band.

    Returns list of (left_traj_idx, right_traj_idx) pairs to merge.
    """
    # Collect trajectory stubs that end before or start after the band
    left_stubs: list[int] = []   # traj indices that end just before the band
    right_stubs: list[int] = []  # traj indices that start just after the band

    for r in range(band.row_start, band.row_end + 1):
        traj = trajectories[r]
        _, last = _matched_strip_range(traj)

        if last <= band.strip_end:
            # This trajectory ends before/within the band — it's a left stub
            # But only if it has data before the band
            if last >= 0 and last <= band.strip_end:
                left_stubs.append(r)

    # Look for trajectories that start after the band but aren't in the
    # sorted trajectory list yet (they'd be separate trajectories)
    # Actually, we need to find right stubs: trajectories whose first match
    # is after the band
    for r in range(len(trajectories)):
        first, _ = _matched_strip_range(trajectories[r])
        if first > band.strip_start and r not in left_stubs:
            # Check if this trajectory's perp position is in the band's row range
            traj_perp = trajectories[r].mean_perp
            band_perps = [trajectories[br].mean_perp
                          for br in range(band.row_start, band.row_end + 1)]
            if band_perps:
                min_perp = min(band_perps) - spacing_px
                max_perp = max(band_perps) + spacing_px
                if min_perp <= traj_perp <= max_perp:
                    right_stubs.append(r)

    if not left_stubs or not right_stubs:
        return []

    n_left = len(left_stubs)
    n_right = len(right_stubs)

    # Build cost matrix for group assignment
    size = max(n_left, n_right)
    cost = np.full((size, size), 1e9, dtype=np.float64)

    for i, li in enumerate(left_stubs):
        for j, rj in enumerate(right_stubs):
            score = _stitch_score(trajectories[li], trajectories[rj], spacing_px, config)
            # Replace inf with large finite value — scipy rejects all-inf rows
            cost[i, j] = min(score, 1e8) if score != float("inf") else 1e8

    if not np.isfinite(cost).all():
        logger.warning("Group stitch cost matrix has non-finite values, clamping")
        cost = np.clip(cost, 0, 1e8)

    # Solve with Hungarian (appropriate here — it's a small N×M problem
    # and we enforce order via the ambiguity check below)
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs: list[tuple[int, int]] = []
    scores: list[float] = []

    for ri, ci in zip(row_ind, col_ind):
        if ri < n_left and ci < n_right and cost[ri, ci] < 1e8:
            pairs.append((left_stubs[ri], right_stubs[ci]))
            scores.append(cost[ri, ci])

    if not pairs:
        return []

    # Ambiguity check: for each accepted pair, ensure the best match is
    # significantly better than the second-best alternative
    filtered_pairs: list[tuple[int, int]] = []
    for idx, ((li_idx, ri_idx), best_score) in enumerate(zip(pairs, scores)):
        # Find the original matrix indices
        i = left_stubs.index(li_idx)
        j = right_stubs.index(ri_idx)

        # Second-best for this left stub (across all right candidates)
        row_costs = sorted([cost[i, jj] for jj in range(n_right) if jj != j])
        second_best = row_costs[0] if row_costs else float("inf")

        if second_best == float("inf") or best_score == 0:
            # No ambiguity — only one viable option
            filtered_pairs.append((li_idx, ri_idx))
        elif second_best / max(best_score, 1e-6) >= config.stitch_ambiguity_ratio:
            filtered_pairs.append((li_idx, ri_idx))
        else:
            logger.debug(
                "Rejected ambiguous stitch: traj %d→%d (score=%.3f, 2nd=%.3f, ratio=%.2f)",
                li_idx, ri_idx, best_score, second_best,
                second_best / max(best_score, 1e-6),
            )

    # Enforce row order: reject pairs that would cross
    filtered_pairs.sort(key=lambda p: p[0])
    order_safe: list[tuple[int, int]] = []
    last_right = -1
    for li, ri in filtered_pairs:
        if ri > last_right:
            order_safe.append((li, ri))
            last_right = ri
        else:
            logger.debug(
                "Rejected order-violating stitch: traj %d→%d (would cross previous)",
                li, ri,
            )

    # Reject pairs with excessive direction change at join
    angle_safe: list[tuple[int, int]] = []
    for li, ri in order_safe:
        angle = _join_angle_deg(trajectories[li], trajectories[ri])
        if angle is not None and angle > config.stitch_max_join_angle_deg:
            logger.debug(
                "Rejected join-angle stitch: traj %d→%d (angle=%.1f deg > %.1f)",
                li, ri, angle, config.stitch_max_join_angle_deg,
            )
            continue
        angle_safe.append((li, ri))

    return angle_safe


def stitch_trajectories(
    trajectories: list[RowTrajectory],
    n_strips: int,
    spacing_px: float,
    config: PipelineConfig,
) -> tuple[list[RowTrajectory], list[OcclusionGap]]:
    """Post-tracking stitching pass.

    Reconnects broken trajectories and detects internal occlusion gaps.

    Args:
        trajectories: Sorted by mean_perp (from track_rows).
        n_strips: Total number of strips.
        spacing_px: Row spacing in pixels.
        config: Pipeline configuration.

    Returns:
        (stitched_trajectories, occlusion_gaps)
    """
    if not config.stitch_enabled or len(trajectories) < 2:
        # Even without stitching, populate segments for all trajectories
        for traj in trajectories:
            traj.segments = _compute_segments(traj)
        return trajectories, []

    logger.info("Stitching: %d trajectories, %d strips", len(trajectories), n_strips)

    # Step 1: Build presence matrix
    presence = _build_presence_matrix(trajectories, n_strips)

    # Step 2: Detect group occlusion bands
    bands = _detect_occlusion_bands(presence, config.min_group_occlusion_rows)
    logger.info("Detected %d occlusion band(s)", len(bands))

    # Step 3: Group-aware stitching for occlusion bands
    all_merges: list[tuple[int, int]] = []
    band_rows: set[int] = set()

    for band in bands:
        logger.info(
            "  Band: rows %d-%d, strips %d-%d",
            band.row_start, band.row_end, band.strip_start, band.strip_end,
        )
        pairs = _group_stitch(band, trajectories, spacing_px, config)
        all_merges.extend(pairs)
        for r in range(band.row_start, band.row_end + 1):
            band_rows.add(r)

    # Step 4: Pairwise stitching for isolated dropouts (rows not in bands)
    # Find trajectories that end early but aren't part of a band
    for i in range(len(trajectories)):
        if i in band_rows:
            continue
        # Already merged?
        if any(m[0] == i or m[1] == i for m in all_merges):
            continue

        _, last_i = _matched_strip_range(trajectories[i])
        if last_i < 0:
            continue

        best_score = float("inf")
        best_j = -1

        for j in range(len(trajectories)):
            if j == i or j in band_rows:
                continue
            if any(m[0] == j or m[1] == j for m in all_merges):
                continue

            first_j, _ = _matched_strip_range(trajectories[j])
            if first_j <= last_i:
                continue  # j doesn't start after i

            score = _stitch_score(trajectories[i], trajectories[j], spacing_px, config)
            if score < best_score:
                best_score = score
                best_j = j

        if best_j >= 0 and best_score < float("inf"):
            # Check no other trajectory is a similarly good match (ambiguity)
            second_best = float("inf")
            for j in range(len(trajectories)):
                if j == i or j == best_j or j in band_rows:
                    continue
                if any(m[0] == j or m[1] == j for m in all_merges):
                    continue
                first_j, _ = _matched_strip_range(trajectories[j])
                if first_j <= last_i:
                    continue
                score = _stitch_score(trajectories[i], trajectories[j], spacing_px, config)
                if score < second_best:
                    second_best = score

            if second_best == float("inf") or \
               second_best / max(best_score, 1e-6) >= config.stitch_ambiguity_ratio:
                # Check join angle before accepting
                angle = _join_angle_deg(trajectories[i], trajectories[best_j])
                if angle is not None and angle > config.stitch_max_join_angle_deg:
                    logger.debug(
                        "Rejected pairwise stitch join angle: traj %d→%d (angle=%.1f deg)",
                        i, best_j, angle,
                    )
                else:
                    all_merges.append((i, best_j))
            else:
                logger.debug(
                    "Rejected ambiguous pairwise stitch: traj %d→%d "
                    "(score=%.3f, 2nd=%.3f)",
                    i, best_j, best_score, second_best,
                )

    # Step 5: Execute merges
    logger.info("Executing %d stitch merge(s)", len(all_merges))

    # Build merge chains (A→B, B→C becomes A→B→C)
    merge_target: dict[int, int] = {}  # right → left (canonical)
    for left, right in all_merges:
        # Follow chain to find the root left
        root = left
        while root in merge_target:
            root = merge_target[root]
        merge_target[right] = root

    # Group by root
    groups: dict[int, list[int]] = {}
    for right, root in merge_target.items():
        groups.setdefault(root, [root]).append(right)
    # Add roots that aren't targets
    for left, right in all_merges:
        root = left
        while root in merge_target:
            root = merge_target[root]
        if root not in groups:
            groups[root] = [root]

    merged_indices: set[int] = set()
    result: list[RowTrajectory] = []

    # Sort roots by their position to maintain order
    sorted_roots = sorted(groups.keys())
    for root in sorted_roots:
        chain = sorted(set(groups[root]))
        merged_indices.update(chain)

        # Merge the chain left-to-right
        merged = trajectories[chain[0]]
        for idx in chain[1:]:
            merged = _stitch_pair(merged, trajectories[idx], n_strips)
        result.append(merged)

    # Add unmerged trajectories
    for i, traj in enumerate(trajectories):
        if i not in merged_indices:
            result.append(traj)

    # Re-sort by mean perpendicular position
    result.sort(key=lambda t: t.mean_perp)
    for i, t in enumerate(result):
        t.track_id = i

    # Step 6: Populate segment metadata for all trajectories
    for traj in result:
        traj.segments = _compute_segments(traj)

    # Step 7: Build OcclusionGap objects
    # Re-detect bands from the stitched trajectories (catches gaps that were
    # between separate trajectories, now visible as interior gaps after merge)
    post_presence = _build_presence_matrix(result, n_strips)
    post_bands = _detect_occlusion_bands(post_presence, config.min_group_occlusion_rows)

    occlusion_gaps: list[OcclusionGap] = []
    # Combine pre-stitch and post-stitch band detections (deduplicated)
    all_bands = bands + post_bands
    seen_band_keys: set[tuple[int, int, int, int]] = set()
    for band in all_bands:
        key = (band.row_start, band.row_end, band.strip_start, band.strip_end)
        if key in seen_band_keys:
            continue
        seen_band_keys.add(key)
        gap = OcclusionGap(
            start_strip=band.strip_start,
            end_strip=band.strip_end,
            affected_row_indices=list(range(band.row_start, band.row_end + 1)),
            gap_type="unknown",
        )
        occlusion_gaps.append(gap)

    logger.info(
        "Stitching complete: %d trajectories (from %d), %d occlusion gap(s)",
        len(result), len(trajectories), len(occlusion_gaps),
    )

    return result, occlusion_gaps
