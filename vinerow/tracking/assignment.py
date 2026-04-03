"""
Row tracking via sequential linear assignment (Hungarian algorithm).

Links candidates across strips into coherent row trajectories. Seeds from
the densest strip (best coverage), then tracks forward and backward
independently, merging the two passes for each seed row.

Includes a recovery pass that creates trajectories from orphaned candidates
beyond long gaps where allow_births=False prevents new track creation.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from vinerow.config import PipelineConfig
from vinerow.debug.row_diagnostics import StripEvent
from vinerow.types import CoarseOrientation, RowCandidate, RowTrajectory

logger = logging.getLogger(__name__)


class _Track:
    """Internal mutable track state during sequential processing."""

    def __init__(self, track_id: int, first_candidate: RowCandidate, n_strips: int):
        self.track_id = track_id
        self.candidates: list[RowCandidate | None] = [None] * n_strips
        self.candidates[first_candidate.strip_index] = first_candidate
        self.last_perp = first_candidate.perp_position
        self.prev_perp = first_candidate.perp_position
        self.birth_perp = first_candidate.perp_position  # stable identity reference
        self.birth_strip = first_candidate.strip_index
        self.last_matched_strip = first_candidate.strip_index
        self.consecutive_skips = 0
        self.alive = True

    @property
    def predicted_perp(self) -> float:
        """Linear extrapolation from last two matched positions."""
        pred = self.last_perp + (self.last_perp - self.prev_perp)
        if not np.isfinite(pred):
            logger.warning(
                "Track %d: non-finite predicted_perp=%.1f (last=%.1f, prev=%.1f, birth=%.1f) — falling back to last_perp",
                self.track_id, pred, self.last_perp, self.prev_perp, self.birth_perp,
            )
            return self.last_perp
        return pred

    def match(self, candidate: RowCandidate) -> None:
        self.candidates[candidate.strip_index] = candidate
        self.prev_perp = self.last_perp
        self.last_perp = candidate.perp_position
        self.last_matched_strip = candidate.strip_index
        self.consecutive_skips = 0

    def skip(self) -> None:
        self.consecutive_skips += 1

    def to_trajectory(self) -> RowTrajectory:
        death = self.birth_strip
        for i, c in enumerate(self.candidates):
            if c is not None:
                death = i
        return RowTrajectory(
            track_id=self.track_id,
            candidates=list(self.candidates),
            birth_strip=self.birth_strip,
            death_strip=death,
        )


def _emit(
    events: dict[int, list[StripEvent]] | None,
    track_id: int,
    strip_index: int,
    event: str,
    **kwargs,
) -> None:
    """Append a StripEvent if diagnostics are enabled."""
    if events is not None:
        events.setdefault(track_id, []).append(
            StripEvent(strip_index=strip_index, event=event, **kwargs)
        )


def _process_strip(
    alive_tracks: list[_Track],
    strip_cands: list[RowCandidate],
    spacing_px: float,
    skip_penalty: float,
    birth_penalty: float,
    max_skip_strips: int,
    config: PipelineConfig,
    all_tracks: list[_Track],
    next_track_id: int,
    n_strips: int,
    allow_births: bool = True,
    strip_index: int = -1,
    strip_events: dict[int, list[StripEvent]] | None = None,
    strip_mean_likelihood: float = 0.0,
) -> int:
    """Process one strip: match candidates to tracks via Hungarian assignment.

    Returns updated next_track_id.
    """
    if not alive_tracks and not strip_cands:
        return next_track_id

    if not alive_tracks:
        if allow_births:
            for c in strip_cands:
                _emit(strip_events, next_track_id, strip_index, "birth",
                       perp_actual=c.perp_position, strength=c.strength)
                all_tracks.append(_Track(next_track_id, c, n_strips))
                next_track_id += 1
        return next_track_id

    if not strip_cands:
        for t in alive_tracks:
            t.skip()
            if t.consecutive_skips >= max_skip_strips:
                t.alive = False
                _emit(strip_events, t.track_id, strip_index, "death",
                       reason="skip_limit", perp_predicted=t.predicted_perp)
            else:
                _emit(strip_events, t.track_id, strip_index, "skip",
                       reason="empty_strip", perp_predicted=t.predicted_perp)
        return next_track_id

    n_tracks = len(alive_tracks)
    n_cands = len(strip_cands)

    # Augmented cost matrix for Hungarian assignment
    size = n_tracks + n_cands
    cost = np.full((size, size), 1e9, dtype=np.float64)

    # Real assignment costs
    for i, track in enumerate(alive_tracks):
        pred_perp = track.predicted_perp
        if not np.isfinite(pred_perp):
            logger.warning(
                "Track %d: non-finite predicted_perp in cost matrix (birth=%.1f, last=%.1f, prev=%.1f)",
                track.track_id, track.birth_perp, track.last_perp, track.prev_perp,
            )
        for j, cand in enumerate(strip_cands):
            pos_cost = abs(pred_perp - cand.perp_position)
            pos_cost = min(pos_cost, 1e6)  # safety clamp
            strength_cost = config.strength_weight_factor * spacing_px * (1.0 - cand.strength)
            cost[i, j] = config.position_weight * pos_cost + strength_cost

    # Skip costs
    for i in range(n_tracks):
        cost[i, n_cands + i] = skip_penalty * (1 + alive_tracks[i].consecutive_skips * config.skip_escalation_rate)

    # Birth costs
    for j in range(n_cands):
        cost[n_tracks + j, j] = birth_penalty if allow_births else 1e9

    # Dummy-to-dummy
    for i in range(n_cands):
        for j in range(n_tracks):
            cost[n_tracks + i, n_cands + j] = 0.0

    # Verify no inf/nan leaked into the cost matrix
    if not np.isfinite(cost).all():
        bad = np.argwhere(~np.isfinite(cost))
        for bi, bj in bad[:5]:  # log first 5 offenders
            logger.warning(
                "Non-finite cost[%d,%d]=%.1f (n_tracks=%d, n_cands=%d)",
                bi, bj, cost[bi, bj], n_tracks, n_cands,
            )
        cost = np.clip(cost, 0, 1e9)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_tracks: set[int] = set()
    matched_cands: set[int] = set()

    # Likelihood-corridor validation threshold
    lk_ratio_min = config.min_candidate_likelihood_ratio

    for ri, ci in zip(row_ind, col_ind):
        if ri < n_tracks and ci < n_cands:
            track = alive_tracks[ri]
            cand = strip_cands[ci]
            pos_error = abs(track.predicted_perp - cand.perp_position)
            if pos_error < config.validation_threshold_factor * spacing_px:
                # Likelihood-corridor check: reject if candidate likelihood
                # is too weak relative to the strip mean
                lk_ratio = cand.likelihood / strip_mean_likelihood if strip_mean_likelihood > 1e-6 else 1.0
                if lk_ratio < lk_ratio_min:
                    track.skip()
                    _emit(strip_events, track.track_id, strip_index, "skip",
                           reason="low_likelihood",
                           perp_predicted=track.predicted_perp,
                           perp_actual=cand.perp_position,
                           strength=round(lk_ratio, 3))
                    if track.consecutive_skips >= max_skip_strips:
                        track.alive = False
                        _emit(strip_events, track.track_id, strip_index, "death",
                               reason="skip_limit")
                    continue

                track.match(cand)
                matched_tracks.add(ri)
                matched_cands.add(ci)
                _emit(strip_events, track.track_id, strip_index, "match",
                       perp_predicted=track.predicted_perp,
                       perp_actual=cand.perp_position,
                       position_error=round(pos_error, 2),
                       strength=round(cand.strength, 3))
            else:
                track.skip()
                _emit(strip_events, track.track_id, strip_index, "skip",
                       reason="validation_rejected",
                       perp_predicted=track.predicted_perp,
                       perp_actual=cand.perp_position,
                       position_error=round(pos_error, 2))
                if track.consecutive_skips >= max_skip_strips:
                    track.alive = False
                    _emit(strip_events, track.track_id, strip_index, "death",
                           reason="skip_limit")
        elif ri < n_tracks and ci >= n_cands:
            track = alive_tracks[ri]
            if ri not in matched_tracks:
                track.skip()
                _emit(strip_events, track.track_id, strip_index, "skip",
                       reason="no_candidate",
                       perp_predicted=track.predicted_perp)
                if track.consecutive_skips >= max_skip_strips:
                    track.alive = False
                    _emit(strip_events, track.track_id, strip_index, "death",
                           reason="skip_limit")

    # Birth new tracks for unmatched candidates
    if allow_births:
        for cj in range(n_cands):
            if cj not in matched_cands:
                c = strip_cands[cj]
                _emit(strip_events, next_track_id, strip_index, "birth",
                       perp_actual=c.perp_position, strength=round(c.strength, 3))
                all_tracks.append(_Track(next_track_id, c, n_strips))
                next_track_id += 1

    return next_track_id


def _track_direction(
    seed_cands: list[RowCandidate],
    strip_order: list[int],
    by_strip: dict[int, list[RowCandidate]],
    n_strips: int,
    spacing_px: float,
    skip_penalty: float,
    birth_penalty: float,
    max_skip_strips: int,
    config: PipelineConfig,
    strip_events: dict[int, list[StripEvent]] | None = None,
    strip_mean_likelihoods: dict[int, float] | None = None,
) -> list[_Track]:
    """Track one direction (forward or backward) from seed candidates."""
    tracks: list[_Track] = []
    next_id = 0
    for c in seed_cands:
        tracks.append(_Track(next_id, c, n_strips))
        next_id += 1

    for s in strip_order:
        strip_cands = by_strip.get(s, [])
        alive = [t for t in tracks if t.alive]
        sml = strip_mean_likelihoods.get(s, 0.0) if strip_mean_likelihoods else 0.0
        next_id = _process_strip(
            alive, strip_cands, spacing_px, skip_penalty, birth_penalty,
            max_skip_strips, config, tracks, next_id, n_strips,
            allow_births=False,
            strip_index=s,
            strip_events=strip_events,
            strip_mean_likelihood=sml,
        )

    return tracks


def track_rows(
    candidates: list[RowCandidate],
    strip_centers: list[float],
    coarse: CoarseOrientation,
    config: PipelineConfig,
) -> list[RowTrajectory]:
    """Link candidates across strips into coherent row trajectories.

    Seeds from the densest strip, tracks forward and backward separately,
    then merges the two passes.
    """
    n_strips = len(strip_centers)
    if n_strips == 0 or not candidates:
        return []

    spacing_px = coarse.spacing_px
    skip_penalty = config.skip_penalty_factor * spacing_px
    birth_penalty = config.birth_penalty_factor * spacing_px
    max_skip_strips = config.max_consecutive_skips

    # Group candidates by strip
    by_strip: dict[int, list[RowCandidate]] = defaultdict(list)
    for c in candidates:
        by_strip[c.strip_index].append(c)
    for s in by_strip:
        by_strip[s].sort(key=lambda c: c.perp_position)

    # Seed from the densest strip
    seed_strip = max(by_strip.keys(), key=lambda s: len(by_strip[s]))
    seed_cands = by_strip[seed_strip]

    # Record seed info in diagnostics
    from vinerow.debug.row_diagnostics import current as diag_current
    diag = diag_current()
    if diag:
        diag.seed_strip = seed_strip
        diag.n_seed_candidates = len(seed_cands)

    logger.info(
        "Tracking: %d seed candidates from strip %d (densest), spacing_px=%.1f",
        len(seed_cands), seed_strip, spacing_px,
    )

    # Per-strip mean likelihood for corridor validation
    strip_mean_likelihoods: dict[int, float] = {}
    for s, cands in by_strip.items():
        lks = [c.likelihood for c in cands if c.likelihood > 0]
        strip_mean_likelihoods[s] = sum(lks) / len(lks) if lks else 0.0

    # Per-strip event collection (keyed by track_id)
    strip_events: dict[int, list[StripEvent]] = {} if diag else None

    # Emit seed birth events
    if strip_events is not None:
        for i, c in enumerate(seed_cands):
            _emit(strip_events, i, seed_strip, "birth",
                   perp_actual=c.perp_position, strength=round(c.strength, 3),
                   reason="seed")

    # Track forward and backward independently
    forward_strips = list(range(seed_strip + 1, n_strips))
    backward_strips = list(range(seed_strip - 1, -1, -1))

    fwd_tracks = _track_direction(
        seed_cands, forward_strips, by_strip, n_strips,
        spacing_px, skip_penalty, birth_penalty, max_skip_strips, config,
        strip_events=strip_events,
        strip_mean_likelihoods=strip_mean_likelihoods,
    )
    bwd_tracks = _track_direction(
        seed_cands, backward_strips, by_strip, n_strips,
        spacing_px, skip_penalty, birth_penalty, max_skip_strips, config,
        strip_events=strip_events,
        strip_mean_likelihoods=strip_mean_likelihoods,
    )

    # Merge forward and backward: each seed candidate gets ONE merged track
    merged_tracks: list[_Track] = []
    for i, seed_c in enumerate(seed_cands):
        ft = fwd_tracks[i] if i < len(fwd_tracks) else None
        bt = bwd_tracks[i] if i < len(bwd_tracks) else None

        merged = _Track(i, seed_c, n_strips)
        merged.candidates = [None] * n_strips
        merged.candidates[seed_strip] = seed_c

        # Copy forward results
        if ft is not None:
            for s in forward_strips:
                if ft.candidates[s] is not None:
                    merged.candidates[s] = ft.candidates[s]

        # Copy backward results
        if bt is not None:
            for s in backward_strips:
                if bt.candidates[s] is not None:
                    merged.candidates[s] = bt.candidates[s]

        merged_tracks.append(merged)

    # Filter by minimum track length
    trajectories = []
    for track in merged_tracks:
        traj = track.to_trajectory()
        if traj.n_matched >= config.min_track_length:
            trajectories.append(traj)

    # Recovery pass: find orphaned candidates not matched to any track
    # and group them into new trajectories. This handles the case where
    # allow_births=False in directional passes causes candidates beyond
    # a long gap to be dropped entirely.
    #
    # Conservative: only recover groups that look like real row segments,
    # not edge noise or scattered false peaks.
    matched_set: set[tuple[int, float]] = set()
    existing_perps: list[float] = []  # perp positions already covered
    for traj in trajectories:
        existing_perps.append(traj.mean_perp)
        for c in traj.candidates:
            if c is not None:
                matched_set.add((c.strip_index, c.perp_position))

    orphaned: list[RowCandidate] = []
    for c in candidates:
        if (c.strip_index, c.perp_position) not in matched_set:
            orphaned.append(c)

    if orphaned:
        logger.info("Recovery: %d orphaned candidates found", len(orphaned))
        if diag:
            diag.n_orphaned_candidates = len(orphaned)

        # Block-relative strength floor for recovery
        all_strengths = [c.strength for c in candidates]
        block_median_strength = float(np.median(all_strengths)) if all_strengths else 0.5
        strength_floor = block_median_strength * config.recovery_strength_ratio
        logger.debug(
            "Recovery: block_median_strength=%.3f, strength_floor=%.3f",
            block_median_strength, strength_floor,
        )

        # Group orphaned candidates by perp proximity (cluster into rows)
        orphaned.sort(key=lambda c: c.perp_position)
        orphan_groups: list[list[RowCandidate]] = []
        current_group: list[RowCandidate] = [orphaned[0]]

        for c in orphaned[1:]:
            if abs(c.perp_position - current_group[-1].perp_position) < spacing_px * 0.6:
                current_group.append(c)
            else:
                orphan_groups.append(current_group)
                current_group = [c]
        orphan_groups.append(current_group)

        # Conservative recovery: only accept groups that are clearly real rows
        min_recovery_length = max(config.min_track_length, 5)  # at least 5 matched
        next_id = len(trajectories)
        max_recovery = 2 * len(seed_cands)  # defensive cap
        n_recovered = 0

        for group in orphan_groups:
            if len(group) < min_recovery_length:
                continue

            cands_arr: list[RowCandidate | None] = [None] * n_strips
            for c in group:
                if cands_arr[c.strip_index] is None or c.strength > cands_arr[c.strip_index].strength:
                    cands_arr[c.strip_index] = c

            first_strip = next((i for i, c in enumerate(cands_arr) if c is not None), 0)
            last_strip = next((i for i in range(n_strips - 1, -1, -1) if cands_arr[i] is not None), 0)
            n_matched = sum(1 for c in cands_arr if c is not None)
            strip_span = last_strip - first_strip + 1

            if n_matched < min_recovery_length:
                continue

            # Reject if strip span is too short (scattered candidates, not a row)
            if strip_span < min_recovery_length:
                continue

            # Reject if density is too low (less than 50% of strips matched)
            if n_matched / strip_span < 0.5:
                continue

            # Reject if mean strength is below the block-relative floor
            group_mean_strength = sum(c.strength for c in group) / len(group)
            if group_mean_strength < strength_floor:
                logger.debug(
                    "Recovery: rejected weak group (mean_strength=%.3f < floor=%.3f, n=%d)",
                    group_mean_strength, strength_floor, len(group),
                )
                continue

            # Reject if this perp position is already well-covered by an existing track
            group_perp = sum(c.perp_position for c in group) / len(group)
            if not np.isfinite(group_perp):
                logger.warning("Recovery: skipping group with non-finite perp (n=%d)", len(group))
                continue
            already_covered = any(
                abs(group_perp - ep) < spacing_px * 0.4
                for ep in existing_perps
            )
            if already_covered:
                continue

            # Defensive cap on recovery count
            if n_recovered >= max_recovery:
                logger.warning(
                    "Recovery: hit cap (%d), skipping remaining %d groups",
                    max_recovery, len(orphan_groups) - orphan_groups.index(group),
                )
                break

            traj = RowTrajectory(
                track_id=next_id,
                candidates=cands_arr,
                birth_strip=first_strip,
                death_strip=last_strip,
            )
            trajectories.append(traj)
            existing_perps.append(group_perp)
            n_recovered += 1
            next_id += 1
            if diag:
                diag.n_recovered_trajectories += 1
            logger.info(
                "  Recovered trajectory: perp=%.0f, matched=%d, strips %d-%d",
                traj.mean_perp, n_matched, first_strip, last_strip,
            )

    # Sort by mean perpendicular position
    trajectories.sort(key=lambda t: t.mean_perp)
    for i, t in enumerate(trajectories):
        t.track_id = i

    # Attach per-strip events to diagnostics
    if diag and strip_events:
        diag._strip_events = strip_events  # stashed for pipeline to consume

    logger.info(
        "Tracking complete: %d tracks from %d seeds (min_length=%d)",
        len(trajectories), len(seed_cands), config.min_track_length,
    )

    return trajectories
