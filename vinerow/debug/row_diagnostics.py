"""
Per-row diagnostic logging.

Collects decision metadata during tracking, stitching, fitting, and
post-processing. Outputs a JSON file per block with one entry per
trajectory/row, including birth/death reasons, match quality, skip
reasons, recovery info, stitch decisions, and fitting metrics.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StripEvent:
    """One strip's tracking decision for a trajectory."""
    strip_index: int
    event: str              # "match", "skip", "birth", "death"
    perp_predicted: float | None = None
    perp_actual: float | None = None
    position_error: float | None = None
    strength: float | None = None
    reason: str | None = None   # e.g., "validation_rejected", "no_candidate", "skip_limit"


@dataclass
class TrajectoryDiagnostic:
    """Diagnostic record for a single trajectory."""
    track_id: int
    mean_perp: float = 0.0
    n_matched: int = 0
    n_strips: int = 0

    # Birth info
    birth_strip: int = 0
    birth_perp: float = 0.0
    birth_strength: float = 0.0
    birth_source: str = "seed"      # "seed" | "recovery"

    # Death info
    death_strip: int = 0
    death_reason: str = "end_of_block"  # "skip_limit" | "end_of_block" | "alive"

    # Recovery info
    is_recovered: bool = False
    recovery_n_orphans: int = 0
    recovery_density: float = 0.0
    recovery_mean_strength: float = 0.0

    # Stitching info
    is_stitched: bool = False
    stitch_gap_strips: int = 0
    stitch_score: float = 0.0
    stitch_source_ids: list[int] = field(default_factory=list)

    # Fitting info
    spline_max_curvature: float = 0.0
    confidence_raw: float = 0.0     # before completeness penalty
    confidence_final: float = 0.0   # after penalty
    length_m: float = 0.0
    completeness: float = 0.0

    # Filtering info
    passed_filters: bool = True
    filter_reason: str = ""         # "low_confidence" | "too_short" | ""

    # Per-strip events (optional, can be large)
    strip_events: list[StripEvent] = field(default_factory=list)


@dataclass
class BlockDiagnostics:
    """Diagnostic record for an entire block."""
    block_name: str = ""
    vineyard_name: str = ""
    n_candidates: int = 0
    n_strips: int = 0
    seed_strip: int = 0
    n_seed_candidates: int = 0
    n_tracks_after_tracking: int = 0
    n_orphaned_candidates: int = 0
    n_recovered_trajectories: int = 0
    n_tracks_after_stitching: int = 0
    n_occlusion_gaps: int = 0
    n_tracks_after_fitting: int = 0
    n_rows_after_filtering: int = 0
    trajectories: list[TrajectoryDiagnostic] = field(default_factory=list)

    def save(self, output_dir: str = "output/diagnostics") -> str:
        """Save diagnostics to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        safe_name = f"{self.vineyard_name}_{self.block_name}".replace(" ", "_")
        path = Path(output_dir) / f"{safe_name}_diagnostics.json"

        # Convert to dict, excluding large strip_events by default
        data = asdict(self)
        for traj in data.get("trajectories", []):
            # Keep strip events only if there are interesting ones (skips, deaths)
            events = traj.get("strip_events", [])
            if len(events) > 200:
                # Summarize: keep only non-match events
                traj["strip_events"] = [e for e in events if e.get("event") != "match"]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved diagnostics to %s", path)
        return str(path)


# Global collector for the current block (set by pipeline, filled by stages)
_current: BlockDiagnostics | None = None


def start_block(block_name: str = "", vineyard_name: str = "") -> BlockDiagnostics:
    """Start collecting diagnostics for a new block."""
    global _current
    _current = BlockDiagnostics(block_name=block_name, vineyard_name=vineyard_name)
    return _current


def current() -> BlockDiagnostics | None:
    """Get the current block's diagnostics collector."""
    return _current


def finish_block(output_dir: str = "output/diagnostics") -> str | None:
    """Save and reset the current block's diagnostics."""
    global _current
    if _current is None:
        return None
    path = _current.save(output_dir)
    _current = None
    return path
