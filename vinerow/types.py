"""
Core data structures for the vineyard row detection pipeline.

All stages communicate through these dataclasses. Debug/visualization
arrays are optional and excluded from repr for readability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class ChannelQuality:
    """Signal quality metrics for a single preprocessed channel."""
    name: str
    std_dev: float          # standard deviation within mask (higher = more signal)
    contrast: float         # (p95 - p5) / 255 within mask
    weight: float           # computed weight for fusion (0–1)


@dataclass
class PreprocessedChannels:
    """Multi-channel preprocessed imagery ready for downstream stages."""
    exg: np.ndarray                     # Excess Green vegetation index (uint8)
    luminance: np.ndarray               # Grayscale (uint8)
    normalized_veg: np.ndarray          # ExG / (R+G+B+eps) normalized (uint8)
    structure_mag: np.ndarray           # Structure tensor anisotropy magnitude (float32, 0–1)
    mask: np.ndarray                    # Binary polygon mask (uint8, 0 or 255), eroded
    original_mask: np.ndarray           # Original mask before erosion
    fused: np.ndarray                   # Weighted combination of channels (float32, 0–1)
    channel_qualities: list[ChannelQuality]
    image_bgr: np.ndarray              # Original BGR image (for debug overlays)

    @property
    def best_channel_name(self) -> str:
        """Name of the highest-weighted channel."""
        if not self.channel_qualities:
            return "luminance"
        return max(self.channel_qualities, key=lambda c: c.weight).name


# ---------------------------------------------------------------------------
# Stage 2: Coarse Orientation
# ---------------------------------------------------------------------------


@dataclass
class CoarseOrientation:
    """Global orientation and spacing estimate from 2D FFT."""
    angle_deg: float              # 0–180, image coordinates
    angle_confidence: float       # 0–1
    angle_uncertainty_deg: float  # +/- half-width of FFT peak
    spacing_m: float
    spacing_px: float
    spacing_confidence: float     # 0–1
    spacing_uncertainty_m: float
    row_count_estimate: int
    # Debug
    log_magnitude: np.ndarray | None = field(default=None, repr=False)
    peak_position: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Stage 3: Ridge Likelihood
# ---------------------------------------------------------------------------

# The likelihood map is just an np.ndarray (float32, 0–1).
# No special type needed.


# ---------------------------------------------------------------------------
# Stage 4: Candidates
# ---------------------------------------------------------------------------


@dataclass
class RowCandidate:
    """A single row centerline candidate point found in one strip."""
    x: float                      # image x coordinate
    y: float                      # image y coordinate
    strip_index: int
    perp_position: float          # perpendicular distance from reference axis
    strength: float               # 0–1 from likelihood map peak height (normalized per-strip)
    half_width_px: float = 0.0    # peak half-width in pixels
    likelihood: float = 0.0       # raw likelihood map value at this position (0–1, not normalized)


# ---------------------------------------------------------------------------
# Segmented Row Primitives (used by Stages 5–7)
# ---------------------------------------------------------------------------


@dataclass
class RowSegment:
    """One contiguous visible or inferred section of a row."""
    start_strip: int        # first strip of this segment
    end_strip: int          # last strip of this segment
    is_visible: bool        # True = matched candidates, False = inferred gap
    start_point_idx: int = -1  # index into FittedRow.centerline_px (-1 = not set)
    end_point_idx: int = -1    # index into FittedRow.centerline_px (-1 = not set)


GapType = Literal["building", "obstacle", "signal_dropout", "unknown"]


@dataclass
class OcclusionGap:
    """A detected internal gap where multiple rows are interrupted."""
    start_strip: int
    end_strip: int
    affected_row_indices: list[int]   # logical row indices interrupted
    gap_type: GapType = "unknown"


# ---------------------------------------------------------------------------
# Stage 5: Tracking
# ---------------------------------------------------------------------------


@dataclass
class RowTrajectory:
    """A tracked row across multiple strips."""
    track_id: int
    candidates: list[RowCandidate | None]  # one per strip, None = gap
    birth_strip: int
    death_strip: int
    segments: list[RowSegment] | None = None          # populated by stitching pass
    source_trajectory_ids: list[int] | None = None     # if stitched from multiple tracks

    @property
    def n_matched(self) -> int:
        return sum(1 for c in self.candidates if c is not None)

    @property
    def mean_perp(self) -> float:
        matched = [c.perp_position for c in self.candidates if c is not None]
        return sum(matched) / len(matched) if matched else 0.0

    @property
    def mean_strength(self) -> float:
        matched = [c.strength for c in self.candidates if c is not None]
        return sum(matched) / len(matched) if matched else 0.0


# ---------------------------------------------------------------------------
# Stage 6: Fitting
# ---------------------------------------------------------------------------


@dataclass
class FittedRow:
    """A fitted vine row centerline."""
    row_index: int
    centerline_px: list[tuple[float, float]]       # polyline in image coords
    centerline_geo: list[tuple[float, float]] | None = None  # polyline in (lng, lat)
    confidence: float = 0.0
    length_m: float = 0.0
    curvature_max_deg_per_m: float = 0.0
    spacing_to_prev_m: float | None = None
    local_spacing_profile: list[float] | None = None  # spacing at sample points along row
    segments: list[RowSegment] | None = None           # visible vs inferred sections
    likelihood_profile: list[float] | None = None      # likelihood values sampled along centerline
    ensemble_confidence: float | None = None           # agreement between ML and classical (0-1)


# ---------------------------------------------------------------------------
# Stage 7: Quality & Results
# ---------------------------------------------------------------------------


class QualityFlag(Flag):
    """Bit flags for quality issues detected in the block."""
    NONE = 0
    LOW_CONFIDENCE = auto()
    SPACING_IRREGULAR = auto()
    MISSING_ROWS = auto()
    WEAK_SIGNAL = auto()
    ORIENTATION_UNCERTAIN = auto()
    FEW_ROWS = auto()
    HEADLAND_DISTORTION = auto()
    HARMONIC_SPACING = auto()
    PHASE_CORRECTED = auto()
    INTERNAL_OCCLUSION = auto()


@dataclass
class StageTimings:
    """Per-stage processing times in seconds."""
    acquisition: float = 0.0
    preprocessing: float = 0.0
    orientation: float = 0.0
    ridge: float = 0.0
    candidates: float = 0.0
    tracking: float = 0.0
    fitting: float = 0.0
    postprocessing: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.acquisition + self.preprocessing + self.orientation
            + self.ridge + self.candidates + self.tracking
            + self.fitting + self.postprocessing
        )


@dataclass
class BlockRowDetectionResult:
    """Complete output of the row detection pipeline for one block."""
    # Core outputs
    rows: list[FittedRow]
    row_count: int
    dominant_angle_deg: float         # from fitted rows (not just FFT)
    dominant_angle_bearing: float     # geographic bearing for DB storage (0–360)
    angle_confidence: float

    # Spacing
    mean_spacing_m: float
    median_spacing_m: float
    spacing_std_m: float
    spacing_range_m: tuple[float, float]

    # Quality
    overall_confidence: float
    quality_flags: QualityFlag

    # Metadata
    timings: StageTimings
    image_size: tuple[int, int]       # (width, height)
    meters_per_pixel: float
    tile_source: str
    zoom_level: int

    # Occlusion metadata
    occlusion_gaps: list[OcclusionGap] = field(default_factory=list)

    # Debug artifacts (optional — excluded from serialization by default)
    coarse_orientation: CoarseOrientation | None = field(default=None, repr=False)
    likelihood_map: np.ndarray | None = field(default=None, repr=False)
    candidate_points: list[RowCandidate] | None = field(default=None, repr=False)
    preprocessed: PreprocessedChannels | None = field(default=None, repr=False)
