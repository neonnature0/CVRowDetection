"""
Pipeline configuration with sensible defaults for vineyard row detection.

All tunable parameters are centralized here. Users can override individual
fields via the CLI or programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """All tunable parameters for the row detection pipeline."""

    # --- Spacing bounds (meters) ---
    min_spacing_m: float = 1.0
    max_spacing_m: float = 5.0

    # --- Preprocessing ---
    mask_erosion_px: int = 3            # erode mask boundary to avoid edge artifacts
    exg_signal_threshold: float = 15.0  # std dev threshold for ExG viability

    # --- Ridge detection (Stage 3) ---
    ridge_scale_factor: float = 0.2     # sigma = factor * spacing_px
    ridge_angle_tolerance_deg: float = 15.0  # suppress responses outside this angle range
    ridge_mode: str = "ml"                 # hessian|luminance|exg_only|gabor|ensemble|hessian_small|hessian_large|ml|ml_ensemble
    ml_model_path: str = "training/checkpoints_fpn/best_model.pth"  # path to trained checkpoint
    ml_align_rows: bool = False     # rotate image to align rows vertically before ML inference
    ml_decoder: str = "fpn"         # "unet" or "fpn"

    # --- Candidate extraction (Stage 4) ---
    strip_width_factor: float = 3.0     # strip width = factor * spacing_px
    strip_overlap: float = 0.5          # fraction overlap between adjacent strips
    peak_min_distance_factor: float = 0.80  # min distance between peaks = factor * spacing_px
    peak_min_prominence: float = 0.10   # minimum peak prominence (fraction of max)
    edge_strip_fraction: float = 0.2    # first/last N% of strips are "edge" strips
    edge_prominence_factor: float = 0.5 # prominence multiplier for edge strips

    # --- Phase correction (Stage 4 post-processing) ---
    phase_correction_enabled: bool = True   # adaptive texture-based phase check
    phase_correction_threshold: float = 0.6 # fraction of strips that must vote "shift"

    # --- Tracking (Stage 5) ---
    position_weight: float = 1.0        # cost weight for perpendicular displacement
    strength_weight_factor: float = 0.2  # strength cost = factor * spacing_px * (1 - strength)
    skip_penalty_factor: float = 0.5    # gap coasting penalty = factor * spacing_px
    birth_penalty_factor: float = 0.8   # new track penalty = factor * spacing_px
    min_track_length: int = 3           # discard tracks with fewer matched candidates
    max_consecutive_skips: int = 8      # max strips a track can coast without a match
    validation_threshold_factor: float = 0.75  # match rejection = factor * spacing_px
    min_candidate_likelihood_ratio: float = 0.3  # candidate likelihood / strip mean; below = skip
    skip_escalation_rate: float = 0.25  # skip cost multiplier per consecutive skip
    recovery_strength_ratio: float = 0.5  # recovery min strength = ratio × block median strength
    gap_bridge_enabled: bool = False    # post-tracking gap bridge pass (legacy, unused)
    gap_bridge_lookahead: int = 8       # max strips to search beyond track death (legacy)

    # --- Post-tracking stitching (Stage 5b) ---
    stitch_enabled: bool = True             # enable post-tracking segment stitching
    max_stitch_gap_strips: int = 20         # max strip gap for stitching two segments
    stitch_perp_tolerance: float = 0.5      # max perp offset = factor × spacing_px
    stitch_slope_tolerance: float = 0.3     # max slope difference (perp units per strip)
    min_group_occlusion_rows: int = 3       # min adjacent rows missing to flag group occlusion
    stitch_ambiguity_ratio: float = 1.5     # best match must be this much better than 2nd best
    stitch_max_join_angle_deg: float = 15.0 # max direction change at stitch join point

    # --- Fitting (Stage 6) ---
    curvature_soft_limit: float = 10.0  # deg/m — curvature above this penalizes confidence
    spline_smoothing_m: float = 0.2     # allowed deviation from smooth curve (meters)
    centerline_sample_interval_px: float = 10.0  # sample spline every N pixels
    spline_extrapolate_factor: float = 0.0       # endpoint extrapolation = factor * spacing_px (0=off)
    completeness_denominator_factor: float = 0.5  # completeness = n_matched / (strips * factor)

    # --- Post-processing (Stage 7) ---
    min_row_confidence: float = 0.15    # rows below this confidence are discarded
    min_row_length_fraction: float = 0.2  # rows shorter than this fraction of max are discarded
    spacing_cv_warning: float = 0.15    # spacing CV above this triggers SPACING_IRREGULAR flag
    missing_row_factor: float = 1.5     # gap > factor * median_spacing triggers MISSING_ROWS

    # --- Debug ---
    save_debug_artifacts: bool = True
    debug_output_dir: str = "output"

    # --- Tile fetching ---
    tile_cache_dir: str = "output/.tile_cache"

    def validate(self) -> list[str]:
        """Return a list of validation warnings (empty = valid)."""
        warnings = []
        if self.min_spacing_m >= self.max_spacing_m:
            warnings.append(
                f"min_spacing_m ({self.min_spacing_m}) >= max_spacing_m ({self.max_spacing_m})"
            )
        if self.strip_overlap < 0 or self.strip_overlap >= 1.0:
            warnings.append(f"strip_overlap ({self.strip_overlap}) should be in [0, 1)")
        if self.min_track_length < 1:
            warnings.append(f"min_track_length ({self.min_track_length}) must be >= 1")
        return warnings
