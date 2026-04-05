# CV Row Detection — Session Handover (2026-04-05)

## Project Location
`D:\Cordyn + Supamode\Cordyn\apps\web\scripts\cv-row-detection\`

## Git State
- Branch: `main`
- **WARNING**: Working tree has uncommitted changes that diverge from committed state. The pipeline files (`pipeline.py`, `config.py`, `splines.py`) were restored via `git checkout` to an older state (`865fc4b`) but this was not committed. The committed HEAD (`7a5371b`) contains the global profile method. The working tree contains the old tracker pipeline.
- Run `git diff` and `git status` to see exact state before doing anything.

## What Was Done This Session

### Stages 1-3: Tracker Improvements (Committed)
- `0f2abf0` — Inf/NaN guards, per-strip event logging (StripEvent diagnostics)
- `628e98e` — Curvature confidence penalty, likelihood-corridor validation, centerline likelihood profiles
- `fd99eba` — Strength-aware stitch scoring, join angle validation (15°), recovery strength floor

### Stage 4-5: Fitting Refinements (Committed)
- `d0d650c` — Weighted centroid refinement (sub-pixel candidate positions)
- `fee9b3f` — Likelihood-based endpoint trimming

### Architectural Fixes (Committed)
- `2aab957` — Recovery pass: strip-overlap-aware `already_covered` check. Fixed the bug where continuation segments were rejected because they shared the same perp position as the truncated first half. **Babich Main +27 rows.**
- `1268aca` — Validation gate: `min(predicted_perp, last_perp)` for acceptance. Prevents single-jitter track deaths. **Other Vineyard A +4 rows.**

### Gap Detection System (Committed, partially working)
- `caf6073` — Post-fit strip-based support analysis with hysteresis, segment-aware rendering
- `e73f80f` — Always use support-based segments (CAUSED REGRESSION — rows truncated)
- `8bc8618` — Reverted to only replace segments when gaps found
- `865fc4b` — Added ExG vegetation check to gap detection

**Gap detection status**: Infrastructure is built (RowSegment with point indices, strip-based support, hysteresis, ExG check) but does NOT visually activate on Brooklands A/B because the block polygons include buildings — the mask has no holes, the ML model predicts high likelihood over buildings, and the tracker assigns candidates through them.

### Global Profile Method (Committed at `7a5371b`, then broken)
- New `vinerow/detection/global_profile.py` — project likelihood onto perpendicular axis, detect peaks, create straight lines at FFT angle clipped to mask
- Prototype showed excellent results: 100% recall on Block A (26/26), B (82/82), C (110/110), D (111/111), North Block (251/251)
- Then I swapped peak detection from likelihood to ExG to fix a phase offset on Babich Main — **this destroyed detection on most blocks** (Block B: 0 rows). ExG lacks contrast at satellite resolution.
- Attempted to revert but the user reports the restored state still doesn't match the quality they saw before the changes
- **Current working tree has been restored to pre-global-profile state (old tracker at `865fc4b`)** but this may not be the exact state the user wants

## Key Files

| File | Purpose |
|------|---------|
| `vinerow/pipeline.py` | Main orchestrator — routes between tracker and global profile |
| `vinerow/config.py` | All config params (row_detection_method, gap params, etc.) |
| `vinerow/tracking/assignment.py` | Strip-based Hungarian tracker (old method) |
| `vinerow/tracking/stitching.py` | Post-tracking segment stitching |
| `vinerow/fitting/splines.py` | Spline fitting + gap detection + endpoint trimming |
| `vinerow/detection/global_profile.py` | Global perpendicular profile method (new) |
| `vinerow/candidates/extraction.py` | Strip candidate extraction |
| `vinerow/debug/row_diagnostics.py` | Per-strip event diagnostics |
| `visual_verify.py` | Overlay image generation |
| `test_blocks.json` | 12 benchmark blocks with ground truth |

## Known Issues

1. **Lines through buildings**: Block polygons don't exclude buildings. The mask covers them, the ML model predicts high likelihood over them, and every detection method draws through them. Fix: either add inner rings to polygons, or train the ML model to distinguish buildings from vine rows.

2. **Babich Main phase offset**: The ML likelihood peaks are offset ~8 pixels from actual canopy centers on this block. The old tracker's phase correction partially compensated. The global profile method doesn't have phase correction. ExG peaks align with canopy but ExG lacks contrast for peak detection at satellite resolution.

3. **Working tree / HEAD mismatch**: The committed HEAD has the global profile method, but the working tree has been reverted to the old tracker. This needs to be resolved — either commit the revert or reset.

4. **User reports current output doesn't match pre-session quality**: The user says the current overlays don't match what they had before this session started. The exact pre-session state was commit `a31e806` (WIP from crashed session). It's possible that the uncommitted changes from that crashed session included a working state that was never committed.

## Research Findings

25 papers on vine row detection reviewed (see `relevant_research/`). Key finding: no published method uses our strip-based Hungarian tracker approach. The dominant methods are:
1. **Pixel-based segmentation + Hough transform** (works at 0.02-0.05m resolution, NOT our 0.11m)
2. **Deep learning segmentation** (U-Net family — our ML ridge model is in this family)
3. **Perpendicular profile analysis** — the global profile approach we prototyped

The global profile method is the most promising for our resolution but needs phase correction for some blocks.

## Config Parameters Added This Session

```python
# Tracking
min_candidate_likelihood_ratio: float = 0.3
recovery_strength_ratio: float = 0.5
stitch_max_join_angle_deg: float = 15.0

# Fitting
curvature_soft_limit: float = 10.0
endpoint_trim_likelihood_ratio: float = 0.3
endpoint_trim_min_run: int = 3

# Gap detection
gap_min_candidate_strength: float = 0.10
gap_strength_ratio: float = 0.5
gap_max_candidate_residual_factor: float = 0.3
gap_likelihood_dilation_strips: int = 2
gap_likelihood_threshold: float = 0.3
gap_min_consecutive_unsupported: int = 3
gap_min_consecutive_supported: int = 2
gap_min_visible_segment_length_m: float = 2.0
gap_exg_ratio: float = 0.6

# Detection method
row_detection_method: str = "global_profile" | "tracker"
```

## Benchmark Results (Best achieved)

### Old Tracker (commit `1268aca`)
| Block | Rows | Recall |
|-------|------|--------|
| Block A | 26 | 100% |
| Block B | 80 | 98% |
| Block C | 108 | 98% |
| Block D | 111 | 100% |
| Block D2 | 109 | 99% |
| North Block | 249 | 99% |
| South Block | 286 | 99% |
| Other Vineyard A | 95 | 96% |
| B10 | 37 | 95% |
| B12 | 55 | 90% |
| Babich Main | 126 | N/A |

### Global Profile (prototype, not integrated pipeline)
| Block | Rows | Recall |
|-------|------|--------|
| Block A | 26 | 100% |
| Block B | 82 | 100% |
| Block C | 110 | 100% |
| Block D | 111 | 100% |
| Block D2 | 109 | 99% |
| North Block | 251 | 100% |
| South Block | 287 | 99% |
| Other Vineyard A | 92 | 93% |
| B10 | 38 | 97% |
| B12 | 60 | 98% |
| Babich Main | 96 | N/A |

## Recommendations for Next Session

1. **Resolve git state first**: `git status`, decide whether to keep old tracker or global profile, commit cleanly.
2. **If continuing global profile**: Add phase correction (check if likelihood peaks vs ExG peaks are offset, shift if needed per-block). Don't swap the entire profile signal to ExG — just use it for a phase offset correction.
3. **If staying with old tracker**: The user's main complaint is lines going through buildings. This cannot be fixed without either (a) polygon exclusion rings or (b) training the ML model to suppress non-vineyard areas.
4. **Don't make multiple untested changes at once.** Commit and benchmark each change separately.
