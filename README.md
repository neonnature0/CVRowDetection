# Vineyard Row Detection from Aerial Imagery

Detects vine row positions, orientation, and spacing from satellite/aerial photos. Give it a block boundary polygon and it figures out where every row is.

Built for spray planning, planting records, and compliance tracking in New Zealand and Canadian vineyards.

## How Detection Works

The pipeline takes a block boundary polygon and runs 7 stages:

1. **Tile Acquisition** — fetches aerial tiles for the block area, stitches them, masks to the polygon boundary
2. **Multi-Channel Preprocessing** — converts RGB to several derived channels (ExG vegetation index, luminance, structure tensor) and fuses them with quality weighting
3. **2D FFT Orientation** — finds row angle and spacing from the frequency domain. Parallel rows create distinct peaks in the spectrum whose position encodes angle and spacing directly
4. **Ridge Likelihood** — builds a per-pixel probability map of "how likely is this pixel on a row centerline." Uses either a Gabor bandpass filter (classical) or a trained FPN model (ML mode)
5. **Candidate Extraction** — slices the image into strips perpendicular to the row direction, finds peaks in each strip independently. Handles curved rows that a single global projection would miss
6. **Row Tracking** — connects candidates across strips using the Hungarian algorithm for globally optimal assignment. Handles gaps, missing vines, and fading rows at block edges
7. **Centerline Fitting** — fits cubic smoothing splines through tracked points to produce smooth polyline centerlines. Trims weak endpoints and detects gaps using vegetation checks

Each stage reduces uncertainty one dimension at a time. The FFT gives angle and spacing, the ridge detector gives per-pixel likelihood, strip extraction gives candidate positions, tracking gives row identity, and fitting gives the final curves.

## GUI

A browser-based interface for the full workflow — no terminal needed. Run:

```bash
pip install ".[gui]"
python -m gui.server
```

Opens at `http://127.0.0.1:8765` with five sections:

### Add Blocks

MapLibre map with LINZ aerial tiles centred on Marlborough. Draw block boundary polygons using Terra Draw, saved with anonymous 6-hex-digit names to avoid identification bias during evaluation. Previously drawn boundaries show as coloured overlays so you can see what's already mapped.

### Annotate

Sequential workflow through detected blocks. For each block you get three choices:

- **Accept** — detection looks good, save as-is and move to the next block
- **Edit** — opens the matplotlib annotation tool in a separate window where you can drag control points, add/delete rows, undo/redo. The GUI waits for you to save and close
- **Blind** — draws rows from scratch on the raw aerial image without seeing any pipeline output. This creates unbiased ground truth that isn't derived from the pipeline's own detections, which matters for honest evaluation

### Library

Grid view of all blocks with thumbnails, stage badges (draft / detected / annotated / verified), and per-block actions: Detect, Tune, Annotate, Blind, Delete. Click any thumbnail to open it full-screen with zoom and pan for detailed inspection.

### Tuning

The tuning panel is where you experiment with pipeline parameters on individual blocks. Click **Tune** on any detected block in the Library to open it.

**Parameters you can adjust:**

| Parameter | Stage | What it does |
|-----------|-------|-------------|
| Ridge Mode | 3 | Switch between ML, Gabor, ensemble — the single biggest lever |
| Gabor Scale | 3 | Filter bandwidth. Narrower = more selective, wider = more forgiving |
| Peak Sensitivity | 4 | How strong a peak must be to count as a row. Lower = find weaker rows |
| Position Weight | 5 | How strictly the tracker follows straight lines vs allowing curves |
| Spline Smoothing | 6 | How wiggly rows can be. Higher = straighter, lower = follows curves |
| Endpoint Trim Ratio | 6 | How aggressively row tails are trimmed where vines fade |
| Endpoint Trim Run | 6 | How many weak strips before trimming kicks in |
| Min Confidence | 7 | Final filter — rows below this are discarded |
| Ensemble Confidence | — | Run both ML and Gabor, colour by agreement (doubles detection time) |

**Before/after comparison:** After adjusting parameters and clicking "Run with these params", the panel shows the default overlay alongside the tuned overlay. Click either image to open in the lightbox for pixel-level inspection.

**Onion-skin diff:** Toggle "Show Diff" to overlay the tuned row lines (magenta) directly on top of the default overlay. This makes it easy to see exactly where rows shifted rather than trying to eye-jump between two separate images.

**Saving tuned configs:** If the tuned result looks better, click "Apply as Default" to promote it. Per-block configurations are saved alongside the detection results so you don't lose what worked. Two reset buttons: "Reset Saved" goes back to your last saved config, "True Defaults" goes back to the pipeline's factory defaults.

**Ensemble confidence mode:** When the checkbox is on, detection runs both ML and Gabor as independent methods, then scores each row by how well they agree. Rows both methods find in the same position light up green (high trust). Rows only one method finds show as red (worth checking manually). This is the most reliable way to spot phantom detections and edge cases without looking at every row yourself.

### Train

Generate ML training patches from your completed annotations, then train the FPN model with live progress streaming (epoch count, loss, Dice score). After training:

- Click "Clear All Detections" to wipe cached results so re-detection uses the new model
- Add `--calibrate` to the training command for **temperature scaling** — this spreads the model's confidence values across a meaningful range instead of everything being 0.95. Helps downstream stages like endpoint trimming and gap detection make better decisions

### Verify

Pick N random blocks (default 10), run detection on all of them, and view the results as a scrollable grid of overlay images with per-block metrics. Click any thumbnail to inspect in the lightbox. Quick way to check how the pipeline is doing across a range of blocks without looking at each one individually.

## Tips for Best ML Training Results

**Use blind annotation for at least some blocks.** Pipeline-seeded annotations (where you edit the pipeline's own output) create a circular dependency — the model learns to reproduce whatever the pipeline already does, including its mistakes. Blind annotations break this cycle and give the model genuinely independent targets.

**Centre your rows on the physical canopy, not the brightness peak.** Sun angle shifts the bright side of the canopy off-centre. If you always accept pipeline output without correcting the centering, the model inherits that bias. When annotating, zoom in and place rows on the actual canopy midline, especially on blocks with visible shadow asymmetry.

**Annotate diverse blocks, not just easy ones.** A few challenging blocks (grassed inter-rows, curved rows, blocks with gaps or buildings) teach the model more than ten easy high-contrast blocks. The model already handles easy blocks well — it needs to learn the hard cases.

**Use temperature scaling after training** (`--calibrate` flag). Without it, the model outputs near-1.0 confidence on everything, which makes downstream stages like endpoint trimming and gap detection less effective because they can't distinguish strong from weak predictions.

**Check ensemble confidence on new blocks.** When you detect a block you haven't seen before, turn on Ensemble Confidence in the tune panel. If ML and Gabor agree closely, the detection is reliable. If they disagree, that block probably needs manual review or annotation before trusting the result.

**Retrain periodically as you add blocks.** Each new annotated block improves the training set. After annotating 5-10 new blocks, regenerate training data and retrain. The model's accuracy on novel blocks should improve noticeably with each round.

## Evaluation

The evaluation tool (`evaluate_gt.py`) reports honest metrics at three match thresholds:

- **Loose (0.4x spacing)** — ~1m tolerance, the traditional benchmark threshold
- **Medium (0.2x spacing)** — ~0.5m, catches rows that are matched but off-centre
- **Strict (0.1x spacing)** — ~0.25m, only rows that are genuinely well-placed

If your F1 is high at loose but drops sharply at strict, most rows are being found but many are off-centre (likely a sun-angle or annotation bias issue). If F1 holds across all three thresholds, your rows are genuinely well-placed.

The evaluation also reports **shape error** (mean point-to-polyline distance between detected and GT curves), which catches curve errors that the perpendicular-matching metric misses. Both unweighted and row-weighted means are shown so large blocks get proportional influence.

Blind annotations are marked with `[B]` in the output table, with separate means reported when both blind and pipeline-seeded annotations exist.

## LiDAR Elevation Data

For New Zealand blocks, the pipeline can fetch free 1m LiDAR data from LINZ (via AWS Open Data). Computes canopy height (DSM minus DEM) to validate that a drawn polygon is actually a vineyard and flag areas with trees or buildings.

## Tile Sources

Aerial imagery is auto-selected by longitude:

| Source | Coverage | Resolution | Auth |
|--------|----------|-----------|------|
| LINZ | New Zealand | ~0.1m/px | Free API key |
| Kelowna | Okanagan, BC | ~0.2m/px | None |
| ArcGIS World Imagery | Global | ~0.3m/px | None |

## Ridge Detection Modes

| Mode | Method | Use when |
|------|--------|----------|
| `ml` (default) | Trained FPN model | General use, handles grassed inter-rows |
| `gabor` | Gabor bandpass filter | High-contrast blocks, bare soil inter-rows |
| `ml_ensemble` | Max of ML + Gabor | Fallback when ML alone struggles |
| `hessian` | Hessian eigenvalue | Alternative classical method |

## Setup

```bash
python -m venv venv        # Python 3.11 or 3.12 (not 3.14)
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install .              # core deps
pip install ".[ml]"        # add PyTorch + training deps
pip install ".[gui]"       # add FastAPI for the GUI
```

Copy `.env.example` to `.env` and add your LINZ API key (free from [data.linz.govt.nz](https://data.linz.govt.nz/)).

**Note:** The annotation editor launches matplotlib as a subprocess, so the GUI server must run on the same machine you're using (not a remote server).

## CLI Usage

```bash
python cli.py --block "a3f2c1"           # single block
python cli.py --all                       # all blocks
python cli.py --geojson boundary.geojson  # from GeoJSON file
python benchmark.py                       # regression benchmark
python evaluate_gt.py --report            # ground truth evaluation
python visual_verify.py                   # generate overlay images
```

## ML Training

```bash
python generate_training_data.py          # create patches from annotations
python -m training.train --decoder fpn    # train the model
python -m training.train --decoder fpn --calibrate  # train + temperature scaling
```

Or use the GUI's Train tab which handles patch generation and training with live progress.

## Project History

**Gen 1** — Hough transform + 1D FFT. Abandoned (too noisy, poor spacing accuracy). Deleted from repo, recoverable from git tag `pre-decouple`.

**Gen 2** — Current 7-stage pipeline with 2D FFT, Gabor filter, Hungarian tracking, spline fitting.

**Gen 3** — ML ridge detection (FPN with MobileNet-v2 encoder) replacing the Gabor filter in Stage 3, with temperature scaling for calibrated confidence values.
