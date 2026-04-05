# Vineyard Row Detection from Aerial Imagery

Detects vine row positions, orientation, and spacing from satellite/aerial photos. Give it a block boundary polygon and it figures out where every row is.

Built for spray planning, planting records, and compliance tracking in New Zealand and Canadian vineyards.

## How Detection Works

The pipeline takes a block boundary polygon and runs 7 stages:

1. **Tile Acquisition** — fetches aerial tiles for the block area, stitches them, masks to the polygon boundary
2. **Multi-Channel Preprocessing** — converts RGB to several derived channels (ExG vegetation index, luminance, structure tensor) and fuses them with quality weighting
3. **2D FFT Orientation** — finds row angle and spacing from the frequency domain. Parallel rows create distinct peaks in the spectrum whose position encodes angle and spacing directly
4. **Ridge Likelihood** — builds a per-pixel probability map of "how likely is this pixel on a row centerline." Uses either a Gabor bandpass filter (classical) or a trained U-Net (ML mode)
5. **Candidate Extraction** — slices the image into strips perpendicular to the row direction, finds peaks in each strip independently. Handles curved rows that a single global projection would miss
6. **Row Tracking** — connects candidates across strips using the Hungarian algorithm for globally optimal assignment. Handles gaps, missing vines, and fading rows at block edges
7. **Centerline Fitting** — fits cubic smoothing splines through tracked points to produce smooth polyline centerlines

Each stage reduces uncertainty one dimension at a time. The FFT gives angle and spacing, the ridge detector gives per-pixel likelihood, strip extraction gives candidate positions, tracking gives row identity, and fitting gives the final curves.

## GUI

A browser-based interface for the full workflow. Run:

```bash
pip install ".[gui]"
python -m gui.server
```

Opens at `http://127.0.0.1:8765` with five sections:

**Add Blocks** — MapLibre map with LINZ aerial tiles. Draw block boundary polygons, saved with anonymous hex names to avoid identification bias.

**Annotate** — Sequential workflow through detected blocks. Accept the detection as-is, edit rows in matplotlib (opens as a separate window), or blind-annotate from scratch for unbiased ground truth.

**Library** — Grid view of all blocks with thumbnails, stage badges, and actions (detect, tune, annotate, delete). Click thumbnails to inspect at full resolution with zoom/pan.

**Tune** — Adjust 6 key pipeline parameters with sliders and see before/after comparison overlays. Covers ridge mode, Gabor scale, peak sensitivity, tracker position weight, spline smoothing, and confidence threshold.

**Train** — Generate ML training patches from completed annotations, train the U-Net model with live progress, then bulk-invalidate detection caches to re-detect with the new model.

**Verify** — Pick N random blocks, run detection on all of them, view the results as a scrollable overlay grid.

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
```

Or use the GUI's Train tab which does both with progress streaming.

## Project History

**Gen 1** — Hough transform + 1D FFT. Abandoned (too noisy, poor spacing accuracy). Deleted from repo, recoverable from git tag `pre-decouple`.

**Gen 2** — Current 7-stage pipeline with 2D FFT, Gabor filter, Hungarian tracking, spline fitting. 95.9% mean F1 on 11 benchmark blocks.

**Gen 3** — ML ridge detection (FPN with MobileNet-v2 encoder) replacing the Gabor filter in Stage 3. In progress.
