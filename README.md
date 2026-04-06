# Vineyard Row Detection from Aerial Imagery

An in-development pipeline for detecting vine row positions, orientation, and spacing from aerial imagery. Currently being developed and tested on New Zealand vineyards.

**This is not a pretrained model.** To use it, you clone the repo, draw block boundaries on a map, annotate your own rows (or accept the pipeline's detections), and train your own model. Results depend on how much and how well you annotate. The pipeline works out of the box with classical detection (Gabor filter), but the ML mode needs training data from your own blocks before it becomes useful.

## How It Works

The pipeline takes a block boundary polygon and runs seven stages:

1. **Tile acquisition** — fetches aerial tiles (LINZ NZ imagery at ~0.1m/px), stitches them, masks to the polygon boundary
2. **Preprocessing** — converts RGB to derived channels (ExG vegetation index, luminance, structure tensor) and fuses them
3. **FFT orientation** — finds row angle and spacing from the 2D frequency domain
4. **Ridge likelihood** — builds a per-pixel probability map of row centerlines using either a trained FPN model (ML mode, default) or a Gabor bandpass filter (classical mode)
5. **Candidate extraction** — slices the image into strips perpendicular to the row direction, finds peaks in each strip
6. **Row tracking** — connects candidates across strips using the Hungarian algorithm
7. **Centerline fitting** — fits cubic smoothing splines, trims weak endpoints, detects gaps

Each stage reduces uncertainty one dimension at a time: FFT gives angle/spacing, ridge detection gives per-pixel likelihood, strip extraction gives candidate positions, tracking gives row identity, fitting gives the final curves.

## Installation

Requires Python 3.11 or 3.12 (not 3.14 — scipy hangs on 3.14).

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

pip install .              # core pipeline
pip install ".[ml]"        # adds PyTorch + training dependencies
pip install ".[gui]"       # adds FastAPI for the browser GUI
```

Copy `.env.example` to `.env` and add your LINZ API key (free from [data.linz.govt.nz](https://data.linz.govt.nz/)).

## GUI

The browser-based interface handles the full workflow without needing the terminal. Start it with:

```bash
python -m gui.server
```

Opens at `http://127.0.0.1:8765` with six tabs:

**Add Blocks** — MapLibre map with LINZ aerial tiles. Draw block boundary polygons, saved with anonymous hex names. Previously drawn boundaries show as coloured overlays.

**Annotate** — Step through detected blocks sequentially. Three choices per block: **Accept** (detection looks good), **Edit** (opens the matplotlib annotation editor to drag control points, add/delete rows), or **Blind** (draw rows from scratch on the raw aerial image without seeing any pipeline output — this creates unbiased ground truth).

**Library** — Grid of all blocks with thumbnails, stage badges (draft / detected / annotated / verified), and per-block actions. Each card shows a difficulty rating (1–5, set manually) and a region dropdown (auto-detected from boundary coordinates, manually overridable). Click any thumbnail to inspect in a full-screen lightbox with zoom and pan.

**Train** — Generate training patches from annotations, then train the FPN model with live progress (epoch count, loss, Dice score). After training, click "Clear All Detections" to re-detect with the new model.

**Verify** — Run detection on N random blocks and view results as a scrollable grid of overlay images with per-block metrics.

**Progress** — Tracks pipeline improvement over time. Five panels: runs timeline (with bootstrap CIs on F1), per-region F1 breakdown (detects if adding blocks in one region degrades another), learning curve (power-law fit predicting the F1 ceiling), block trajectories (worst blocks over time), and paired run comparison with statistical significance testing.

### Tuning

Click **Tune** on any detected block in the Library to open the parameter tuning panel. Nine adjustable parameters spanning ridge detection, tracking, and fitting stages. After adjusting, click "Run with these params" to see a side-by-side before/after comparison. Toggle "Show Diff" to overlay tuned rows (magenta) on the default overlay. If the tuned result is better, click "Apply as Default" to promote it.

**Ensemble confidence mode**: runs both ML and Gabor as independent methods, then scores each row by how well they agree. Green = both methods found the row. Red = only one did. The most reliable way to spot phantom detections without checking every row.

## CLI Usage

```bash
python cli.py --block "a3f2c1"           # single block by name
python cli.py --all                       # all blocks in the registry
python cli.py --geojson boundary.geojson  # from a GeoJSON file
python evaluate_gt.py --report            # ground truth evaluation
python visual_verify.py                   # generate overlay images
```

## Training

The ML ridge mode uses an FPN decoder with a MobileNet-v2 encoder. To train:

```bash
python generate_training_data.py                       # create patches from annotations
python -m training.train --decoder fpn                  # train
python -m training.train --decoder fpn --calibrate      # train + temperature scaling
```

Or use the GUI's Train tab, which handles patch generation and training with live progress streaming.

**Temperature scaling** (`--calibrate`): learns a single scalar that rescales model confidence values across a meaningful range instead of everything clustering near 1.0. Improves downstream stages (endpoint trimming, gap detection) that rely on confidence differences.

### Tips for Better Results

- **Use blind annotation for some blocks.** Editing the pipeline's own output creates a circular dependency — the model learns to reproduce its own mistakes. Blind annotation breaks this.
- **Centre rows on the physical canopy, not the brightness peak.** Sun angle shifts the bright side off-centre.
- **Annotate diverse blocks.** A few hard blocks (grassed inter-rows, curves, buildings) teach more than ten easy ones.
- **Retrain periodically.** After annotating 5–10 new blocks, regenerate training data and retrain.

## Evaluation

`evaluate_gt.py` reports metrics at three match thresholds:

- **Loose (0.4x spacing)** — ~1m tolerance
- **Medium (0.2x spacing)** — ~0.5m, catches off-centre matches
- **Strict (0.1x spacing)** — ~0.25m, only well-placed rows

Also reports shape error (mean point-to-polyline distance between detected and GT curves), separate means for blind vs pipeline-seeded annotations, and per-block breakdowns.

Evaluation results are automatically recorded to the tracking system for the Progress view.

## Tracking and Progress

The tracking system records structured data from every training run, evaluation, and parameter tuning event. Stored in `tracking/runs.json` and `tracking/per_block_results.json` (both created automatically on first use).

Each run record includes aggregate F1 at three thresholds with bootstrap confidence intervals, per-region breakdowns, failure mode counts, and config diffs for tuning runs. The Progress tab in the GUI visualises this data across runs so you can see whether changes are actually helping.

**Region detection**: blocks are automatically assigned to a wine region based on their boundary centroid (nearest-neighbor lookup against `data/nz_wine_regions.json`). The Progress view shows per-region F1 over time — the key chart for catching "adding blocks in region X degraded region Y."

## Project Structure

```
vinerow/              Detection pipeline (7 stages)
  acquisition/          Tile fetching and geo utilities
  preprocessing/        Multi-channel image preprocessing
  orientation/          2D FFT angle/spacing detection
  ridge/                Ridge likelihood (ML and classical)
  candidates/           Strip-based candidate extraction
  tracking/             Hungarian assignment + stitching
  fitting/              Spline fitting, gap detection, trimming
  postprocessing/       Metrics, quality flags

gui/                  Browser-based GUI (FastAPI + Alpine.js)
  routers/              API endpoints (blocks, detection, annotation,
                        training, verify, progress, tiles, elevation)
  services/             Block registry, detection cache, task runner
  static/               Frontend (HTML, CSS, JS — no build step)

training/             ML model training (FPN + MobileNet-v2)
tracking/             Progress tracking (storage, metrics, hooks)
blocks/               Region detection logic
block_detection/      Experimental block boundary detection (separate model)
data/                 Region reference data, block registry
```

## Ridge Detection Modes

| Mode | Method | When to use |
|------|--------|-------------|
| `ml` (default) | Trained FPN model | General use, handles grassed inter-rows well |
| `gabor` | Gabor bandpass filter | High-contrast blocks, bare soil inter-rows, no training needed |
| `ml_ensemble` | Max of ML + Gabor | When ML alone struggles on a particular block |
| `hessian` | Hessian eigenvalue | Alternative classical method |

## Requirements

- Python 3.11 or 3.12
- LINZ API key (free) for NZ aerial imagery
- The annotation editor uses matplotlib in a subprocess, so the GUI must run locally (not on a headless server)

## Experimental Features

**LiDAR canopy height**: For New Zealand blocks with LiDAR coverage, the pipeline can fetch free 1m elevation data from LINZ (via AWS Open Data). Computes canopy height (DSM minus DEM) to validate that a drawn polygon is actually a vineyard and flag areas with trees or buildings. Requires `rasterio` (`pip install rasterio`), currently limited to Marlborough datasets.

## License

MIT — see [LICENSE](LICENSE).
