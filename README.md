# Vineyard Row Detection from Aerial Imagery

Automated detection of vine row positions, orientation, and spacing from aerial/satellite imagery. Given a vineyard block boundary polygon, the pipeline fetches aerial tiles, isolates the block region, and runs a 7-stage computer vision pipeline to detect individual row centerlines as curved polylines.

Designed to auto-populate vineyard block geometry for spray planning, planting records, and compliance tracking.

## Results

Tested on 12 vineyard blocks across 5 vineyards (New Zealand + Canada):

| Metric | Score |
|--------|-------|
| **Mean F1** | 95.9% |
| **Mean Precision** | 98.2% |
| **Mean Recall** | 93.9% |
| **Localization Error** | 0.107m |
| **Blocks with F1 > 95%** | 7 / 11 |

Evaluated against human-annotated ground truth using bipartite (Hungarian) matching.

## How It Works

The pipeline runs 7 sequential stages:

```
Input: Block boundary polygon (GeoJSON)
  |
  v
Stage 0: Tile Acquisition -----> Fetch aerial tiles, stitch, mask to polygon
Stage 1: Multi-Channel Preproc -> ExG, luminance, normalized vegetation, structure tensor
Stage 2: 2D FFT Orientation ----> Coarse row angle + spacing from frequency domain
Stage 3: Ridge Likelihood ------> Per-pixel row probability map (Gabor filter or U-Net)
Stage 4: Candidate Extraction --> Strip-based adaptive peak finding along perpendicular axis
Stage 5: Row Tracking ----------> Bidirectional Hungarian assignment across strips
Stage 6: Centerline Fitting ----> Cubic smoothing splines + geographic coordinate conversion
Stage 7: Post-Processing -------> Spacing stats, quality flags, confidence scoring
  |
  v
Output: List of FittedRow objects (polyline centerlines in pixel + geo coords)
```

### Stage 3: Ridge Detection Modes

| Mode | Method | Best For |
|------|--------|----------|
| `gabor` (default) | Gabor bandpass filter tuned to row frequency | High-contrast blocks, bare soil inter-rows |
| `ml` | U-Net (MobileNet-v2 encoder) trained on annotated data | Low-contrast blocks, grassed inter-rows |
| `ml_ensemble` | Max of Gabor and U-Net per pixel | Mixed conditions |
| `hessian` | Hessian eigenvalue ridge detection on all channels | Alternative classical method |
| `ensemble` | Max of Hessian and Gabor | Broader classical coverage |

## Tile Sources

Aerial imagery is automatically selected by longitude:

| Source | Coverage | Resolution | API Key Required |
|--------|----------|-----------|-----------------|
| **LINZ** | New Zealand | ~0.1m/px (zoom 20) | Yes (free from [data.linz.govt.nz](https://data.linz.govt.nz/)) |
| **Kelowna** | Okanagan, BC, Canada | ~0.2m/px (zoom 19) | No |
| **ArcGIS World Imagery** | Global fallback | ~0.3m/px (zoom 19) | No |

## Project Structure

```
CVRowDetection/
|
|-- vinerow/                    # Core row detection library
|   |-- acquisition/            #   Tile fetching, geo utilities
|   |-- preprocessing/          #   Multi-channel image processing
|   |-- orientation/            #   2D FFT angle/spacing detection
|   |-- ridge/                  #   Ridge likelihood (Gabor + ML modes)
|   |-- candidates/             #   Strip-based candidate extraction
|   |-- tracking/               #   Hungarian row tracking
|   |-- fitting/                #   Spline centerline fitting
|   |-- detection/              #   Global profile row detection method
|   |-- postprocessing/         #   Metrics, quality flags
|   |-- loaders/                #   Pluggable block-data backends (JSON, GeoJSON, Supabase)
|   |-- debug/                  #   Diagnostics and debug artifacts
|   |-- config.py               #   All tunable parameters
|   |-- pipeline.py             #   Main orchestrator
|   +-- types.py                #   Data types (FittedRow, etc.)
|
|-- block_detection/            # Block boundary detection model (separate from row detection)
|   |-- encoder.py              #   SharedEncoder (MobileNet-v2 + FPN)
|   |-- heads/                  #   Detection heads
|   |-- predict_blocks.py       #   Inference
|   +-- train_blocks.py         #   Training
|
|-- training/                   # ML row model training
|   |-- dataset.py              #   PyTorch Dataset + augmentations
|   |-- model.py                #   U-Net model definition
|   |-- train.py                #   Training loop (BCE + Dice loss)
|   +-- predict.py              #   Full-block patch-stitched inference
|
|-- data/
|   +-- blocks/
|       +-- test_blocks.json    #   Benchmark block definitions (12 blocks, inline GeoJSON)
|
|-- models/                     # Pre-trained model weights (download via models/download_models.py)
|
|-- gui/                        # Future: interactive frontend (planned)
|
|-- dataset/                    # (gitignored) Training data, cached images, annotations
|
|-- cli.py                      # Command-line interface
|-- benchmark.py                # Regression benchmark (all blocks)
|-- evaluate_gt.py              # Ground truth evaluation (P/R/F1)
|-- annotate.py                 # Interactive annotation tool (matplotlib)
|-- map_annotator.py            # Web-based map annotation (Leaflet)
|-- visual_verify.py            # Generate overlay verification images
|-- generate_training_data.py   # Convert annotations to ML training patches
+-- prepare_dataset.py          # Prepare annotation JSONs from pipeline output
```

## Setup

### 1. Create virtual environment

```bash
cd CVRowDetection
python -m venv venv        # Use Python 3.11 or 3.12 (not 3.14 — scipy issues)

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install .
```

For ML training/inference (optional):

```bash
pip install ".[ml]"
```

### 3. Configure environment

```bash
cp .env.example .env
```

| Variable | Required For | Description |
|----------|-------------|-------------|
| `LINZ_API_KEY` | NZ imagery | Free API key from [LINZ Data Service](https://data.linz.govt.nz/) |

ArcGIS and Kelowna sources do not require API keys.

## Usage

### Run the pipeline on a single block

```bash
python cli.py --block "Block C"
python cli.py --block "B10" --ridge-mode gabor
```

### Benchmark all blocks

```bash
python benchmark.py
python benchmark.py --blocks "Block C,B10" --ridge-mode gabor
```

### Evaluate against ground truth

```bash
python evaluate_gt.py --status complete --report
```

Produces per-block precision/recall/F1 table + overlay images in `dataset/evaluation/`.

### Visual verification

```bash
python visual_verify.py --overlay-only
python visual_verify.py --blocks "Main Block" --overlay-only
```

Generates row overlay images in `output/visual_verify/`.

### Annotate blocks

```bash
python annotate.py --block "Block C"
```

Interactive matplotlib tool for correcting row positions. Controls:

| Action | Input |
|--------|-------|
| Select row | Left-click near row |
| Drag control point | Left-click + drag on selected row's point |
| Pan | Right-drag |
| Zoom | Scroll wheel |
| Add row | Press `A`, click on image |
| Delete row | Press `D`, click on row |
| Insert control point | Press `I` or right-click on selected row |
| Remove control point | Press `X` or `Delete` near point |
| Save | Press `S` |
| Mark complete | Press `M` |

## ML Training

### Generate training data

```bash
python generate_training_data.py --patch-size 256 --overlap 0.25
```

Produces paired image patches and row-likelihood heatmaps from annotated blocks.

### Train the U-Net

```bash
python -m training.train --epochs 100 --batch-size 8 --lr 1e-4 --patience 20
```

Trains a U-Net with MobileNet-v2 encoder on CPU. Checkpoints saved to `training/checkpoints/`.

### Evaluate trained model

```bash
python -m training.train --run-test --checkpoint training/checkpoints/best_model.pth
```

## Adding New Blocks

1. Add the block boundary to `data/blocks/test_blocks.json` (GeoJSON polygon, `[longitude, latitude]` coordinate order)
2. Run `python prepare_dataset.py` to fetch tiles and create annotation JSON
3. Run `python annotate.py --block "YourBlock"` to correct row positions
4. Run `python generate_training_data.py` to regenerate ML training patches
5. Retrain with `python -m training.train`

## Tech Stack

- **Python 3.11** with NumPy, SciPy, OpenCV
- **PyTorch** + segmentation-models-pytorch (U-Net with MobileNet-v2 encoder)
- **Albumentations** for training data augmentation
- **Matplotlib** for the annotation tool
- Aerial imagery from **LINZ** (NZ), **Kelowna Open Data** (Canada), **ArcGIS World Imagery** (global)

## Key Algorithms

- **2D FFT** for coarse orientation detection — periodic parallel rows produce conjugate peaks in the frequency domain whose polar coordinates encode angle and spacing
- **Gabor bandpass filter** for ridge likelihood — tuned to the detected row frequency, with phase-independent energy envelope and oriented suppression
- **Texture-based phase correction** — adaptive mechanism that detects when the Gabor locks onto inter-row features instead of vine canopy by comparing local texture at detected positions vs midpoints
- **Harmonic resolution** — spectral tiebreaker that checks for half-period confusion (row-to-gap vs row-to-row spacing) using sub-harmonic power analysis
- **Hungarian algorithm** for row tracking — bidirectional assignment from the densest strip, with position and strength cost weighting
- **Douglas-Peucker simplification** for converting dense polylines to sparse control points in the annotation tool

---

## Project History

This project evolved through several generations. The legacy files are kept in the repo for reference.

### Generation 1: Classical Prototypes (early 2026)

The original approach tried two independent methods to detect row angle and spacing:

**Hough Transform** (`hough_detector.py`) — ran Canny edge detection on the aerial image, then probabilistic Hough line detection to find line segments. Clustered segments by angle to find the dominant orientation. Spacing was estimated from perpendicular distances between parallel lines.

- Strengths: simple, intuitive
- Problems: extremely sensitive to noise, struggled with incomplete rows or canopy overlap, spacing accuracy was poor (12-31% error)
- **Verdict: abandoned** — too noisy for production use

**1D FFT Angle Sweep** (`fft_detector.py`) — took 1D FFT slices through the 2D frequency spectrum at many angles. The angle with the strongest periodic signal was the row orientation. Peak frequency gave spacing.

- Strengths: robust to noise, handled partial rows well
- Problems: slow (angular sweep is O(n) in angle resolution), couldn't handle curved rows
- **Verdict: replaced** by 2D FFT

**2D FFT** (`fft2d_detector.py`) — single 2D FFT on the whole image. Periodic parallel rows produce a conjugate pair of peaks whose polar coordinates directly encode angle and spacing. Much faster and more accurate than the 1D sweep.

- Result: angle error <1 degree, spacing error <1% on blocks with >40 rows
- **Verdict: kept** — absorbed into `vinerow/orientation/fft2d.py`

Supporting files from this generation:
- `detect_rows.py` — monolithic pipeline that orchestrated Hough + FFT
- `image_preprocessor.py` — ExG vegetation index + CLAHE contrast enhancement
- `row_locator.py` — grid-stamping approach that placed a fixed-spacing grid at the detected angle
- `tile_fetcher.py`, `geo_utils.py` — tile fetching and coordinate conversion (moved into `vinerow/acquisition/`)
- `debug_visualizer.py` — generated debug images for the old pipeline
- `test_angle_conversion.py` — one-off unit test for angle conventions

### Generation 2: Production Pipeline (`vinerow/` package)

Rebuilt from scratch as a modular 7-stage pipeline. Key improvements over Gen 1:

- **Multi-channel preprocessing** instead of just ExG — luminance, normalized vegetation, structure tensor anisotropy, with quality-weighted fusion
- **Gabor bandpass filter** instead of Hessian-only ridge detection — tuned to the FFT-detected row frequency for much better selectivity
- **Strip-based candidate extraction** instead of grid stamping — no spacing assumption, finds peaks purely from the data
- **Hungarian tracking** instead of nearest-neighbor — globally optimal row-to-candidate assignment, handles gaps and missing detections
- **Spline fitting** instead of straight lines — models curved rows with cubic smoothing splines
- **Harmonic resolution** — detects and corrects half-period confusion in the FFT (row-to-gap vs row-to-row)
- **Phase correction** — texture-based adaptive check that detects when the Gabor locks onto inter-row grass instead of vine canopy

Result: 95.9% mean F1 across 11 blocks, evaluated against human-annotated ground truth.

### Generation 3: ML Ridge Detection (in progress)

Training a lightweight U-Net (MobileNet-v2 encoder, ~6.6M params) to replace the Gabor filter in Stage 3. The model takes RGB image patches as input and predicts a per-pixel row-likelihood heatmap. Everything else in the pipeline stays the same.

Motivation: the Gabor filter struggles on blocks where both vine canopy and inter-row grass are green (low spectral contrast). A learned model can pick up on subtler texture and color patterns that hand-crafted filters miss.

- Training data: 1,988 patches (256x256) from 11 annotated blocks
- Architecture: U-Net with MobileNet-v2 encoder, BCE + Dice loss
- Training: CPU-only, ~12 min/epoch, early stopping

### Legacy Files

Generation 1 files (hough_detector, fft_detector, detect_rows, etc.) were deleted during
the standalone decoupling. They are recoverable from git tag `pre-decouple`.
