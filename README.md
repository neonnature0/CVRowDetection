# Vineyard Row Detection from Aerial Imagery

Automated detection of vine row positions, orientation, and spacing from aerial/satellite imagery. Given a vineyard block boundary polygon, the pipeline fetches aerial tiles, isolates the block region, and runs a 7-stage computer vision pipeline to detect individual row centerlines as curved polylines.

Used in production as part of [Cordyn](https://cordyn.app) — a vineyard management platform — to auto-populate block geometry for spray planning, planting records, and compliance tracking.

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
cv-row-detection/
|
|-- vinerow/                    # Production pipeline package
|   |-- acquisition/            #   Tile fetching, geo utilities
|   |-- preprocessing/          #   Multi-channel image processing
|   |-- orientation/            #   2D FFT angle/spacing detection
|   |-- ridge/                  #   Ridge likelihood (Gabor + ML modes)
|   |-- candidates/             #   Strip-based candidate extraction
|   |-- tracking/               #   Hungarian row tracking
|   |-- fitting/                #   Spline centerline fitting
|   |-- postprocessing/         #   Metrics, quality flags
|   |-- config.py               #   All tunable parameters
|   |-- pipeline.py             #   Main orchestrator
|   +-- types.py                #   Data types (FittedRow, etc.)
|
|-- training/                   # ML training pipeline
|   |-- dataset.py              #   PyTorch Dataset + augmentations
|   |-- model.py                #   U-Net model definition
|   |-- train.py                #   Training loop (BCE + Dice loss)
|   |-- predict.py              #   Full-block patch-stitched inference
|   +-- checkpoints/            #   Saved model weights
|
|-- dataset/
|   |-- annotations/            #   Human-verified row positions (JSON)
|   |-- images/                 #   Cached block aerial images + masks
|   +-- training/               #   Generated patches + targets for ML
|
|-- cli.py                      # Command-line interface
|-- benchmark.py                # Regression benchmark (all blocks)
|-- evaluate_gt.py              # Ground truth evaluation (P/R/F1)
|-- annotate.py                 # Interactive annotation tool (matplotlib)
|-- visual_verify.py            # Generate overlay verification images
|-- generate_training_data.py   # Convert annotations to ML training patches
|-- prepare_dataset.py          # Prepare annotation JSONs from pipeline output
+-- test_blocks.json            # Block definitions (boundaries + ground truth)
```

## Setup

### 1. Create virtual environment

```bash
cd apps/web/scripts/cv-row-detection
python -m venv venv        # Use Python 3.11 or 3.12 (not 3.14 — scipy issues)

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For ML training/inference (optional):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install segmentation-models-pytorch albumentations
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

1. Add the block boundary to `test_blocks.json` (GeoJSON polygon, `[longitude, latitude]` coordinate order)
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
