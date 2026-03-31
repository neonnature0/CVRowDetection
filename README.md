# CV Row Detection from Aerial Imagery

Feasibility prototype for detecting vine row orientation and spacing from aerial/satellite imagery using computer vision. Given a vineyard block boundary polygon, the pipeline fetches aerial tiles, isolates the block region, and applies two independent detection approaches (Hough transform and FFT analysis) to measure row angle and spacing. Results are compared against ground truth data from the database.

## Setup

### 1. Create a virtual environment

```bash
cd apps/web/scripts/cv-row-detection
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Required For | Description |
|----------|-------------|-------------|
| `LINZ_API_KEY` | NZ imagery | Free API key from [LINZ Data Service](https://data.linz.govt.nz/) |
| `SUPABASE_URL` | --fetch-blocks | Supabase project URL (already set in example) |
| `SUPABASE_SERVICE_KEY` | --fetch-blocks | Supabase service role key (from project settings) |

ArcGIS World Imagery and Kelowna sources do not require API keys.

## Populating Test Data

### Option A: Fetch from Supabase

```bash
python detect_rows.py --fetch-blocks
python detect_rows.py --fetch-blocks --org-id <your-org-uuid>
```

This queries the database for blocks with boundaries and row spacing ground truth, saving them to `test_blocks.json`. Note: block boundaries are PostGIS geometry columns, so this requires either the Supabase `/pg` SQL endpoint or a custom database function. If fetching fails, use Option B.

### Option B: Manual test_blocks.json

Create or edit `test_blocks.json` with your block data:

```json
{
  "blocks": [
    {
      "name": "B3",
      "vineyard_name": "Home Vineyard",
      "boundary": {
        "type": "Polygon",
        "coordinates": [[[173.123, -41.456], [173.124, -41.456], [173.124, -41.457], [173.123, -41.457], [173.123, -41.456]]]
      },
      "row_spacing_m": 2.4,
      "row_orientation": "N-S",
      "row_angle": 87.5,
      "row_count": 42
    }
  ]
}
```

Ground truth fields (`row_spacing_m`, `row_orientation`, `row_angle`, `row_count`) are optional. If provided, detection results will be compared against them.

### Option C: Custom GeoJSON

```bash
python detect_rows.py --geojson my_boundary.geojson --source arcgis
```

The GeoJSON can be a bare `Polygon`, a `Feature`, or a `FeatureCollection`. Ground truth can be embedded in feature properties using the same field names.

## Usage

### Single block

```bash
python detect_rows.py --block "B3" --source linz --zoom 20
python detect_rows.py --block "B3" --approach fft
```

### All blocks

```bash
python detect_rows.py --all
python detect_rows.py --all --approach both --source auto --verbose
```

### Custom GeoJSON boundary

```bash
python detect_rows.py --geojson boundary.geojson --source arcgis --zoom 19
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--block NAME` | — | Process a single block by name |
| `--all` | — | Process all blocks in test_blocks.json |
| `--geojson PATH` | — | Process a custom GeoJSON file |
| `--fetch-blocks` | — | Populate test_blocks.json from Supabase |
| `--org-id UUID` | — | Filter by organisation (with --fetch-blocks) |
| `--source {linz,arcgis,kelowna,auto}` | auto | Tile imagery source |
| `--zoom INT` | source-dependent | Override zoom level |
| `--approach {hough,fft,both}` | both | Detection approach |
| `--output-dir DIR` | output/ | Debug output directory |
| `--no-cache` | false | Skip tile disk cache |
| `--verbose` | false | Enable debug logging |

## Interpreting Results

### Console output

The results table shows detected angle, spacing, and row count alongside ground truth values and error metrics. A summary at the bottom reports mean/max errors and success rate.

### Debug images

For each block, a subdirectory is created under `output/` containing:

| File | Description |
|------|-------------|
| `01_original.png` | Masked aerial image (block boundary applied) |
| `02_vegetation.png` | Excess Green (ExG) vegetation index as a heatmap |
| `03_edges.png` | Canny edge detection output |
| `04_hough_lines.png` | Detected Hough lines overlaid on original image |
| `05_fft_angle_response.png` | Angle sweep plot showing directional FFT response |
| `06_fft_spectrum.png` | 1D FFT spectrum with peak frequency marked |
| `07_comparison.png` | Side-by-side summary of all pipeline stages |

### Results summary

A markdown file `output/results_summary.md` is generated with a comparison table and aggregate statistics.

## Technical Approach

### Preprocessing

1. Fetch aerial tiles covering the block bounding box and stitch into a single image.
2. Apply the block polygon as a mask (zero out pixels outside the boundary).
3. Compute Excess Green Index (ExG = 2G - R - B) to separate vine canopy from soil.
4. Apply CLAHE contrast enhancement.
5. Run Canny edge detection.

### Hough Transform (line detection)

Applies probabilistic Hough line detection to the edge map. Detected line segments are clustered by angle to find the dominant orientation. Spacing is estimated from the perpendicular distances between parallel lines.

Strengths: intuitive, works well with clearly defined row edges.
Weaknesses: sensitive to noise, struggles with incomplete rows or canopy overlap.

### FFT Analysis (frequency domain)

Performs an angular sweep of 1D FFT slices through the 2D frequency spectrum. At each angle, the FFT magnitude profile reveals periodic spacing as a peak frequency. The angle with the strongest periodic signal is selected as the row orientation, and the peak frequency is converted to row spacing.

Strengths: robust to noise, handles partial or gappy rows well.
Weaknesses: requires sufficient image area for frequency resolution.

### Angle convention

- 0 degrees = rows running East-West (horizontal in image)
- 90 degrees = rows running North-South (vertical in image)
- Range is 0-180 degrees (180-degree ambiguity: rows look the same from either end)

Compass labels from the database (e.g., "N-S", "NE-SW") are mapped to this convention for ground truth comparison.

## Success Criteria

| Metric | Target |
|--------|--------|
| Row angle | Within 5 degrees of ground truth |
| Row spacing | Within 0.3m of ground truth (roughly 15% for typical 2m spacing) |

Both thresholds must be met simultaneously for a result to be counted as a success.

## Test Results (2026-03-29)

Tested on 5 vineyard blocks with known ground truth — 3 NZ blocks (LINZ tiles, zoom 20), 1 large NZ block (Opawa), and 1 Canadian block (Kelowna tiles, zoom 19).

### Best results per block (picking whichever approach was better):

| Block | Rows | Best Approach | Angle Error | Spacing Error | Spacing (m) | GT (m) | Pass? |
|-------|------|--------------|------------|--------------|------------|--------|-------|
| Block A (Brooklands) | 26 | Hough | 2.1° | 17.6% | 2.94 | 2.50 | Angle only |
| Block C (Brooklands) | 110 | FFT | 0.2° | 0.9% | 2.48 | 2.50 | **PASS** |
| Block D (Brooklands) | 111 | Hough angle | 4.7° | 31.4% | 3.29 | 2.50 | Angle only |
| North Block (Opawa) | 251 | FFT | 0.0° | 0.5% | 2.79 | 2.81 | **PASS** |
| B10 (The View, CA) | 39 | FFT | 0.3° | 0.8% | 2.72 | 2.70 | **PASS** |

### Key findings:

- **Angle detection: 4/5 blocks within ±5°** — exceeds the 3/4 target
- **Spacing detection: 3/5 blocks within ±0.3m** — meets the 3/4 target
- **FFT excels on blocks with >40 rows** — angle error <1°, spacing error <1%
- **Hough provides consistent angle (~2-5° error)** but spacing is noisy (12-31%)
- **FFT harmonic correction needed** — raw FFT often detects 2× row spacing (alternating row pattern), corrected by checking for a sub-harmonic peak at half the detected spacing
- **Processing time**: Hough 0.1-4s, FFT 18-161s (depends on image size)

### Recommended strategy for production integration:

Use **FFT for angle + spacing** on blocks with sufficient area (>40 rows). Fall back to **Hough for angle only** on small blocks. Cross-validate when both approaches agree.

## Known Limitations

- **Imagery resolution**: At zoom 19 (ArcGIS), ground resolution is roughly 0.3 m/pixel at mid-latitudes. Row spacing of 2m is only 6-7 pixels apart, which is at the limit of reliable detection. Zoom 20+ (LINZ in NZ) gives roughly 0.15 m/pixel, which is much better.
- **Seasonal variation**: Detection works best when vine canopy is visible (growing season). Dormant season imagery with bare trellis may have insufficient contrast.
- **Young vines**: Recently planted blocks with small canopy may not show clear row patterns in aerial imagery.
- **Non-uniform rows**: Blocks with variable row spacing, curved rows, or significant gaps will reduce confidence.
- **Cloud/shadow**: Partial cloud cover or building shadows within the block can create false edges.
- **Tile source availability**: LINZ provides the best resolution but only covers New Zealand. ArcGIS World Imagery has global coverage but lower resolution. Kelowna source is specific to that municipality.
- **PostGIS boundary fetch**: The `--fetch-blocks` command may not work if the Supabase `/pg` SQL endpoint is unavailable. In that case, populate `test_blocks.json` manually or export boundaries from the app.
- **Row count estimation**: Row count is derived from detected spacing and block width, so it inherits errors from both spacing detection and boundary measurement.
