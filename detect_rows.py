#!/usr/bin/env python3
"""
CV Row Detection from Aerial Imagery — Feasibility Prototype

Detects vine row orientation and spacing from aerial/satellite imagery
within a drawn block boundary using computer vision.

Usage:
    python detect_rows.py --block "B3" --source linz --zoom 20
    python detect_rows.py --all --approach both
    python detect_rows.py --fetch-blocks --org-id <uuid>
    python detect_rows.py --geojson boundary.geojson --source arcgis --zoom 19
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from geo_utils import polygon_bbox, meters_per_pixel
from tile_fetcher import (
    TILE_SOURCES,
    fetch_and_stitch,
    auto_select_source,
    default_zoom,
    get_api_key,
)
from image_preprocessor import preprocess
from hough_detector import detect as hough_detect
from fft_detector import detect as fft_detect
from fft2d_detector import detect as fft2d_detect
from debug_visualizer import save_pipeline_debug, create_comparison_summary

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compass-to-angle mapping
# ---------------------------------------------------------------------------

# In our image coordinate system:
#   0°  = rows running East-West (horizontal)
#   90° = rows running North-South (vertical)
# Row orientation from the DB is a compass label describing the direction
# the rows run. We convert to our 0-180° convention below.
# Note: 180° ambiguity — N-S and S-N are the same orientation.

COMPASS_TO_DEGREES: dict[str, float] = {
    'N-S': 90.0,   'S-N': 90.0,
    'E-W': 0.0,    'W-E': 0.0,
    'NE-SW': 45.0, 'SW-NE': 45.0,
    'NW-SE': 135.0, 'SE-NW': 135.0,
    # Single-direction compass labels (legacy/alternate formats)
    'N': 90.0,  'S': 90.0,
    'E': 0.0,   'W': 0.0,
    'NE': 45.0, 'SW': 45.0,
    'NW': 135.0, 'SE': 135.0,
    'NNE': 67.5, 'SSW': 67.5,
    'ENE': 22.5, 'WSW': 22.5,
    'NNW': 112.5, 'SSE': 112.5,
    'ESE': 157.5, 'WNW': 157.5,
}

# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """Result of one detection run (one approach on one block)."""
    block_name: str
    vineyard_name: str
    approach: str               # 'hough' or 'fft'
    detected_angle: float       # degrees, 0-180
    detected_spacing_m: float   # meters
    detected_row_count: int
    confidence: float           # 0.0 - 1.0
    ground_truth_spacing: float | None
    ground_truth_orientation: str | None
    ground_truth_row_count: int | None
    spacing_error_pct: float | None
    angle_error_deg: float | None
    processing_time_s: float


# ---------------------------------------------------------------------------
# Test block loading
# ---------------------------------------------------------------------------


def load_test_blocks(path: str = 'test_blocks.json') -> list[dict]:
    """Load test block data from JSON file.

    Expected format:
    {
      "blocks": [
        {
          "name": "B3",
          "vineyard_name": "Example Vineyard",
          "boundary": { "type": "Polygon", "coordinates": [[[lng,lat],...]] },
          "row_spacing_m": 2.4,
          "row_orientation": "N-S",
          "row_angle": 87.5,
          "row_count": 42
        },
        ...
      ]
    }

    Args:
        path: Path to the JSON file.

    Returns:
        List of block dicts.
    """
    resolved = Path(path)
    if not resolved.exists():
        logger.error("Test blocks file not found: %s", resolved)
        logger.info("Run with --fetch-blocks to populate it from Supabase, "
                     "or create it manually.")
        return []

    with open(resolved, 'r', encoding='utf-8') as f:
        data = json.load(f)

    blocks = data.get('blocks', [])
    logger.info("Loaded %d test blocks from %s", len(blocks), resolved)
    return blocks


def fetch_blocks_from_supabase(org_id: str | None = None) -> list[dict]:
    """Query Supabase REST API for blocks with boundaries and ground truth.

    Uses SUPABASE_URL and SUPABASE_SERVICE_KEY env vars.
    Because block boundaries are PostGIS geometry columns (not directly
    accessible as GeoJSON via PostgREST SELECT), we use the Supabase
    SQL RPC endpoint to run a query that calls ST_AsGeoJSON.

    Args:
        org_id: Optional organisation account ID to filter by.

    Returns:
        List of block dicts suitable for processing.

    Raises:
        SystemExit: If required env vars are not set.
    """
    supabase_url = os.environ.get('SUPABASE_URL')
    service_key = os.environ.get('SUPABASE_SERVICE_KEY')

    if not supabase_url or not service_key:
        logger.error(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env to fetch blocks.\n"
            "Copy .env.example to .env and fill in the values."
        )
        sys.exit(1)

    # Build SQL query
    where_clause = ""
    if org_id:
        # Sanitize: only allow UUID characters
        safe_org_id = ''.join(c for c in org_id if c in '0123456789abcdef-')
        where_clause = f"AND b.organisation_id = '{safe_org_id}'"

    sql = f"""
    SELECT
        b.name,
        v.name AS vineyard_name,
        ST_AsGeoJSON(b.boundary)::json AS boundary_geojson,
        b.row_spacing_m,
        b.row_orientation,
        b.row_angle,
        p.total_row_count AS row_count
    FROM vineyard.blocks b
    JOIN vineyard.vineyards v ON v.id = b.vineyard_id
    LEFT JOIN vineyard.plantings p ON p.block_id = b.id AND p.is_current = true
    WHERE b.boundary IS NOT NULL
    {where_clause}
    ORDER BY v.name, b.name
    """

    logger.info("Fetching blocks from Supabase...")
    logger.debug("SQL: %s", sql.strip())

    # Use the Supabase SQL endpoint (pg_graphql or rpc)
    # The standard approach is POST to /rest/v1/rpc/sql if such a function exists.
    # Alternatively, use the management API. We'll try the PostgREST RPC approach first.
    headers = {
        'apikey': service_key,
        'Authorization': f'Bearer {service_key}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation',
    }

    # Try the Supabase SQL endpoint (available via supabase-js but also as REST)
    # POST /rest/v1/rpc/{function_name} is the standard pattern.
    # Since we don't have a custom RPC function, we'll use the pg_net approach
    # or fall back to a simpler PostgREST query.

    # Approach: Use the Supabase Management API SQL endpoint
    # POST {url}/rest/v1/rpc — doesn't work without a defined function.
    # Instead, use the /pg endpoint if available, or the query endpoint.

    # Most reliable: use the Supabase project's SQL query via the management API.
    # But that requires a different auth token. Let's try a simpler approach:
    # Create a temporary RPC or just use PostgREST with computed columns.

    # Simplest reliable approach: use the pg endpoint
    # Supabase exposes: POST /rest/v1/rpc/... for custom functions
    # Since we can't guarantee a custom function exists, let's use the
    # raw SQL approach via the Supabase Data API (requires service role key).

    # The /pg endpoint is available at the project level for SQL queries
    pg_url = f"{supabase_url}/pg"

    try:
        # Try the Supabase pg endpoint (available since Supabase GA)
        resp = requests.post(
            pg_url,
            headers=headers,
            json={"query": sql.strip()},
            timeout=30,
        )

        if resp.status_code == 404:
            # pg endpoint not available — fall back to PostgREST approach
            logger.warning("Supabase /pg endpoint not available, trying PostgREST approach")
            return _fetch_blocks_postgrest(supabase_url, headers, org_id)

        resp.raise_for_status()
        rows = resp.json()

    except requests.exceptions.RequestException as exc:
        logger.warning("Failed to use /pg endpoint: %s. Trying PostgREST.", exc)
        return _fetch_blocks_postgrest(supabase_url, headers, org_id)

    # Handle different response formats
    if isinstance(rows, dict):
        rows = rows.get('rows', rows.get('data', []))

    blocks = []
    for row in rows:
        boundary_geojson = row.get('boundary_geojson')
        if boundary_geojson is None:
            continue

        # Ensure boundary_geojson is parsed
        if isinstance(boundary_geojson, str):
            boundary_geojson = json.loads(boundary_geojson)

        blocks.append({
            'name': row.get('name', 'Unknown'),
            'vineyard_name': row.get('vineyard_name', 'Unknown'),
            'boundary': boundary_geojson,
            'row_spacing_m': row.get('row_spacing_m'),
            'row_orientation': row.get('row_orientation'),
            'row_angle': row.get('row_angle'),
            'row_count': row.get('row_count'),
        })

    logger.info("Fetched %d blocks with boundaries from Supabase", len(blocks))
    return blocks


def _fetch_blocks_postgrest(
    supabase_url: str,
    headers: dict,
    org_id: str | None,
) -> list[dict]:
    """Fallback: fetch blocks via PostgREST SELECT.

    PostgREST cannot call ST_AsGeoJSON on geometry columns directly,
    so this approach only works if blocks have a text/jsonb boundary
    column or if a database view exposes boundary as GeoJSON.

    If the boundary column is raw geometry, this will return blocks
    without boundaries, and the user will need to populate
    test_blocks.json manually or create a database view/function.

    Args:
        supabase_url: Supabase project URL.
        headers: Request headers with auth.
        org_id: Optional org filter.

    Returns:
        List of block dicts (may be empty if boundaries aren't readable).
    """
    url = f"{supabase_url}/rest/v1/blocks"
    params = {
        'select': 'name,row_spacing_m,row_orientation,row_angle,vineyard:vineyards(name)',
        'boundary': 'not.is.null',
    }
    if org_id:
        params['organisation_id'] = f'eq.{org_id}'

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
    except requests.exceptions.RequestException as exc:
        logger.error("PostgREST query failed: %s", exc)
        logger.info(
            "The boundary column is a PostGIS geometry type and cannot be read "
            "as GeoJSON via PostgREST without a view or function.\n"
            "Options:\n"
            "  1. Create test_blocks.json manually with GeoJSON boundaries\n"
            "  2. Create a Supabase SQL function that returns ST_AsGeoJSON\n"
            "  3. Export blocks from the app UI"
        )
        return []

    blocks = []
    for row in rows:
        vineyard_info = row.get('vineyard', {})
        vineyard_name = vineyard_info.get('name', 'Unknown') if isinstance(vineyard_info, dict) else 'Unknown'

        blocks.append({
            'name': row.get('name', 'Unknown'),
            'vineyard_name': vineyard_name,
            'boundary': None,  # Cannot read geometry via PostgREST
            'row_spacing_m': row.get('row_spacing_m'),
            'row_orientation': row.get('row_orientation'),
            'row_angle': row.get('row_angle'),
            'row_count': None,
        })

    if blocks:
        logger.warning(
            "Fetched %d blocks but boundaries are NOT available via PostgREST. "
            "You need to add GeoJSON boundaries manually to test_blocks.json, "
            "or create a database function that returns ST_AsGeoJSON.",
            len(blocks),
        )
    return blocks


# ---------------------------------------------------------------------------
# Ground truth comparison
# ---------------------------------------------------------------------------


def compare_with_ground_truth(
    detected_angle: float,
    detected_spacing: float,
    detected_count: int,
    ground_truth: dict,
) -> dict:
    """Compare detection results against known ground truth.

    Handles the 180-degree angle ambiguity: row orientation is the same
    at angle X and angle X+180 (rows running NE-SW look the same from
    either end). We compute the minimum angular distance considering
    this wrap-around.

    Args:
        detected_angle: Detected row angle in degrees (0-180).
        detected_spacing: Detected row spacing in meters.
        detected_count: Detected row count.
        ground_truth: Dict with optional keys:
            row_spacing_m, row_orientation, row_angle, row_count

    Returns:
        Dict with keys: spacing_error_pct, angle_error_deg, row_count_error.
        Values are None if the corresponding ground truth is not available.
    """
    result = {
        'spacing_error_pct': None,
        'angle_error_deg': None,
        'row_count_error': None,
    }

    # Spacing comparison
    gt_spacing = ground_truth.get('row_spacing_m')
    if gt_spacing is not None and gt_spacing > 0:
        result['spacing_error_pct'] = abs(detected_spacing - gt_spacing) / gt_spacing * 100.0

    # Angle comparison
    # Prefer row_angle (numeric) over row_orientation (compass string)
    gt_angle = ground_truth.get('row_angle')
    if gt_angle is not None:
        # row_angle is stored as geographic bearing (0°=North, 90°=East)
        # Convert to image coordinates (0°=horizontal/E-W, 90°=vertical/N-S)
        # bearing θ → image direction: dx=sin(θ), dy=-cos(θ)
        # image angle = atan2(dy, dx) = atan2(-cos(θ), sin(θ))
        bearing_rad = math.radians(float(gt_angle))
        gt_angle_deg = math.degrees(
            math.atan2(-math.cos(bearing_rad), math.sin(bearing_rad))
        ) % 180.0
    else:
        gt_orientation = ground_truth.get('row_orientation')
        if gt_orientation and gt_orientation in COMPASS_TO_DEGREES:
            gt_angle_deg = COMPASS_TO_DEGREES[gt_orientation]
        else:
            gt_angle_deg = None

    if gt_angle_deg is not None:
        # Normalize both angles to 0-180 range (row orientation has 180° symmetry)
        det_norm = detected_angle % 180.0
        gt_norm = gt_angle_deg % 180.0

        # Minimum angular distance with 180° wrap-around
        diff = abs(det_norm - gt_norm)
        result['angle_error_deg'] = min(diff, 180.0 - diff)

    # Row count comparison
    gt_count = ground_truth.get('row_count')
    if gt_count is not None and gt_count > 0:
        result['row_count_error'] = detected_count - gt_count

    return result


# ---------------------------------------------------------------------------
# Block processing
# ---------------------------------------------------------------------------


def _get_centroid(geojson_coords: list) -> tuple[float, float]:
    """Compute centroid (lng, lat) from a GeoJSON coordinate ring."""
    coords = np.array(geojson_coords)
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def process_block(
    block: dict,
    source_name: str,
    zoom: int,
    approach: str = 'both',
    output_dir: str = 'output',
    use_cache: bool = True,
    locate_rows: bool = False,
) -> list[DetectionResult]:
    """Process a single block through the full detection pipeline.

    Steps:
        1. Extract boundary coordinates and compute centroid.
        2. Fetch and stitch aerial imagery tiles.
        3. Preprocess (vegetation index, CLAHE, edge detection).
        4. Run detector(s) (Hough, FFT, or both).
        5. Convert pixel measurements to meters.
        6. Compare with ground truth if available.
        7. Save debug images.
        8. Return results.

    Args:
        block: Block dict with 'name', 'boundary', 'vineyard_name', and
               optional ground truth fields.
        source_name: Tile source key ('linz', 'arcgis', 'kelowna', 'auto').
        zoom: Tile zoom level.
        approach: Detection approach ('hough', 'fft', 'both').
        output_dir: Directory for debug output.
        use_cache: Whether to use disk cache for tiles.

    Returns:
        List of DetectionResult (one per approach used).
    """
    block_name = block.get('name', 'Unknown')
    vineyard_name = block.get('vineyard_name', 'Unknown')
    boundary = block.get('boundary')

    if boundary is None:
        logger.error("Block '%s' has no boundary — skipping", block_name)
        return []

    # Extract exterior ring coordinates
    if isinstance(boundary, dict):
        coords = boundary.get('coordinates', [[]])[0]
    elif isinstance(boundary, list):
        coords = boundary[0] if len(boundary) > 0 and isinstance(boundary[0][0], list) else boundary
    else:
        logger.error("Block '%s' has invalid boundary format — skipping", block_name)
        return []

    if len(coords) < 3:
        logger.error("Block '%s' boundary has fewer than 3 coordinates — skipping", block_name)
        return []

    # 1. Centroid for source selection and meter conversion
    centroid_lng, centroid_lat = _get_centroid(coords)
    logger.info("Processing block '%s' (vineyard: %s) — centroid: %.4f, %.4f",
                block_name, vineyard_name, centroid_lng, centroid_lat)

    # Auto-select source if requested
    actual_source = source_name
    if source_name == 'auto':
        actual_source = auto_select_source(centroid_lng)
        logger.info("Auto-selected source: %s", actual_source)

    if actual_source not in TILE_SOURCES:
        logger.error("Unknown tile source: %s", actual_source)
        return []

    source_config = TILE_SOURCES[actual_source]
    actual_zoom = min(zoom, source_config.max_zoom)
    if actual_zoom != zoom:
        logger.warning("Zoom %d exceeds max %d for %s — clamped to %d",
                        zoom, source_config.max_zoom, actual_source, actual_zoom)

    cache_dir = os.path.join(output_dir, '.tile_cache') if use_cache else None

    # 2. Fetch and stitch tiles
    logger.info("Fetching tiles from %s at zoom %d...", actual_source, actual_zoom)
    try:
        masked_image, mask, tile_origin = fetch_and_stitch(
            source_config, coords, actual_zoom, actual_source, cache_dir
        )
    except Exception as exc:
        logger.error("Failed to fetch tiles for block '%s': %s", block_name, exc)
        return []

    if masked_image.size == 0 or mask.size == 0:
        logger.error("Empty image or mask for block '%s' — skipping", block_name)
        return []

    tile_size = source_config.tile_size
    mpp = meters_per_pixel(centroid_lat, actual_zoom, tile_size)

    # 3. Preprocess
    logger.info("Preprocessing...")
    try:
        prep = preprocess(masked_image, mask, mpp=mpp)
    except Exception as exc:
        logger.error("Preprocessing failed for block '%s': %s", block_name, exc)
        return []

    # Ground truth for comparison
    ground_truth = {
        'row_spacing_m': block.get('row_spacing_m'),
        'row_orientation': block.get('row_orientation'),
        'row_angle': block.get('row_angle'),
        'row_count': block.get('row_count'),
    }
    results: list[DetectionResult] = []

    # 4. Run detector(s) — FFT first so it can provide a prior for Hough
    hough_result = None
    fft_result = None

    if approach in ('fft', 'both'):
        logger.info("Running FFT detector...")
        t0 = time.monotonic()
        try:
            fft_result = fft_detect(prep, centroid_lat, actual_zoom, tile_size=tile_size)
            elapsed = time.monotonic() - t0

            if fft_result is not None:
                comparison = compare_with_ground_truth(
                    fft_result.angle_degrees, fft_result.spacing_meters,
                    fft_result.row_count, ground_truth
                )

                results.append(DetectionResult(
                    block_name=block_name,
                    vineyard_name=vineyard_name,
                    approach='fft',
                    detected_angle=fft_result.angle_degrees,
                    detected_spacing_m=fft_result.spacing_meters,
                    detected_row_count=fft_result.row_count,
                    confidence=fft_result.confidence,
                    ground_truth_spacing=ground_truth.get('row_spacing_m'),
                    ground_truth_orientation=ground_truth.get('row_orientation'),
                    ground_truth_row_count=ground_truth.get('row_count'),
                    spacing_error_pct=comparison['spacing_error_pct'],
                    angle_error_deg=comparison['angle_error_deg'],
                    processing_time_s=elapsed,
                ))
                logger.info(
                    "FFT result: angle=%.1f°, spacing=%.2fm, rows=%d, confidence=%.2f (%.2fs)",
                    fft_result.angle_degrees, fft_result.spacing_meters,
                    fft_result.row_count, fft_result.confidence, elapsed,
                )
            else:
                logger.warning("FFT detection returned no result for block '%s'", block_name)
        except Exception as exc:
            logger.error("FFT detection failed for block '%s': %s", block_name, exc)

    if approach in ('hough', 'both'):
        logger.info("Running Hough detector...")
        t0 = time.monotonic()
        try:
            # Pass FFT angle as prior when available, confident, and plausible
            fft_prior = None
            if (fft_result is not None
                    and fft_result.confidence >= 0.5
                    and 0.5 <= fft_result.spacing_meters <= 10.0):
                fft_prior = (fft_result.angle_degrees, fft_result.confidence)
            hough_result = hough_detect(prep, centroid_lat, actual_zoom, fft_prior=fft_prior, tile_size=tile_size)
            elapsed = time.monotonic() - t0

            if hough_result is not None:
                comparison = compare_with_ground_truth(
                    hough_result.angle_degrees, hough_result.spacing_meters,
                    hough_result.row_count, ground_truth
                )

                results.append(DetectionResult(
                    block_name=block_name,
                    vineyard_name=vineyard_name,
                    approach='hough',
                    detected_angle=hough_result.angle_degrees,
                    detected_spacing_m=hough_result.spacing_meters,
                    detected_row_count=hough_result.row_count,
                    confidence=hough_result.confidence,
                    ground_truth_spacing=ground_truth.get('row_spacing_m'),
                    ground_truth_orientation=ground_truth.get('row_orientation'),
                    ground_truth_row_count=ground_truth.get('row_count'),
                    spacing_error_pct=comparison['spacing_error_pct'],
                    angle_error_deg=comparison['angle_error_deg'],
                    processing_time_s=elapsed,
                ))
                logger.info(
                    "Hough result: angle=%.1f°, spacing=%.2fm, rows=%d, confidence=%.2f (%.2fs)",
                    hough_result.angle_degrees, hough_result.spacing_meters,
                    hough_result.row_count, hough_result.confidence, elapsed,
                )
            else:
                logger.warning("Hough detection returned no result for block '%s'", block_name)
        except Exception as exc:
            logger.error("Hough detection failed for block '%s': %s", block_name, exc)

    fft2d_result = None
    if approach in ('fft2d', 'all'):
        logger.info("Running 2D FFT detector...")
        t0 = time.monotonic()
        try:
            fft2d_result = fft2d_detect(prep, centroid_lat, actual_zoom, tile_size=tile_size)
            elapsed = time.monotonic() - t0

            if fft2d_result is not None:
                comparison = compare_with_ground_truth(
                    fft2d_result.angle_degrees, fft2d_result.spacing_meters,
                    fft2d_result.row_count, ground_truth
                )

                results.append(DetectionResult(
                    block_name=block_name,
                    vineyard_name=vineyard_name,
                    approach='fft2d',
                    detected_angle=fft2d_result.angle_degrees,
                    detected_spacing_m=fft2d_result.spacing_meters,
                    detected_row_count=fft2d_result.row_count,
                    confidence=fft2d_result.confidence,
                    ground_truth_spacing=ground_truth.get('row_spacing_m'),
                    ground_truth_orientation=ground_truth.get('row_orientation'),
                    ground_truth_row_count=ground_truth.get('row_count'),
                    spacing_error_pct=comparison['spacing_error_pct'],
                    angle_error_deg=comparison['angle_error_deg'],
                    processing_time_s=elapsed,
                ))
                logger.info(
                    "2D FFT result: angle=%.1f°, spacing=%.2fm, rows=%d, confidence=%.2f (%.2fs)",
                    fft2d_result.angle_degrees, fft2d_result.spacing_meters,
                    fft2d_result.row_count, fft2d_result.confidence, elapsed,
                )
            else:
                logger.warning("2D FFT detection returned no result for block '%s'", block_name)
        except Exception as exc:
            logger.error("2D FFT detection failed for block '%s': %s", block_name, exc)

    # 6. Row locator (individual row positions)
    row_locator_result = None
    if locate_rows and fft2d_result is not None:
        from row_locator import locate_rows as run_row_locator
        logger.info("Running row locator...")
        try:
            row_locator_result = run_row_locator(prep, fft2d_result, centroid_lat, actual_zoom, tile_size=tile_size)
            if row_locator_result is not None:
                _print_row_locator_summary(row_locator_result, block_name, ground_truth)
        except Exception as exc:
            logger.error("Row locator failed for block '%s': %s", block_name, exc)

    # 7. Save debug images
    try:
        save_pipeline_debug(
            original_bgr=masked_image,
            vegetation=prep.vegetation,
            edges=prep.edges,
            mask=prep.mask,
            hough_result=hough_result,
            fft_result=fft_result,
            fft2d_result=fft2d_result,
            block_name=block_name,
            output_dir=output_dir,
            row_locator_result=row_locator_result,
        )
    except Exception as exc:
        logger.warning("Failed to save debug images for '%s': %s", block_name, exc)

    return results


def _print_row_locator_summary(result, block_name: str, ground_truth: dict) -> None:
    """Print per-row summary table for the row locator result."""
    gt_count = ground_truth.get('row_count')
    print(f"\n--- Row Locator: {block_name} ---")
    for row in result.rows:
        sp_str = f"{row.spacing_to_previous:.2f}m" if row.spacing_to_previous is not None else "   --"
        print(
            f"  Row {row.row_index:>3d}: perp={row.mean_perpendicular:>8.1f}px, "
            f"spacing={sp_str}, conf={row.confidence:.2f}, len={row.length_px:.0f}px"
        )
    count_str = f"{result.total_row_count}"
    if gt_count:
        count_str += f" (GT: {gt_count})"
    print(
        f"  Summary: {count_str} rows, "
        f"mean={result.mean_spacing_m:.2f}m "
        f"(std={result.spacing_std_m:.3f}m), "
        f"range=[{result.spacing_range_m[0]:.2f}, {result.spacing_range_m[1]:.2f}]m"
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_results_table(results: list[DetectionResult]) -> None:
    """Pretty-print a results table to the console.

    Args:
        results: List of DetectionResult instances.
    """
    if not results:
        print("\nNo results to display.")
        return

    # Header
    header = (
        f"{'Block':<15} {'Vineyard':<20} {'Method':<7} "
        f"{'Angle':>7} {'Spacing':>9} {'Rows':>5} {'Conf':>6} "
        f"{'GT Angle':>10} {'GT Space':>9} {'GT Rows':>8} "
        f"{'Ang Err':>8} {'Spc Err':>8} {'Time':>6}"
    )
    sep = '-' * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in results:
        gt_orient = r.ground_truth_orientation or '—'
        gt_spacing = f"{r.ground_truth_spacing:.2f}" if r.ground_truth_spacing else '—'
        gt_rows = str(r.ground_truth_row_count) if r.ground_truth_row_count else '—'
        angle_err = f"{r.angle_error_deg:.1f}°" if r.angle_error_deg is not None else '—'
        spacing_err = f"{r.spacing_error_pct:.1f}%" if r.spacing_error_pct is not None else '—'

        print(
            f"{r.block_name:<15} {r.vineyard_name:<20} {r.approach:<7} "
            f"{r.detected_angle:>6.1f}° {r.detected_spacing_m:>8.2f}m {r.detected_row_count:>5} "
            f"{r.confidence:>5.2f} "
            f"{gt_orient:>10} {gt_spacing:>9} {gt_rows:>8} "
            f"{angle_err:>8} {spacing_err:>8} {r.processing_time_s:>5.1f}s"
        )

    print(sep)

    # Summary statistics
    spacing_errors = [r.spacing_error_pct for r in results if r.spacing_error_pct is not None]
    angle_errors = [r.angle_error_deg for r in results if r.angle_error_deg is not None]

    if spacing_errors:
        print(f"\nSpacing error — mean: {sum(spacing_errors)/len(spacing_errors):.1f}%, "
              f"max: {max(spacing_errors):.1f}%")
    if angle_errors:
        print(f"Angle error   — mean: {sum(angle_errors)/len(angle_errors):.1f}°, "
              f"max: {max(angle_errors):.1f}°")

    # Success rate
    if spacing_errors and angle_errors:
        successes = sum(
            1 for r in results
            if r.spacing_error_pct is not None and r.angle_error_deg is not None
            and r.spacing_error_pct <= 15.0 and r.angle_error_deg <= 5.0
        )
        total = sum(1 for r in results if r.spacing_error_pct is not None)
        if total > 0:
            print(f"Success rate  — {successes}/{total} "
                  f"({100*successes/total:.0f}%) within +-5 deg, +-15% spacing")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser."""
    parser = argparse.ArgumentParser(
        description='CV Row Detection from Aerial Imagery — Feasibility Prototype',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_rows.py --block "B3" --source linz --zoom 20
  python detect_rows.py --all --approach both
  python detect_rows.py --fetch-blocks --org-id <uuid>
  python detect_rows.py --geojson boundary.geojson --source arcgis --zoom 19
        """,
    )

    # Input selection (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--block', type=str, metavar='NAME',
        help='Process a single block from test_blocks.json by name',
    )
    input_group.add_argument(
        '--all', action='store_true',
        help='Process all blocks from test_blocks.json',
    )
    input_group.add_argument(
        '--geojson', type=str, metavar='PATH',
        help='Process a custom GeoJSON polygon file',
    )
    input_group.add_argument(
        '--fetch-blocks', action='store_true',
        help='Populate test_blocks.json from Supabase (requires .env credentials)',
    )

    # Filters
    parser.add_argument(
        '--org-id', type=str, metavar='UUID',
        help='Filter by organisation ID (used with --fetch-blocks)',
    )

    # Tile source
    parser.add_argument(
        '--source', type=str, default='auto',
        choices=['linz', 'arcgis', 'kelowna', 'auto'],
        help='Tile imagery source (default: auto-select based on longitude)',
    )
    parser.add_argument(
        '--zoom', type=int, default=None,
        help='Override tile zoom level (default: source-dependent)',
    )

    # Detection approach
    parser.add_argument(
        '--approach', type=str, default='fft2d',
        choices=['hough', 'fft', 'fft2d', 'both', 'all'],
        help='Detection approach to use (default: fft2d)',
    )

    # Output
    parser.add_argument(
        '--output-dir', type=str, default='output',
        help='Directory for debug images and results (default: output/)',
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Skip tile disk cache (re-download all tiles)',
    )

    # Row locator
    parser.add_argument(
        '--locate-rows', action='store_true',
        help='Run row locator after fft2d detection to find individual row positions',
    )

    # Logging
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable debug logging',
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Auto-enable fft2d when row location is requested
    if args.locate_rows and args.approach not in ('fft2d', 'all'):
        logger.info("--locate-rows requires fft2d; switching approach to 'all'")
        args.approach = 'all'

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s',
        datefmt='%H:%M:%S',
    )

    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Mode: fetch blocks from Supabase ---
    if args.fetch_blocks:
        blocks = fetch_blocks_from_supabase(args.org_id)
        if not blocks:
            logger.error("No blocks fetched. See messages above for details.")
            sys.exit(1)

        out_path = Path('test_blocks.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'blocks': blocks}, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d blocks to %s", len(blocks), out_path)
        # Print summary
        with_boundary = sum(1 for b in blocks if b.get('boundary') is not None)
        with_spacing = sum(1 for b in blocks if b.get('row_spacing_m') is not None)
        print(f"\nFetched {len(blocks)} blocks: "
              f"{with_boundary} with boundary, {with_spacing} with row spacing ground truth")
        return

    # --- Mode: process GeoJSON file ---
    if args.geojson:
        geojson_path = Path(args.geojson)
        if not geojson_path.exists():
            logger.error("GeoJSON file not found: %s", geojson_path)
            sys.exit(1)

        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

        # Support both bare Polygon and FeatureCollection
        if geojson_data.get('type') == 'FeatureCollection':
            features = geojson_data.get('features', [])
        elif geojson_data.get('type') == 'Feature':
            features = [geojson_data]
        elif geojson_data.get('type') == 'Polygon':
            features = [{'type': 'Feature', 'geometry': geojson_data, 'properties': {}}]
        else:
            logger.error("Unsupported GeoJSON type: %s", geojson_data.get('type'))
            sys.exit(1)

        blocks = []
        for i, feat in enumerate(features):
            geom = feat.get('geometry', {})
            props = feat.get('properties', {})
            blocks.append({
                'name': props.get('name', f'GeoJSON-{i+1}'),
                'vineyard_name': props.get('vineyard', 'Custom'),
                'boundary': geom,
                'row_spacing_m': props.get('row_spacing_m'),
                'row_orientation': props.get('row_orientation'),
                'row_angle': props.get('row_angle'),
                'row_count': props.get('row_count'),
            })

        if not blocks:
            logger.error("No features found in GeoJSON file")
            sys.exit(1)

        logger.info("Loaded %d features from %s", len(blocks), geojson_path)
        all_results = _process_blocks(blocks, args)
        _finalize(all_results, args.output_dir)
        return

    # --- Mode: process from test_blocks.json ---
    blocks = load_test_blocks()
    if not blocks:
        print("\nNo test blocks available. Options:")
        print("  1. Run: python detect_rows.py --fetch-blocks")
        print("  2. Manually add blocks to test_blocks.json")
        print("  3. Use: python detect_rows.py --geojson <path>")
        sys.exit(1)

    if args.block:
        # Find the specific block
        matching = [b for b in blocks if b['name'].lower() == args.block.lower()]
        if not matching:
            available = ', '.join(b['name'] for b in blocks)
            logger.error("Block '%s' not found. Available: %s", args.block, available)
            sys.exit(1)
        blocks = matching
    elif not args.all:
        # Neither --block nor --all specified — show help
        parser.print_help()
        print(f"\nAvailable blocks: {', '.join(b['name'] for b in blocks)}")
        return

    all_results = _process_blocks(blocks, args)
    _finalize(all_results, args.output_dir)


def _process_blocks(blocks: list[dict], args: argparse.Namespace) -> list[DetectionResult]:
    """Process a list of blocks with the given CLI arguments.

    Args:
        blocks: List of block dicts.
        args: Parsed CLI arguments.

    Returns:
        List of all DetectionResult instances.
    """
    all_results: list[DetectionResult] = []

    for i, block in enumerate(blocks, 1):
        block_name = block.get('name', 'Unknown')
        print(f"\n{'='*60}")
        print(f"Processing block {i}/{len(blocks)}: {block_name}")
        print(f"{'='*60}")

        # Determine zoom
        source_name = args.source
        if source_name == 'auto':
            coords = _extract_coords(block)
            if coords:
                centroid_lng = np.mean([c[0] for c in coords])
                source_name = auto_select_source(centroid_lng)
            else:
                source_name = 'arcgis'

        zoom = args.zoom if args.zoom is not None else default_zoom(source_name)

        results = process_block(
            block=block,
            source_name=args.source,  # Pass original — process_block handles 'auto'
            zoom=zoom,
            approach=args.approach,
            output_dir=args.output_dir,
            use_cache=not args.no_cache,
            locate_rows=args.locate_rows,
        )
        all_results.extend(results)

    return all_results


def _extract_coords(block: dict) -> list | None:
    """Extract coordinate ring from a block's boundary field."""
    boundary = block.get('boundary')
    if boundary is None:
        return None
    if isinstance(boundary, dict):
        return boundary.get('coordinates', [[]])[0]
    if isinstance(boundary, list):
        return boundary[0] if len(boundary) > 0 and isinstance(boundary[0][0], list) else boundary
    return None


def _finalize(results: list[DetectionResult], output_dir: str) -> None:
    """Print results table and save comparison summary.

    Args:
        results: All detection results.
        output_dir: Output directory for the summary file.
    """
    print_results_table(results)

    if results:
        summary_path = os.path.join(output_dir, 'results_summary.md')
        create_comparison_summary(
            [asdict(r) for r in results],
            summary_path,
        )
        print(f"\nResults summary saved to: {summary_path}")
        print(f"Debug images saved to: {output_dir}/")


if __name__ == '__main__':
    main()
