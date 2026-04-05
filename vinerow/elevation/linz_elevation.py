"""Fetch LINZ LiDAR elevation data (DSM/DEM) for a block polygon.

Uses the NZ Elevation Open Data on AWS (S3 bucket: nz-elevation).
Data is Cloud Optimized GeoTIFF — we read only the region needed
for a given block polygon via HTTP range requests (no full tile download).

Requires: rasterio (pip install rasterio)

Usage:
    from vinerow.elevation.linz_elevation import get_canopy_height
    csm = get_canopy_height(boundary_coords, region="marlborough_2020-2022")
    # csm is a dict with: height_map (2D array), stats, warnings
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

S3_BASE = "https://nz-elevation.s3-ap-southeast-2.amazonaws.com"

# Available Marlborough datasets (newest first)
MARLBOROUGH_DATASETS = {
    "marlborough_2020-2022": {
        "dsm": f"{S3_BASE}/marlborough/marlborough_2020-2022/dsm_1m/2193/collection.json",
        "dem": f"{S3_BASE}/marlborough/marlborough_2020-2022/dem_1m/2193/collection.json",
    },
    "marlborough_2018": {
        "dsm": f"{S3_BASE}/marlborough/marlborough_2018/dsm_1m/2193/collection.json",
        "dem": f"{S3_BASE}/marlborough/marlborough_2018/dem_1m/2193/collection.json",
    },
}


@lru_cache(maxsize=4)
def _load_collection(url: str) -> list[dict]:
    """Load STAC collection and return list of items with bbox + tiff URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    coll = resp.json()

    items = []
    base_url = url.rsplit("/", 1)[0]
    for link in coll.get("links", []):
        if link.get("rel") == "item":
            href = link["href"]
            if href.startswith("./"):
                href = href[2:]
            items.append({
                "json_url": f"{base_url}/{href}",
                "tiff_url": f"{base_url}/{href.replace('.json', '.tiff')}",
            })
    return items


def _find_covering_tiles(
    collection_url: str,
    min_lng: float,
    min_lat: float,
    max_lng: float,
    max_lat: float,
) -> list[str]:
    """Find STAC items whose bbox overlaps the query bbox. Returns tiff URLs."""
    items = _load_collection(collection_url)

    # We need to fetch each item's bbox to check overlap.
    # For efficiency, batch-check: items are small JSON files.
    covering = []
    for item in items:
        try:
            resp = requests.get(item["json_url"], timeout=10)
            resp.raise_for_status()
            meta = resp.json()
            bbox = meta.get("bbox", [])
            if len(bbox) >= 4:
                # Check overlap: item bbox [w, s, e, n]
                iw, is_, ie, in_ = bbox[0], bbox[1], bbox[2], bbox[3]
                if iw <= max_lng and ie >= min_lng and is_ <= max_lat and in_ >= min_lat:
                    covering.append(item["tiff_url"])
                    if len(covering) >= 4:
                        # Most blocks fit in 1-2 tiles
                        break
        except Exception as e:
            logger.debug("Failed to check item %s: %s", item["json_url"], e)
            continue

    return covering


def get_elevation_for_block(
    boundary_coords: list[list[float]],
    dataset: str = "marlborough_2020-2022",
    layer: str = "dsm",
) -> dict | None:
    """Fetch elevation data for a block polygon.

    Args:
        boundary_coords: GeoJSON polygon coordinates [[lng, lat], ...]
        dataset: Which LINZ dataset to use
        layer: "dsm" (surface, includes canopy) or "dem" (bare ground)

    Returns:
        Dict with keys: tiff_urls, bbox, or None if no coverage.
    """
    if dataset not in MARLBOROUGH_DATASETS:
        logger.error("Unknown dataset: %s", dataset)
        return None

    collection_url = MARLBOROUGH_DATASETS[dataset][layer]

    # Compute bbox from polygon
    lngs = [c[0] for c in boundary_coords]
    lats = [c[1] for c in boundary_coords]
    min_lng, max_lng = min(lngs), max(lngs)
    min_lat, max_lat = min(lats), max(lats)

    logger.info(
        "Searching %s %s for bbox [%.4f, %.4f, %.4f, %.4f]",
        dataset, layer, min_lng, min_lat, max_lng, max_lat,
    )

    tiff_urls = _find_covering_tiles(collection_url, min_lng, min_lat, max_lng, max_lat)

    if not tiff_urls:
        logger.warning("No %s tiles found covering this block", layer)
        return None

    logger.info("Found %d %s tile(s) covering block", len(tiff_urls), layer)
    return {
        "tiff_urls": tiff_urls,
        "bbox": [min_lng, min_lat, max_lng, max_lat],
        "dataset": dataset,
        "layer": layer,
    }


def get_canopy_height(
    boundary_coords: list[list[float]],
    dataset: str = "marlborough_2020-2022",
) -> dict | None:
    """Compute canopy height model (CSM = DSM - DEM) for a block.

    Requires rasterio for reading Cloud Optimized GeoTIFFs via HTTP.

    Returns:
        Dict with: mean_height, max_height, pct_above_05m, pct_above_3m,
        is_likely_vineyard, warnings
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        logger.error("rasterio not installed. pip install rasterio")
        return {"error": "rasterio not installed", "warnings": ["Install rasterio for elevation data"]}

    dsm_info = get_elevation_for_block(boundary_coords, dataset, "dsm")
    dem_info = get_elevation_for_block(boundary_coords, dataset, "dem")

    if not dsm_info or not dem_info:
        return {"error": "No LiDAR coverage for this block", "warnings": ["Block outside LiDAR coverage"]}

    bbox = dsm_info["bbox"]
    warnings = []

    try:
        # Read DSM (includes canopy)
        with rasterio.open(dsm_info["tiff_urls"][0]) as dsm_src:
            window = from_bounds(*bbox, transform=dsm_src.transform)
            dsm_data = dsm_src.read(1, window=window)

        # Read DEM (bare ground)
        with rasterio.open(dem_info["tiff_urls"][0]) as dem_src:
            window = from_bounds(*bbox, transform=dem_src.transform)
            dem_data = dem_src.read(1, window=window)

        # Compute canopy height
        if dsm_data.shape != dem_data.shape:
            # Resize to match
            min_h = min(dsm_data.shape[0], dem_data.shape[0])
            min_w = min(dsm_data.shape[1], dem_data.shape[1])
            dsm_data = dsm_data[:min_h, :min_w]
            dem_data = dem_data[:min_h, :min_w]

        csm = dsm_data - dem_data
        # Mask out nodata and negative values
        valid = (csm > -1) & (csm < 50) & ~np.isnan(csm)
        csm_valid = csm[valid]

        if len(csm_valid) == 0:
            return {"error": "No valid elevation data in block", "warnings": ["All pixels are nodata"]}

        mean_height = float(np.mean(csm_valid))
        max_height = float(np.max(csm_valid))
        pct_above_05m = float(np.sum(csm_valid > 0.5) / len(csm_valid) * 100)
        pct_above_3m = float(np.sum(csm_valid > 3.0) / len(csm_valid) * 100)

        # Vineyard heuristic: most pixels 0.5-2.5m, few above 3m
        is_likely_vineyard = pct_above_05m > 30 and pct_above_3m < 15

        if pct_above_3m > 20:
            warnings.append(f"{pct_above_3m:.0f}% of area has canopy > 3m (trees/buildings?)")
        if pct_above_05m < 20:
            warnings.append(f"Only {pct_above_05m:.0f}% has canopy > 0.5m (bare ground?)")

        return {
            "mean_height_m": round(mean_height, 2),
            "max_height_m": round(max_height, 2),
            "pct_above_05m": round(pct_above_05m, 1),
            "pct_above_3m": round(pct_above_3m, 1),
            "is_likely_vineyard": is_likely_vineyard,
            "grid_size": list(csm.shape),
            "warnings": warnings,
        }

    except Exception as e:
        logger.error("Failed to read elevation data: %s", e)
        return {"error": str(e), "warnings": [f"Elevation read failed: {e}"]}
