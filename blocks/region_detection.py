"""Detect which wine region a block belongs to based on its boundary centroid.

Uses nearest-neighbor lookup against a reference file of region centroids.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

_REGIONS_CACHE: dict[Path, list[dict]] = {}
_DEFAULT_REGIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "nz_wine_regions.json"


def load_regions(path: str | Path | None = None) -> list[dict]:
    """Read the regions JSON file, validate schema, cache the result.

    Each entry must have: name (str), lat (float), lng (float).
    """
    resolved = Path(path) if path else _DEFAULT_REGIONS_PATH
    resolved = resolved.resolve()

    if resolved in _REGIONS_CACHE:
        return _REGIONS_CACHE[resolved]

    if not resolved.exists():
        logger.warning("Regions file not found: %s", resolved)
        return []

    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Regions file must be a JSON array")
        return []

    valid = []
    for entry in data:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and isinstance(entry.get("lat"), (int, float))
            and isinstance(entry.get("lng"), (int, float))
        ):
            valid.append(entry)
        else:
            logger.warning("Skipping invalid region entry: %s", entry)

    _REGIONS_CACHE[resolved] = valid
    return valid


def compute_block_centroid(boundary: dict) -> tuple[float, float]:
    """Compute the centroid of a block boundary as (lat, lng).

    Takes the block's boundary dict in GeoJSON Polygon format:
    {"type": "Polygon", "coordinates": [[[lng, lat], [lng, lat], ...]]}

    Uses unweighted average of vertices (simple centroid).
    """
    coords = boundary["coordinates"][0]  # outer ring

    # GeoJSON is [lng, lat] — we need to average and return (lat, lng)
    n = len(coords)
    # If polygon is closed (first == last), exclude the duplicate
    if n > 1 and coords[0][0] == coords[-1][0] and coords[0][1] == coords[-1][1]:
        coords = coords[:-1]
        n = len(coords)

    if n == 0:
        raise ValueError("Empty boundary coordinates")

    sum_lng = sum(c[0] for c in coords)
    sum_lat = sum(c[1] for c in coords)

    return (sum_lat / n, sum_lng / n)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance between two points in kilometers."""
    R = 6371.0  # Earth radius in km

    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def detect_region(
    boundary: dict,
    regions_path: str | Path | None = None,
    max_distance_km: float = 150.0,
) -> dict:
    """Detect which region a block belongs to.

    Args:
        boundary: GeoJSON Polygon dict with coordinates
        regions_path: path to regions JSON (defaults to data/nz_wine_regions.json)
        max_distance_km: maximum distance to consider a match

    Returns:
        {"region": str, "distance_km": float, "confidence": "high" | "low"}
        - confidence is "high" if distance < 50 km, "low" otherwise
        - region is "Other" if nearest is beyond max_distance_km
    """
    regions = load_regions(regions_path)
    if not regions:
        logger.error("No region reference data available; cannot assign region")
        return {"region": None, "distance_km": 0.0, "confidence": "low"}

    lat, lng = compute_block_centroid(boundary)

    best_name = "Other"
    best_dist = float("inf")

    for r in regions:
        dist = haversine_km(lat, lng, r["lat"], r["lng"])
        if dist < best_dist:
            best_dist = dist
            best_name = r["name"]

    if best_dist > max_distance_km:
        return {"region": "Other", "distance_km": round(best_dist, 2), "confidence": "low"}

    confidence = "high" if best_dist < 50.0 else "low"
    return {"region": best_name, "distance_km": round(best_dist, 2), "confidence": confidence}
