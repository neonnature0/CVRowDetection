"""
Fetch and stitch XYZ map tiles from multiple imagery sources.

Supports three tile sources:
  - LINZ (New Zealand aerial imagery, WebP, up to zoom 22)
  - ArcGIS World Imagery (global, JPEG, up to zoom 19)
  - Kelowna (City of Kelowna ortho, bbox-based ArcGIS export, up to zoom 19)
"""

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv

from vinerow.acquisition.geo_utils import polygon_bbox, tiles_covering_bbox, polygon_to_pixel_mask

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tile source configuration
# ---------------------------------------------------------------------------

EARTH_HALF_CIRCUMFERENCE = 20037508.3427892


@dataclass
class TileSourceConfig:
    name: str
    url_template: str
    tile_size: int
    max_zoom: int
    headers: dict = field(default_factory=dict)
    is_bbox_source: bool = False


TILE_SOURCES: dict[str, TileSourceConfig] = {
    "linz": TileSourceConfig(
        name="LINZ Aerial",
        url_template=(
            "https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad"
            "/{z}/{x}/{y}.webp?api={api_key}"
        ),
        tile_size=256,
        max_zoom=22,
    ),
    "arcgis": TileSourceConfig(
        name="ArcGIS World Imagery",
        url_template=(
            "https://server.arcgisonline.com/ArcGIS/rest/services"
            "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        tile_size=256,
        max_zoom=19,
    ),
    "kelowna": TileSourceConfig(
        name="Kelowna Orthophoto",
        url_template=(
            "https://geo.kelowna.ca/arcgis/rest/services/Ortho_2021/MapServer/export"
        ),
        tile_size=512,
        max_zoom=19,
        is_bbox_source=True,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_api_key(source_name: str) -> str | None:
    """Get API key from environment variables."""
    if source_name == "linz":
        return os.environ.get("LINZ_API_KEY")
    return None


def _tile_to_bbox_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert tile coordinates to Web Mercator (EPSG:3857) bounding box."""
    tile_count = 2 ** z
    tile_size = 2.0 * EARTH_HALF_CIRCUMFERENCE / tile_count
    min_x = -EARTH_HALF_CIRCUMFERENCE + x * tile_size
    max_x = min_x + tile_size
    max_y = EARTH_HALF_CIRCUMFERENCE - y * tile_size
    min_y = max_y - tile_size
    return min_x, min_y, max_x, max_y


def _build_url(source: TileSourceConfig, source_name: str, z: int, x: int, y: int) -> str:
    """Build the request URL for a tile."""
    if source.is_bbox_source:
        min_x, min_y, max_x, max_y = _tile_to_bbox_3857(z, x, y)
        bbox_str = f"{min_x},{min_y},{max_x},{max_y}"
        url = (
            f"{source.url_template}"
            f"?bbox={bbox_str}"
            f"&bboxSR=3857"
            f"&imageSR=3857"
            f"&size={source.tile_size},{source.tile_size}"
            f"&format=jpg"
            f"&f=image"
            f"&layers=show:1"
        )
        return url

    url = source.url_template
    url = url.replace("{z}", str(z))
    url = url.replace("{x}", str(x))
    url = url.replace("{y}", str(y))

    api_key = get_api_key(source_name)
    if api_key and "{api_key}" in url:
        url = url.replace("{api_key}", api_key)
    elif "{api_key}" in url:
        logger.warning(
            "URL template requires {api_key} but none found in environment for %s",
            source_name,
        )

    return url


def _cache_path(cache_dir: str, source_name: str, z: int, x: int, y: int) -> Path:
    """Deterministic cache file path for a tile."""
    return Path(cache_dir) / source_name / str(z) / str(x) / f"{y}.png"


def _image_bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode image bytes (WebP, JPEG, PNG) to a BGR numpy array."""
    try:
        img = Image.open(BytesIO(data))
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        return arr[:, :, ::-1].copy()
    except Exception as exc:
        raise ValueError(f"Failed to decode image ({len(data)} bytes)") from exc


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------


def fetch_tile(
    source: TileSourceConfig,
    z: int,
    x: int,
    y: int,
    source_name: str = "unknown",
    cache_dir: str | None = None,
) -> np.ndarray:
    """Fetch a single tile, decode to numpy BGR array.

    Implements disk caching and retry logic (3 attempts with exponential backoff).
    """
    if cache_dir is not None:
        cp = _cache_path(cache_dir, source_name, z, x, y)
        if cp.exists():
            logger.debug("Cache hit: %s", cp)
            data = cp.read_bytes()
            return _image_bytes_to_bgr(data)

    url = _build_url(source, source_name, z, x, y)
    last_exc: Exception | None = None

    for attempt in range(1, 4):
        try:
            logger.debug("Fetching tile z=%d x=%d y=%d (attempt %d): %s", z, x, y, attempt, url)
            resp = requests.get(url, headers=source.headers, timeout=30)
            resp.raise_for_status()
            data = resp.content

            if len(data) < 100:
                raise ValueError(
                    f"Response too small ({len(data)} bytes), likely an error tile"
                )

            bgr = _image_bytes_to_bgr(data)

            if cache_dir is not None:
                cp = _cache_path(cache_dir, source_name, z, x, y)
                cp.parent.mkdir(parents=True, exist_ok=True)
                img_pil = Image.fromarray(bgr[:, :, ::-1])
                img_pil.save(str(cp), "PNG")
                logger.debug("Cached tile to %s", cp)

            time.sleep(0.1)
            return bgr

        except Exception as exc:
            last_exc = exc
            wait = 0.5 * (2 ** (attempt - 1))
            logger.warning(
                "Tile fetch failed z=%d x=%d y=%d attempt %d: %s. Retrying in %.1fs",
                z, x, y, attempt, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch tile z={z} x={x} y={y} from {source.name} after 3 attempts"
    ) from last_exc


# ---------------------------------------------------------------------------
# Stitch
# ---------------------------------------------------------------------------


def fetch_and_stitch(
    source: TileSourceConfig,
    geojson_coords: list,
    zoom: int,
    source_name: str = "unknown",
    cache_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Fetch all tiles covering a polygon's bounding box, stitch, and mask.

    Returns:
        Tuple of (masked_image, mask, tile_origin).
    """
    bbox = polygon_bbox(geojson_coords)
    logger.info(
        "Polygon bbox: min_lng=%.6f min_lat=%.6f max_lng=%.6f max_lat=%.6f",
        *bbox,
    )

    tiles = tiles_covering_bbox(bbox, zoom)
    if not tiles:
        raise ValueError("No tiles found covering the polygon bbox")

    tx_values = [t[0] for t in tiles]
    ty_values = [t[1] for t in tiles]
    min_tx = min(tx_values)
    max_tx = max(tx_values)
    min_ty = min(ty_values)
    max_ty = max(ty_values)

    cols = max_tx - min_tx + 1
    rows = max_ty - min_ty + 1
    tile_size = source.tile_size

    logger.info(
        "Fetching %d tiles (%d cols x %d rows) at zoom %d, tile_size=%d",
        len(tiles), cols, rows, zoom, tile_size,
    )

    tile_images: dict[tuple[int, int], np.ndarray] = {}
    for tx, ty in tiles:
        try:
            tile_img = fetch_tile(source, zoom, tx, ty, source_name, cache_dir)
            if tile_img.shape[0] != tile_size or tile_img.shape[1] != tile_size:
                tile_img = np.array(
                    Image.fromarray(tile_img[:, :, ::-1]).resize(
                        (tile_size, tile_size), Image.LANCZOS
                    )
                )[:, :, ::-1].copy()
            tile_images[(tx, ty)] = tile_img
        except RuntimeError:
            logger.error("Could not fetch tile z=%d x=%d y=%d — using black", zoom, tx, ty)
            tile_images[(tx, ty)] = np.zeros(
                (tile_size, tile_size, 3), dtype=np.uint8
            )

    img_h = rows * tile_size
    img_w = cols * tile_size
    stitched = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for (tx, ty), tile_img in tile_images.items():
        col_idx = tx - min_tx
        row_idx = ty - min_ty
        y0 = row_idx * tile_size
        x0 = col_idx * tile_size
        stitched[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile_img

    logger.info("Stitched image size: %d x %d px", img_w, img_h)

    tile_origin = (min_tx, min_ty)
    mask = polygon_to_pixel_mask(
        geojson_coords, tile_origin, zoom, tile_size, (img_h, img_w),
    )

    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    masked_image = np.where(mask_3ch == 255, stitched, 0).astype(np.uint8)

    masked_pixels = int(np.count_nonzero(mask))
    total_pixels = img_h * img_w
    logger.info(
        "Mask covers %d / %d pixels (%.1f%%)",
        masked_pixels, total_pixels, 100.0 * masked_pixels / total_pixels if total_pixels else 0,
    )

    return masked_image, mask, tile_origin


# ---------------------------------------------------------------------------
# Source selection helpers
# ---------------------------------------------------------------------------


def auto_select_source(centroid_lng: float) -> str:
    """Auto-select tile source based on centroid longitude."""
    if centroid_lng > 165:
        return "linz"
    if -130 < centroid_lng < -110:
        return "kelowna"
    return "arcgis"


def default_zoom(source_name: str) -> int:
    """Default zoom level for a given source."""
    if source_name == "linz":
        return 20
    return 19
