"""
Coordinate math utilities for slippy map tiles and pixel/meter conversions.

Implements standard XYZ tile math (OSM/Slippy Map convention) plus helpers
for converting between GeoJSON polygon coordinates and pixel-space masks.
"""

import math
import numpy as np
import cv2


def lng_lat_to_tile(lng: float, lat: float, zoom: int) -> tuple[int, int]:
    """Convert WGS84 coords to XYZ tile coordinates.

    Args:
        lng: Longitude in degrees (-180 to 180).
        lat: Latitude in degrees (-85.051 to 85.051).
        zoom: Zoom level (0-22).

    Returns:
        (tx, ty) tile coordinates.
    """
    n = 2 ** zoom
    tx = int(math.floor((lng + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat)
    ty = int(
        math.floor(
            (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
            / 2.0
            * n
        )
    )
    # Clamp to valid range
    tx = max(0, min(n - 1, tx))
    ty = max(0, min(n - 1, ty))
    return tx, ty


def tile_to_lng_lat(tx: int, ty: int, zoom: int) -> tuple[float, float]:
    """Top-left corner of tile in WGS84 (lng, lat).

    Args:
        tx: Tile X coordinate.
        ty: Tile Y coordinate.
        zoom: Zoom level.

    Returns:
        (longitude, latitude) of the tile's top-left corner.
    """
    n = 2 ** zoom
    lng = tx / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n)))
    lat = math.degrees(lat_rad)
    return lng, lat


def tile_bounds(tx: int, ty: int, zoom: int) -> tuple[float, float, float, float]:
    """Bounding box for a tile in WGS84.

    Args:
        tx: Tile X coordinate.
        ty: Tile Y coordinate.
        zoom: Zoom level.

    Returns:
        (min_lng, min_lat, max_lng, max_lat).
    """
    min_lng, max_lat = tile_to_lng_lat(tx, ty, zoom)
    max_lng, min_lat = tile_to_lng_lat(tx + 1, ty + 1, zoom)
    return min_lng, min_lat, max_lng, max_lat


def meters_per_pixel(lat: float, zoom: int, tile_size: int = 256) -> float:
    """Ground resolution in meters per pixel at given latitude and zoom.

    Uses the standard Web Mercator formula adjusted for tile size:
        resolution = (C / tile_size) * cos(lat) / 2^zoom

    The constant 156543.03392 = Earth circumference / 256. When a tile
    source returns 512 px tiles (e.g. Kelowna ortho), the ground resolution
    per pixel is halved.

    Args:
        lat: Latitude in degrees.
        zoom: Zoom level.
        tile_size: Pixel size of each tile (256 standard, 512 for hi-res).

    Returns:
        Meters per pixel.
    """
    lat_rad = math.radians(lat)
    base = 156543.03392 * math.cos(lat_rad) / (2 ** zoom)
    return base * 256 / tile_size


def polygon_bbox(
    geojson_coords: list,
) -> tuple[float, float, float, float]:
    """Compute bounding box from a GeoJSON polygon coordinate ring.

    Args:
        geojson_coords: List of [lng, lat] pairs (exterior ring of a polygon).

    Returns:
        (min_lng, min_lat, max_lng, max_lat).
    """
    coords = np.array(geojson_coords)
    min_lng = float(coords[:, 0].min())
    max_lng = float(coords[:, 0].max())
    min_lat = float(coords[:, 1].min())
    max_lat = float(coords[:, 1].max())
    return min_lng, min_lat, max_lng, max_lat


def tiles_covering_bbox(
    bbox: tuple[float, float, float, float], zoom: int
) -> list[tuple[int, int]]:
    """All (tx, ty) tiles that cover the bounding box.

    Args:
        bbox: (min_lng, min_lat, max_lng, max_lat).
        zoom: Zoom level.

    Returns:
        List of (tx, ty) tile coordinates covering the bbox.
    """
    min_lng, min_lat, max_lng, max_lat = bbox

    # Top-left tile (max_lat is north = smaller ty)
    tx_min, ty_min = lng_lat_to_tile(min_lng, max_lat, zoom)
    # Bottom-right tile (min_lat is south = larger ty)
    tx_max, ty_max = lng_lat_to_tile(max_lng, min_lat, zoom)

    tiles = []
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tiles.append((tx, ty))
    return tiles


def _lng_lat_to_tile_float(lng: float, lat: float, zoom: int) -> tuple[float, float]:
    """Convert WGS84 coords to fractional tile coordinates (not floored).

    Used internally for precise pixel positioning within stitched tile images.
    """
    n = 2 ** zoom
    tx_f = (lng + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    ty_f = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return tx_f, ty_f


def polygon_to_pixel_mask(
    geojson_coords: list,
    tile_origin: tuple[int, int],
    zoom: int,
    tile_size: int,
    img_shape: tuple[int, int],
) -> np.ndarray:
    """Convert GeoJSON polygon ring to a binary mask in pixel space.

    For each coordinate (lng, lat):
      1. Convert to fractional tile coordinates.
      2. Subtract tile_origin (min_tx, min_ty) to get position within stitched image.
      3. Multiply by tile_size to get pixel coordinates.
    Then use cv2.fillPoly to create the mask.

    Args:
        geojson_coords: List of [lng, lat] pairs (exterior ring).
        tile_origin: (min_tx, min_ty) of the stitched image grid.
        zoom: Zoom level.
        tile_size: Pixel size of each tile (e.g. 256 or 512).
        img_shape: (height, width) of the stitched image.

    Returns:
        Binary mask (uint8, 0 or 255) with the same dimensions as img_shape.
    """
    min_tx, min_ty = tile_origin
    pixel_points = []

    for coord in geojson_coords:
        lng, lat = coord[0], coord[1]
        tx_f, ty_f = _lng_lat_to_tile_float(lng, lat, zoom)
        px = (tx_f - min_tx) * tile_size
        py = (ty_f - min_ty) * tile_size
        pixel_points.append([px, py])

    pts = np.array(pixel_points, dtype=np.int32)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def pixel_spacing_to_meters(
    pixel_spacing: float, lat: float, zoom: int, tile_size: int = 256,
) -> float:
    """Convert a pixel-space measurement to meters.

    Args:
        pixel_spacing: Distance in pixels.
        lat: Latitude in degrees (for ground resolution calculation).
        zoom: Zoom level.
        tile_size: Pixel size of each tile (256 standard, 512 for hi-res).

    Returns:
        Distance in meters.
    """
    return pixel_spacing * meters_per_pixel(lat, zoom, tile_size)
