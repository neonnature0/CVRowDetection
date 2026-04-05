"""Tile proxy — hides LINZ API key from the browser."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from gui.config import TILE_CACHE_DIR

load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter()

TILE_URLS = {
    "linz": (
        "https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad"
        "/{z}/{x}/{y}.webp?api={api_key}"
    ),
    "arcgis": (
        "https://server.arcgisonline.com/ArcGIS/rest/services"
        "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
}


@router.get("/{source}/{z}/{x}/{y}")
def get_tile(source: str, z: int, x: int, y: int):
    if source not in TILE_URLS:
        raise HTTPException(404, f"Unknown tile source: {source}")

    # Check disk cache
    cache_path = TILE_CACHE_DIR / source / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        return Response(content=cache_path.read_bytes(), media_type="image/png")

    # Build URL
    url_template = TILE_URLS[source]
    api_key = os.environ.get("LINZ_API_KEY", "")
    url = url_template.format(z=z, x=x, y=y, api_key=api_key)

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Tile fetch failed: %s", exc)
        raise HTTPException(502, "Tile fetch failed")

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)

    # Determine content type
    content_type = resp.headers.get("content-type", "image/png")
    return Response(content=resp.content, media_type=content_type)
