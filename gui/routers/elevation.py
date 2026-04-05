"""Elevation data endpoints — canopy height check for block validation."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from gui.services import block_registry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{name}")
async def get_block_elevation(name: str, dataset: str = "marlborough_2020-2022"):
    """Get canopy height analysis for a block.

    Returns mean/max canopy height, vegetation percentages, and
    a vineyard likelihood assessment. Requires rasterio.
    """
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(404, f"Block '{name}' not found")

    boundary = block.get("boundary")
    if not boundary:
        raise HTTPException(400, "Block has no boundary")

    coords = boundary["coordinates"][0]

    from vinerow.elevation.linz_elevation import get_canopy_height
    result = await asyncio.to_thread(get_canopy_height, coords, dataset)

    if result is None:
        raise HTTPException(404, "No LiDAR coverage for this block")

    return result


@router.get("/{name}/coverage")
async def check_coverage(name: str, dataset: str = "marlborough_2020-2022"):
    """Quick check: does LiDAR data cover this block? (No rasterio needed.)"""
    block = block_registry.get_block(name)
    if block is None:
        raise HTTPException(404, f"Block '{name}' not found")

    coords = block["boundary"]["coordinates"][0]

    from vinerow.elevation.linz_elevation import get_elevation_for_block
    dsm = await asyncio.to_thread(get_elevation_for_block, coords, dataset, "dsm")

    if dsm is None:
        return {"covered": False, "dataset": dataset}

    return {
        "covered": True,
        "dataset": dataset,
        "tile_count": len(dsm["tiff_urls"]),
        "bbox": dsm["bbox"],
    }
