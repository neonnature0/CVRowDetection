#!/usr/bin/env python3
"""
Generate block detection training data from annotations.

Reads block boundaries from:
  1. Standalone GeoJSON annotations (dataset/standalone/*.geojson)
  2. Cordyn DB export (dataset/cordyn_export.geojson)
  3. Existing test_blocks.json

For each property/location, fetches property-level aerial imagery,
generates 2-channel segmentation masks (interior + boundary), extracts
overlapping patches, and splits by property to avoid data leakage.

Usage:
    python prepare_block_dataset.py --all
    python prepare_block_dataset.py --source standalone
    python prepare_block_dataset.py --patch-size 512 --overlap 0.25
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from vinerow.acquisition.tile_fetcher import (
    fetch_and_stitch,
    TILE_SOURCES,
    auto_select_source,
    default_zoom,
)
from vinerow.acquisition.geo_utils import (
    polygon_bbox,
    polygon_to_pixel_mask,
    meters_per_pixel,
    _lng_lat_to_tile_float,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("dataset")
STANDALONE_DIR = DATASET_DIR / "standalone"
BLOCK_TRAINING_DIR = DATASET_DIR / "block_training"
PATCHES_DIR = BLOCK_TRAINING_DIR / "patches"
TARGETS_DIR = BLOCK_TRAINING_DIR / "targets"
BLOCK_IMAGES_DIR = DATASET_DIR / "block_images"
TILE_CACHE_DIR = "output/.tile_cache"


# ---------------------------------------------------------------------------
# Data loading — read block polygons from multiple sources
# ---------------------------------------------------------------------------


@dataclass
class PropertyBlocks:
    """All blocks for a single property/location."""
    name: str
    blocks: list[dict]  # each has 'polygon_lnglat': list[tuple], 'label': str
    source: str  # 'standalone', 'cordyn', 'test_blocks'

    @property
    def all_coords(self) -> list[tuple[float, float]]:
        """Flatten all polygon coords for bounding box computation."""
        coords = []
        for b in self.blocks:
            coords.extend(b["polygon_lnglat"])
        return coords


def load_standalone_annotations() -> list[PropertyBlocks]:
    """Load block annotations from dataset/standalone/*.geojson."""
    results = []
    for path in sorted(STANDALONE_DIR.glob("*.geojson")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        blocks = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            if props.get("feature_type") == "block" and geom.get("type") == "Polygon":
                ring = geom["coordinates"][0]
                if len(ring) > 1 and ring[0] == ring[-1]:
                    ring = ring[:-1]
                blocks.append({
                    "polygon_lnglat": [(c[0], c[1]) for c in ring],
                    "label": props.get("name", "unnamed"),
                })

        if blocks:
            name = data.get("metadata", {}).get("name", path.stem)
            results.append(PropertyBlocks(name=name, blocks=blocks, source="standalone"))
            logger.info("Loaded %d blocks from %s", len(blocks), path.name)

    return results


def load_cordyn_export() -> list[PropertyBlocks]:
    """Load blocks from Cordyn DB export (dataset/cordyn_export.geojson)."""
    export_path = DATASET_DIR / "cordyn_export.geojson"
    if not export_path.exists():
        logger.info("No cordyn_export.geojson found, skipping")
        return []

    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by vineyard/property
    vineyard_blocks: dict[str, list[dict]] = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        vineyard = props.get("vineyard_name", "Unknown")
        ring = geom["coordinates"][0]
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        vineyard_blocks.setdefault(vineyard, []).append({
            "polygon_lnglat": [(c[0], c[1]) for c in ring],
            "label": props.get("name", "unnamed"),
        })

    results = []
    for vineyard, blocks in vineyard_blocks.items():
        results.append(PropertyBlocks(name=vineyard, blocks=blocks, source="cordyn"))
        logger.info("Loaded %d blocks for %s from cordyn export", len(blocks), vineyard)

    return results


def load_test_blocks() -> list[PropertyBlocks]:
    """Load blocks from data/blocks/test_blocks.json."""
    path = Path("data/blocks/test_blocks.json")
    if not path.exists():
        logger.info("No data/blocks/test_blocks.json found, skipping")
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Handle both {"blocks": [...]} and flat list formats
    block_list = raw.get("blocks", raw) if isinstance(raw, dict) else raw

    # Group by vineyard
    vineyard_blocks: dict[str, list[dict]] = {}
    for block in block_list:
        vineyard = block.get("vineyard_name") or block.get("vineyard") or "Unknown"
        # boundary can be GeoJSON geometry object or flat coord list
        boundary = block.get("boundary") or block.get("geojson_coords")
        if not boundary:
            continue
        if isinstance(boundary, dict):
            coords = boundary.get("coordinates", [[]])[0]  # GeoJSON Polygon
        else:
            coords = boundary
        if not coords:
            continue
        vineyard_blocks.setdefault(vineyard, []).append({
            "polygon_lnglat": [(c[0], c[1]) for c in coords],
            "label": block.get("name", "unnamed"),
        })

    results = []
    for vineyard, blocks in vineyard_blocks.items():
        results.append(PropertyBlocks(name=vineyard, blocks=blocks, source="test_blocks"))
        logger.info("Loaded %d blocks for %s from test_blocks.json", len(blocks), vineyard)

    return results


# ---------------------------------------------------------------------------
# Property-level image fetching
# ---------------------------------------------------------------------------


def property_bbox(blocks: list[dict], padding_m: float = 100.0) -> list[list[float]]:
    """Compute a padded bounding box around all blocks as a GeoJSON polygon ring.

    Returns a GeoJSON-style coordinate ring [[lng, lat], ...] for the bbox.
    """
    all_lngs = []
    all_lats = []
    for b in blocks:
        for lng, lat in b["polygon_lnglat"]:
            all_lngs.append(lng)
            all_lats.append(lat)

    min_lng, max_lng = min(all_lngs), max(all_lngs)
    min_lat, max_lat = min(all_lats), max(all_lats)

    # Convert padding from meters to approximate degrees
    mid_lat = (min_lat + max_lat) / 2
    lat_deg_per_m = 1.0 / 111000.0
    lng_deg_per_m = 1.0 / (111000.0 * abs(np.cos(np.radians(mid_lat))))
    pad_lat = padding_m * lat_deg_per_m
    pad_lng = padding_m * lng_deg_per_m

    min_lng -= pad_lng
    max_lng += pad_lng
    min_lat -= pad_lat
    max_lat += pad_lat

    return [
        [min_lng, min_lat],
        [max_lng, min_lat],
        [max_lng, max_lat],
        [min_lng, max_lat],
        [min_lng, min_lat],
    ]


def fetch_property_image(
    prop: PropertyBlocks,
    zoom: int | None = None,
    padding_m: float = 100.0,
) -> tuple[np.ndarray, tuple[int, int], int, str] | None:
    """Fetch aerial imagery for an entire property.

    Returns (image_bgr, tile_origin, zoom, source_name) or None on failure.
    """
    bbox_ring = property_bbox(prop.blocks, padding_m)
    centroid_lng = sum(c[0] for c in bbox_ring[:4]) / 4
    centroid_lat = sum(c[1] for c in bbox_ring[:4]) / 4

    source_name = auto_select_source(centroid_lng)
    source = TILE_SOURCES[source_name]
    if zoom is None:
        zoom = default_zoom(source_name)

    try:
        # fetch_and_stitch expects a polygon ring — our bbox IS a polygon ring
        image, mask, tile_origin = fetch_and_stitch(
            source, bbox_ring, zoom, source_name, TILE_CACHE_DIR,
        )
        # For property images, we want the UNMASKED stitched image
        # fetch_and_stitch masks to the polygon, but our bbox IS the polygon
        # so the mask covers the whole area. The image is already what we need.
        return image, tile_origin, zoom, source_name
    except Exception as e:
        logger.error("Failed to fetch property image for %s: %s", prop.name, e)
        return None


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------


def generate_block_masks(
    blocks: list[dict],
    tile_origin: tuple[int, int],
    zoom: int,
    tile_size: int,
    image_shape: tuple[int, int],
    boundary_width_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate interior and boundary masks for all blocks.

    Args:
        blocks: List of block dicts with 'polygon_lnglat' key.
        tile_origin: (min_tx, min_ty) of the stitched image.
        zoom: Zoom level.
        tile_size: Tile pixel size.
        image_shape: (height, width) of the image.
        boundary_width_px: Width of boundary ring in pixels.

    Returns:
        (interior_mask, boundary_mask) — both float32 [0, 1].
    """
    h, w = image_shape[:2]
    interior = np.zeros((h, w), dtype=np.float32)
    per_block_masks = []

    for block in blocks:
        coords = block["polygon_lnglat"]
        # Close ring for polygon_to_pixel_mask
        ring = [[lng, lat] for lng, lat in coords]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])

        mask = polygon_to_pixel_mask(ring, tile_origin, zoom, tile_size, (h, w))
        per_block_masks.append(mask)
        interior = np.maximum(interior, (mask > 0).astype(np.float32))

    # Boundary = dilated edge of each block
    boundary = np.zeros((h, w), dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_width_px * 2 + 1, boundary_width_px * 2 + 1))

    for mask in per_block_masks:
        mask_f = (mask > 0).astype(np.uint8)
        eroded = cv2.erode(mask_f, kernel)
        edge = mask_f - eroded
        boundary = np.maximum(boundary, edge.astype(np.float32))

    return interior, boundary


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def extract_patches(
    image: np.ndarray,
    interior: np.ndarray,
    boundary: np.ndarray,
    patch_size: int,
    overlap: float,
    min_coverage: float = 0.0,
    max_negative_ratio: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray, int, int, bool]]:
    """Extract overlapping patches from image and target masks.

    Returns list of (image_patch, target_2ch, row, col, is_negative).
    Target is stacked as (H, W, 2) float32 — channel 0=interior, 1=boundary.
    """
    h, w = image.shape[:2]
    step = int(patch_size * (1 - overlap))

    positive_patches = []
    negative_patches = []

    for r in range(0, max(1, h - patch_size + 1), step):
        for c in range(0, max(1, w - patch_size + 1), step):
            int_patch = interior[r : r + patch_size, c : c + patch_size]
            coverage = float(np.mean(int_patch > 0))

            img_patch = image[r : r + patch_size, c : c + patch_size]
            bnd_patch = boundary[r : r + patch_size, c : c + patch_size]
            target = np.stack([int_patch, bnd_patch], axis=-1)  # (H, W, 2)

            if coverage > max(min_coverage, 0.05):
                positive_patches.append((img_patch, target, r, c, False))
            else:
                # Negative patch — no block content
                negative_patches.append((img_patch, target, r, c, True))

    # Balance negatives
    max_negatives = int(len(positive_patches) * max_negative_ratio)
    if len(negative_patches) > max_negatives:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(negative_patches), max_negatives, replace=False)
        negative_patches = [negative_patches[i] for i in indices]

    all_patches = positive_patches + negative_patches
    logger.info(
        "Extracted %d patches (%d positive, %d negative)",
        len(all_patches), len(positive_patches), len(negative_patches),
    )
    return all_patches


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------


def generate_splits(property_names: list[str]) -> dict[str, list[str]]:
    """Generate train/val/test split by property name.

    60/20/20 split. No property appears in multiple splits.
    """
    names = sorted(set(property_names))
    n = len(names)
    train_end = max(1, int(n * 0.6))
    val_end = max(train_end + 1, int(n * 0.8))

    return {
        "train": names[:train_end],
        "val": names[train_end:val_end],
        "test": names[val_end:],
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_property(
    prop: PropertyBlocks,
    patch_size: int = 512,
    overlap: float = 0.25,
    boundary_width_px: int = 3,
    zoom: int | None = None,
) -> int:
    """Process a single property: fetch image, generate masks, extract patches.

    Returns number of patches extracted.
    """
    # Fetch property-level imagery
    result = fetch_property_image(prop, zoom=zoom)
    if result is None:
        return 0

    image, tile_origin, zoom_used, source_name = result
    source = TILE_SOURCES[source_name]

    # Save property image for inspection
    BLOCK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = prop.name.replace(" ", "_").replace("/", "_")
    cv2.imwrite(str(BLOCK_IMAGES_DIR / f"{safe_name}_property.png"), image)

    # Generate masks
    interior, boundary = generate_block_masks(
        prop.blocks, tile_origin, zoom_used, source.tile_size,
        image.shape[:2], boundary_width_px,
    )

    # Save masks for inspection
    cv2.imwrite(
        str(BLOCK_IMAGES_DIR / f"{safe_name}_interior.png"),
        (interior * 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(BLOCK_IMAGES_DIR / f"{safe_name}_boundary.png"),
        (boundary * 255).astype(np.uint8),
    )

    # Extract patches
    patches = extract_patches(
        image, interior, boundary,
        patch_size, overlap,
        min_coverage=0.0,
        max_negative_ratio=1.0,
    )

    # Save patches
    for img_patch, target, r, c, is_neg in patches:
        patch_name = f"{safe_name}_{r:05d}_{c:05d}"
        cv2.imwrite(str(PATCHES_DIR / f"{patch_name}.png"), img_patch)
        np.save(str(TARGETS_DIR / f"{patch_name}.npy"), target)

    return len(patches)


def main():
    parser = argparse.ArgumentParser(description="Generate block detection training data")
    parser.add_argument("--all", action="store_true", help="Load from all sources")
    parser.add_argument("--source", type=str, default="all",
                        choices=["all", "standalone", "cordyn", "test_blocks"],
                        help="Which data source to use")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--boundary-width", type=int, default=3)
    parser.add_argument("--zoom", type=int, default=None)
    parser.add_argument("--padding", type=float, default=100.0, help="Padding around property bbox in meters")
    args = parser.parse_args()

    if args.all:
        args.source = "all"

    # Load block annotations
    properties: list[PropertyBlocks] = []
    if args.source in ("all", "standalone"):
        properties.extend(load_standalone_annotations())
    if args.source in ("all", "cordyn"):
        properties.extend(load_cordyn_export())
    if args.source in ("all", "test_blocks"):
        properties.extend(load_test_blocks())

    if not properties:
        print("No block annotations found. Use map_annotator.py to create some first.")
        sys.exit(1)

    # Create output directories
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)

    total_blocks = sum(len(p.blocks) for p in properties)
    print(f"\nProcessing {len(properties)} properties ({total_blocks} blocks total)")
    print(f"  Patch size: {args.patch_size}px, Overlap: {args.overlap}")
    print(f"  Boundary width: {args.boundary_width}px")
    print(f"  Output: {BLOCK_TRAINING_DIR.resolve()}\n")

    total_patches = 0
    property_names = []

    for i, prop in enumerate(properties):
        print(f"  [{i+1}/{len(properties)}] {prop.name} ({len(prop.blocks)} blocks, src={prop.source})...", end="", flush=True)
        n = process_property(
            prop,
            patch_size=args.patch_size,
            overlap=args.overlap,
            boundary_width_px=args.boundary_width,
            zoom=args.zoom,
        )
        total_patches += n
        if n > 0:
            property_names.append(prop.name)
        print(f" {n} patches")

    # Generate splits
    if property_names:
        splits = generate_splits(property_names)
        splits_path = BLOCK_TRAINING_DIR / "splits.json"
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"\nSplits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Compute statistics
    patch_files = list(PATCHES_DIR.glob("*.png"))
    if patch_files and len(patch_files) <= 500:
        print("Computing channel statistics...")
        means = []
        for pf in patch_files[:100]:
            img = cv2.imread(str(pf)).astype(np.float32) / 255.0
            means.append(img.mean(axis=(0, 1)))
        mean_bgr = np.mean(means, axis=0)
        print(f"  Channel means (BGR): [{mean_bgr[0]:.3f}, {mean_bgr[1]:.3f}, {mean_bgr[2]:.3f}]")

        stats = {
            "total_patches": total_patches,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "boundary_width_px": args.boundary_width,
            "n_properties": len(property_names),
            "channel_means_bgr": mean_bgr.tolist(),
        }
        with open(BLOCK_TRAINING_DIR / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    print(f"\nTotal patches: {total_patches}")
    print(f"Output: {BLOCK_TRAINING_DIR.resolve()}")


if __name__ == "__main__":
    main()
