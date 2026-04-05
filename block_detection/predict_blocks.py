"""Block detection inference — overlapping patch prediction with stitching.

Follows the same patch-and-stitch approach as training/predict.py, adapted
for 2-channel output (interior + boundary) and polygon extraction.

Usage:
    python -m block_detection.predict_blocks --image property.png --encoder checkpoints/encoder.pth --head checkpoints/block_head.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vinerow.acquisition.geo_utils import pixel_to_lnglat

from block_detection.config import DetectionConfig
from block_detection.encoder import SharedEncoder, load_encoder
from block_detection.heads.block_head import (
    BlockDetectionHead,
    BlockDetector,
    load_head,
    masks_to_polygons,
)
from block_detection.types import BlockDetection, BlockDetectionResult

logger = logging.getLogger(__name__)


def predict_property_masks(
    model: BlockDetector,
    image_bgr: np.ndarray,
    patch_size: int = 512,
    stride: int = 384,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run block detection on a full property image.

    Same overlapping-patch inference as training/predict.py but for 2 channels.

    Args:
        model: Trained BlockDetector in mode.
        image_bgr: Full property BGR image (H, W, 3).
        patch_size: Patch size matching training.
        stride: Stride between patches.
        device: Inference device.

    Returns:
        (interior_prob, boundary_prob) -- both (H, W) float32 in [0, 1].
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 2-channel accumulators
    interior_acc = np.zeros((h, w), dtype=np.float64)
    boundary_acc = np.zeros((h, w), dtype=np.float64)
    weight_map = np.zeros((h, w), dtype=np.float64)

    # Generate patch grid
    y_positions = list(range(0, max(1, h - patch_size + 1), stride))
    x_positions = list(range(0, max(1, w - patch_size + 1), stride))

    # Add edge patches
    if y_positions[-1] + patch_size < h:
        y_positions.append(h - patch_size)
    if x_positions[-1] + patch_size < w:
        x_positions.append(w - patch_size)

    n_patches = len(y_positions) * len(x_positions)
    logger.info("Block inference: %d patches (%dx%d grid)", n_patches, len(y_positions), len(x_positions))

    model.eval()
    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                patch = image_rgb[y : y + patch_size, x : x + patch_size]

                # Handle edge patches that may be smaller
                ph, pw = patch.shape[:2]
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                # To tensor: HWC uint8 -> CHW float [0, 1]
                tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(device)

                # Forward pass
                logits = model(tensor)  # (1, 2, H, W)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (2, H, W)

                # Accumulate (only the valid region)
                interior_acc[y : y + ph, x : x + pw] += probs[0, :ph, :pw]
                boundary_acc[y : y + ph, x : x + pw] += probs[1, :ph, :pw]
                weight_map[y : y + ph, x : x + pw] += 1.0

    # Average overlapping regions
    valid = weight_map > 0
    interior = np.where(valid, interior_acc / weight_map, 0.0).astype(np.float32)
    boundary = np.where(valid, boundary_acc / weight_map, 0.0).astype(np.float32)

    return interior, boundary


def predict_blocks(
    image_bgr: np.ndarray,
    encoder_path: str | Path,
    head_path: str | Path,
    config: DetectionConfig | None = None,
    device: str = "cpu",
    tile_origin: tuple[int, int] | None = None,
    zoom: int | None = None,
    tile_size: int = 256,
) -> BlockDetectionResult:
    """Full block detection pipeline: image -> polygons.

    Args:
        image_bgr: Property-level BGR image.
        encoder_path: Path to encoder checkpoint.
        head_path: Path to block head checkpoint.
        config: Detection config (uses defaults if None).
        device: Inference device.
        tile_origin: Optional (min_tx, min_ty) for geographic conversion.
        zoom: Optional zoom level for geographic conversion.
        tile_size: Tile pixel size for geographic conversion.

    Returns:
        BlockDetectionResult with detected block polygons.
    """
    if config is None:
        config = DetectionConfig()

    t0 = time.perf_counter()

    # Load model
    encoder = load_encoder(encoder_path, config, device)
    head = load_head(head_path, config.fpn_channels, config.block_head_hidden, device)
    model = BlockDetector(encoder, head)
    model.to(device)

    # Run patch inference
    interior, boundary = predict_property_masks(
        model, image_bgr,
        patch_size=config.patch_size,
        stride=config.inference_stride,
        device=device,
    )

    # Post-process: masks -> polygons
    blocks = masks_to_polygons(interior, boundary, config)

    # Convert to geographic coordinates if tile info provided
    if tile_origin is not None and zoom is not None:
        for block in blocks:
            block.polygon_lnglat = [
                pixel_to_lnglat(px, py, tile_origin, zoom, tile_size)
                for px, py in block.polygon_px
            ]

    elapsed = time.perf_counter() - t0

    return BlockDetectionResult(
        blocks=blocks,
        interior_mask=interior,
        boundary_mask=boundary,
        image_size=(image_bgr.shape[0], image_bgr.shape[1]),
        processing_time_s=elapsed,
    )


def save_predictions_geojson(result: BlockDetectionResult, path: Path) -> None:
    """Save detection results as GeoJSON."""
    features = []
    for block in result.blocks:
        if block.polygon_lnglat:
            coords = [[lng, lat] for lng, lat in block.polygon_lnglat]
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "block_id": block.block_id,
                    "confidence": round(block.confidence, 3),
                    "area_px": round(block.area_px, 1),
                },
            })

    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "processing_time_s": round(result.processing_time_s, 2),
            "n_blocks": len(result.blocks),
        },
        "features": features,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    logger.info("Saved %d predicted blocks to %s", len(features), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Block detection inference")
    parser.add_argument("--image", type=str, required=True, help="Property image (BGR PNG)")
    parser.add_argument("--encoder", type=str, default="block_detection/checkpoints/encoder.pth")
    parser.add_argument("--head", type=str, default="block_detection/checkpoints/block_head.pth")
    parser.add_argument("--output", type=str, default=None, help="Output GeoJSON path")
    parser.add_argument("--save-masks", action="store_true", help="Save probability masks as images")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--interior-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    config = DetectionConfig(
        patch_size=args.patch_size,
        inference_stride=args.stride,
        interior_threshold=args.interior_threshold,
        boundary_threshold=args.boundary_threshold,
    )

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Could not read image: {args.image}")
        sys.exit(1)

    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"Encoder: {args.encoder}")
    print(f"Head: {args.head}")

    result = predict_blocks(image, args.encoder, args.head, config)

    print(f"\nDetected {len(result.blocks)} blocks in {result.processing_time_s:.1f}s")
    for block in result.blocks:
        print(f"  Block {block.block_id}: area={block.area_px:.0f}px, confidence={block.confidence:.3f}, vertices={len(block.polygon_px)}")

    # Save masks
    if args.save_masks and result.interior_mask is not None:
        image_path = Path(args.image)
        cv2.imwrite(
            str(image_path.with_suffix(".interior.png")),
            (result.interior_mask * 255).astype(np.uint8),
        )
        cv2.imwrite(
            str(image_path.with_suffix(".boundary.png")),
            (result.boundary_mask * 255).astype(np.uint8),
        )
        print("Saved probability masks")

    # Save overlay visualization
    overlay = image.copy()
    for block in result.blocks:
        pts = np.array([(int(x), int(y)) for x, y in block.polygon_px], dtype=np.int32)
        cv2.polylines(overlay, [pts], True, (255, 0, 255), 2)
        if pts.shape[0] > 0:
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(
                overlay, f"#{block.block_id} ({block.confidence:.2f})",
                (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1,
            )

    overlay_path = Path(args.image).replace(Path(args.image).stem + "_overlay.png")
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Saved overlay to {overlay_path}")

    # Save GeoJSON
    output_path = Path(args.output) if args.output else Path(args.image).with_suffix(".predictions.geojson")
    save_predictions_geojson(result, output_path)


if __name__ == "__main__":
    main()
