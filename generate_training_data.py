#!/usr/bin/env python3
"""
Generate ML training data from annotated blocks.

Converts annotations into paired image patches and row-likelihood heatmaps
suitable for training a U-Net or similar segmentation model.

Usage:
    python generate_training_data.py
    python generate_training_data.py --patch-size 512 --overlap 0.25
    python generate_training_data.py --status complete  # Only use complete annotations
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_DIR = Path("dataset")
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
TRAINING_DIR = DATASET_DIR / "training"
PATCHES_DIR = TRAINING_DIR / "patches"
TARGETS_DIR = TRAINING_DIR / "targets"


def generate_heatmap(
    image_shape: tuple[int, int],  # (h, w)
    mask: np.ndarray,
    row_centerlines: list[list[tuple[float, float]]],
    spacing_px: float,
) -> np.ndarray:
    """Generate a row-likelihood heatmap from annotated polyline centerlines.

    Walks along each polyline and stamps a Gaussian at each point, producing
    curved ridges that follow the actual row shape.

    Returns float32 heatmap in [0, 1].
    """
    h, w = image_shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    sigma = max(1.0, spacing_px * 0.15)
    radius = int(sigma * 4)  # stamp radius in pixels

    for centerline in row_centerlines:
        for x, y in centerline:
            ix, iy = int(round(x)), int(round(y))
            # Stamp a small Gaussian patch around each point
            y0 = max(0, iy - radius)
            y1 = min(h, iy + radius + 1)
            x0 = max(0, ix - radius)
            x1 = min(w, ix + radius + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist_sq = (xx.astype(np.float32) - x) ** 2 + (yy.astype(np.float32) - y) ** 2
            patch = np.exp(-0.5 * dist_sq / (sigma * sigma))
            heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)

    heatmap *= (mask > 0).astype(np.float32)
    return heatmap


def extract_patches(
    image: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    overlap: float,
    min_mask_coverage: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Cut image and heatmap into overlapping patches.

    Returns list of (image_patch, heatmap_patch, row_offset, col_offset).
    Patches with mask coverage < min_mask_coverage are discarded.
    """
    h, w = image.shape[:2]
    step = int(patch_size * (1 - overlap))
    patches = []

    for r in range(0, h - patch_size + 1, step):
        for c in range(0, w - patch_size + 1, step):
            mask_patch = mask[r:r+patch_size, c:c+patch_size]
            coverage = np.mean(mask_patch > 0)
            if coverage < min_mask_coverage:
                continue

            img_patch = image[r:r+patch_size, c:c+patch_size]
            hm_patch = heatmap[r:r+patch_size, c:c+patch_size]
            patches.append((img_patch, hm_patch, r, c))

    return patches


def rotate_to_vertical(image: np.ndarray, mask: np.ndarray, heatmap: np.ndarray, angle_deg: float):
    """Rotate image/mask/heatmap so vine rows are vertical (90°)."""
    rotation_angle = 90.0 - angle_deg
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rot_img = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)
    rot_mask = cv2.warpAffine(mask, M, (new_w, new_h), borderValue=0)
    rot_heatmap = cv2.warpAffine(heatmap, M, (new_w, new_h), borderValue=0.0)
    return rot_img, rot_mask, rot_heatmap


def process_annotation(
    annotation_path: Path,
    patch_size: int,
    overlap: float,
    min_coverage: float,
    align_rows: bool = False,
    patches_dir: Path = PATCHES_DIR,
    targets_dir: Path = TARGETS_DIR,
) -> int:
    """Process one annotation file into training patches. Returns patch count."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    image_path = DATASET_DIR / ann["image_file"]
    mask_path = DATASET_DIR / ann["mask_file"]

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        logger.error("Could not load image/mask for %s", annotation_path.name)
        return 0

    h, w = image.shape[:2]
    mpp = ann["meters_per_pixel"]
    angle_deg = ann["angle_deg"]

    # Extract centerlines (handle both polyline and legacy perp format)
    cx, cy = w / 2.0, h / 2.0
    centerlines = []
    mean_perps = []
    for r in ann["rows"]:
        if "centerline_px" in r and r["centerline_px"]:
            cl = [(float(p[0]), float(p[1])) for p in r["centerline_px"]]
            centerlines.append(cl)
            angle_rad = math.radians(angle_deg)
            pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)
            perps = [(p[0] - cx) * pdx + (p[1] - cy) * pdy for p in cl]
            mean_perps.append(float(np.mean(perps)))
        elif "perp_position_px" in r:
            mean_perps.append(r["perp_position_px"])
            # Convert to straight-line polyline for heatmap
            angle_rad = math.radians(angle_deg)
            rdx, rdy = math.cos(angle_rad), math.sin(angle_rad)
            pdx, pdy = -math.sin(angle_rad), math.cos(angle_rad)
            lx = cx + r["perp_position_px"] * pdx
            ly = cy + r["perp_position_px"] * pdy
            diag = math.sqrt(w*w + h*h) / 2
            cl = [(lx + t * rdx, ly + t * rdy) for t in np.arange(-diag, diag, 3.0)]
            cl = [(x, y) for x, y in cl if 0 <= int(x) < w and 0 <= int(y) < h and mask[int(y), int(x)] > 0]
            centerlines.append(cl)

    if len(centerlines) < 2:
        logger.warning("Too few rows in %s, skipping", annotation_path.name)
        return 0

    # Compute mean spacing in pixels from mean perp positions
    mean_perps.sort()
    spacings = [mean_perps[i+1] - mean_perps[i] for i in range(len(mean_perps) - 1)]
    spacing_px = float(np.median(spacings))

    # Generate heatmap from curved polylines
    heatmap = generate_heatmap((h, w), mask, centerlines, spacing_px)

    # Rotate to align rows vertically before patch extraction
    if align_rows:
        image, mask, heatmap = rotate_to_vertical(image, mask, heatmap, angle_deg)

    # Extract patches
    patches = extract_patches(image, heatmap, mask, patch_size, overlap, min_coverage)

    # Save patches
    stem = annotation_path.stem
    for idx, (img_patch, hm_patch, r, c) in enumerate(patches):
        patch_name = f"{stem}_{r:05d}_{c:05d}"
        cv2.imwrite(str(patches_dir / f"{patch_name}.png"), img_patch)
        np.save(str(targets_dir / f"{patch_name}.npy"), hm_patch)

    return len(patches)


def generate_splits(vineyards_per_file: dict[str, str]) -> dict:
    """Generate stratified train/val/test split by vineyard.

    No vineyard appears in multiple splits.
    """
    # Group files by vineyard
    vineyard_files: dict[str, list[str]] = {}
    for fname, vineyard in vineyards_per_file.items():
        vineyard_files.setdefault(vineyard, []).append(fname)

    vineyards = sorted(vineyard_files.keys())
    n = len(vineyards)

    # Simple split: 60% train, 20% val, 20% test
    train_end = max(1, int(n * 0.6))
    val_end = max(train_end + 1, int(n * 0.8))

    splits = {"train": [], "val": [], "test": []}
    for i, v in enumerate(vineyards):
        if i < train_end:
            splits["train"].extend(vineyard_files[v])
        elif i < val_end:
            splits["val"].extend(vineyard_files[v])
        else:
            splits["test"].extend(vineyard_files[v])

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate ML training data")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size in pixels")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap fraction between patches")
    parser.add_argument("--min-coverage", type=float, default=0.2, help="Min mask coverage per patch")
    parser.add_argument("--status", type=str, default="complete",
                        choices=["pending", "modified", "complete", "any"],
                        help="Only use annotations with this status (default: complete)")
    parser.add_argument("--align-rows", action="store_true",
                        help="Rotate blocks so rows are vertical before patch extraction")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: dataset/training or dataset/training_aligned)")
    args = parser.parse_args()

    # Resolve output directories
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.align_rows:
        output_dir = DATASET_DIR / "training_aligned"
    else:
        output_dir = TRAINING_DIR
    patches_dir = output_dir / "patches"
    targets_dir = output_dir / "targets"

    # Find annotations
    ann_files = sorted(
        p for p in ANNOTATIONS_DIR.glob("*.json")
        if p.name != "manifest.json"
    )

    if args.status != "any":
        filtered = []
        for f in ann_files:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("metadata", {}).get("status") == args.status:
                filtered.append(f)
        ann_files = filtered

    if not ann_files:
        print(f"No annotation files with status='{args.status}'.")
        print("Annotate blocks first with: python annotate.py")
        sys.exit(1)

    # Create output directories
    patches_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    align_label = " (align-rows)" if args.align_rows else ""
    print(f"\nGenerating training data from {len(ann_files)} annotations{align_label}...")
    print(f"  Patch size: {args.patch_size}px, Overlap: {args.overlap}, Min coverage: {args.min_coverage}")
    print(f"  Output: {output_dir.resolve()}\n")

    total_patches = 0
    vineyards = {}

    for i, f in enumerate(ann_files):
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        name = d["block_name"]
        vineyard = d["vineyard_name"]
        vineyards[f.stem] = vineyard

        print(f"  [{i+1}/{len(ann_files)}] {name} ({vineyard})...", end="", flush=True)
        n = process_annotation(
            f, args.patch_size, args.overlap, args.min_coverage,
            align_rows=args.align_rows, patches_dir=patches_dir, targets_dir=targets_dir,
        )
        total_patches += n
        print(f" {n} patches")

    # Generate splits
    if vineyards:
        splits = generate_splits(vineyards)
        splits_path = output_dir / "splits.json"
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"\nSplits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Dataset statistics
    print(f"\nTotal patches: {total_patches}")
    print(f"Output: {output_dir.resolve()}")

    # Compute channel statistics if patches exist
    patch_files = list(patches_dir.glob("*.png"))
    if patch_files and len(patch_files) <= 500:
        print("Computing channel statistics...")
        means, stds = [], []
        for pf in patch_files[:100]:  # sample 100 patches
            img = cv2.imread(str(pf)).astype(np.float32) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
        mean_bgr = np.mean(means, axis=0)
        std_bgr = np.mean(stds, axis=0)
        print(f"  Channel means (BGR): [{mean_bgr[0]:.3f}, {mean_bgr[1]:.3f}, {mean_bgr[2]:.3f}]")
        print(f"  Channel stds  (BGR): [{std_bgr[0]:.3f}, {std_bgr[1]:.3f}, {std_bgr[2]:.3f}]")

        stats = {
            "total_patches": total_patches,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "aligned": args.align_rows,
            "channel_means_bgr": mean_bgr.tolist(),
            "channel_stds_bgr": std_bgr.tolist(),
        }
        stats_path = output_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
