#!/usr/bin/env python3
"""
Download and convert Fields of The World (FTW) data for block detection training.

FTW provides 1.6M agricultural field boundaries with Sentinel-2 imagery across
24 countries. We use it to pre-train the block detection model — field boundaries
are visually similar to vineyard blocks from above.

FTW 3-class masks: 0=background, 1=field interior, 2=field boundary
Our format: (H, W, 2) float32 — channel 0=interior, channel 1=boundary

Usage:
    python prepare_ftw_dataset.py                              # Download France + Austria (default)
    python prepare_ftw_dataset.py --countries france,spain,portugal,south_africa
    python prepare_ftw_dataset.py --max-samples 500            # Limit samples per country
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FTW_ROOT = Path("dataset/ftw")
OUTPUT_DIR = Path("dataset/block_training")
PATCHES_DIR = OUTPUT_DIR / "patches"
TARGETS_DIR = OUTPUT_DIR / "targets"

# Wine-growing countries available in FTW
WINE_COUNTRIES = ["france", "austria", "spain", "portugal", "south_africa", "croatia"]


def convert_ftw_sample(
    sample: dict,
    index: int,
    country: str,
    split_prefix: str,
) -> str | None:
    """Convert a single FTW sample to our patch format.

    FTW samples:
        image: (8, 256, 256) float32 — 4 bands x 2 time windows
        mask:  (256, 256) int64 — 0=bg, 1=interior, 2=boundary

    We extract RGB from the first time window (bands 2,1,0 = R,G,B)
    and convert the 3-class mask to our 2-channel format.

    Returns the patch stem name, or None if skipped.
    """
    image_tensor = sample["image"]  # (8, 256, 256) or (C, H, W)
    mask_tensor = sample["mask"]    # (256, 256)

    # FTW image: 8 channels = [B02, B03, B04, B08] x 2 time windows
    # First 4 channels are window A: B02(Blue), B03(Green), B04(Red), B08(NIR)
    # We want RGB = channels [2, 1, 0] (Red, Green, Blue)
    if image_tensor.shape[0] >= 4:
        rgb = image_tensor[[2, 1, 0], :, :]  # R, G, B from window A
    else:
        rgb = image_tensor[:3, :, :]

    # Normalize: FTW recommends dividing by 3000, then clip to [0, 1]
    rgb = rgb.float() / 3000.0
    rgb = torch.clamp(rgb, 0, 1)

    # Convert to uint8 BGR for saving as PNG (matching our existing format)
    rgb_np = (rgb.numpy() * 255).astype(np.uint8)  # (3, H, W)
    rgb_np = rgb_np.transpose(1, 2, 0)  # (H, W, 3) RGB
    bgr_np = rgb_np[:, :, ::-1]  # BGR for cv2

    # Convert 3-class mask to 2-channel float32
    mask_np = mask_tensor.numpy()
    interior = (mask_np == 1).astype(np.float32)
    boundary = (mask_np == 2).astype(np.float32)
    target = np.stack([interior, boundary], axis=-1)  # (H, W, 2)

    # Skip samples with no field content (all background)
    if interior.sum() < 100:
        return None

    # Save
    patch_name = f"ftw_{country}_{split_prefix}_{index:05d}"
    cv2.imwrite(str(PATCHES_DIR / f"{patch_name}.png"), bgr_np)
    np.save(str(TARGETS_DIR / f"{patch_name}.npy"), target)

    return patch_name


def download_and_convert(
    countries: list[str],
    max_samples: int | None = None,
) -> dict[str, int]:
    """Download FTW data and convert to our format.

    Returns dict of {country: n_converted}.
    """
    from torchgeo.datasets import FieldsOfTheWorld

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for country in countries:
        logger.info("Processing FTW country: %s", country)

        for split in ["train", "val"]:
            try:
                ds = FieldsOfTheWorld(
                    root=str(FTW_ROOT),
                    split=split,
                    target="3-class",
                    countries=[country],
                    download=True,
                )
            except Exception as e:
                logger.error("Failed to load FTW %s/%s: %s", country, split, e)
                continue

            n_samples = len(ds)
            if max_samples and n_samples > max_samples:
                n_samples = max_samples

            logger.info("  %s/%s: %d samples (using %d)", country, split, len(ds), n_samples)

            converted = 0
            for i in range(n_samples):
                try:
                    sample = ds[i]
                    name = convert_ftw_sample(sample, i, country, split)
                    if name:
                        converted += 1
                except Exception as e:
                    logger.warning("  Failed sample %d: %s", i, e)
                    continue

                if (i + 1) % 50 == 0:
                    logger.info("  %s/%s: %d/%d processed, %d converted", country, split, i + 1, n_samples, converted)

            results[f"{country}_{split}"] = converted
            logger.info("  %s/%s: %d converted", country, split, converted)

    return results


def update_splits():
    """Update splits.json to include FTW data in training set."""
    splits_path = OUTPUT_DIR / "splits.json"

    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
    else:
        splits = {"train": [], "val": [], "test": []}

    # FTW patches use "ftw_{country}_{split}" prefix
    # Add FTW train data to our train split, FTW val to our val split
    ftw_patches = list(PATCHES_DIR.glob("ftw_*.png"))
    ftw_prefixes = set()
    for p in ftw_patches:
        # Extract "ftw_france_train" from "ftw_france_train_00001.png"
        parts = p.stem.rsplit("_", 1)
        if len(parts) == 2:
            ftw_prefixes.add(parts[0])

    for prefix in sorted(ftw_prefixes):
        if "_train_" in prefix or prefix.endswith("_train"):
            if prefix not in splits["train"]:
                splits["train"].append(prefix)
        elif "_val_" in prefix or prefix.endswith("_val"):
            if prefix not in splits["val"]:
                splits["val"].append(prefix)

    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info("Updated splits: train=%d, val=%d, test=%d", len(splits["train"]), len(splits["val"]), len(splits["test"]))


def main():
    parser = argparse.ArgumentParser(description="Download FTW field boundary data for pre-training")
    parser.add_argument("--countries", type=str, default="france,austria",
                        help="Comma-separated country names")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per country per split")
    parser.add_argument("--list-countries", action="store_true",
                        help="List available countries and exit")
    args = parser.parse_args()

    if args.list_countries:
        from torchgeo.datasets import FieldsOfTheWorld
        print("Available countries:", ", ".join(FieldsOfTheWorld.valid_countries))
        print("\nWine-growing countries:", ", ".join(WINE_COUNTRIES))
        return

    countries = [c.strip() for c in args.countries.split(",")]
    print(f"\nFTW Field Boundary Download")
    print(f"  Countries: {countries}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Output: {OUTPUT_DIR.resolve()}\n")

    results = download_and_convert(countries, args.max_samples)

    print(f"\nConversion results:")
    total = 0
    for key, count in results.items():
        print(f"  {key}: {count} patches")
        total += count
    print(f"  Total: {total} new patches")

    # Update splits
    update_splits()

    # Count total patches
    all_patches = list(PATCHES_DIR.glob("*.png"))
    print(f"\nTotal patches in dataset: {len(all_patches)}")


if __name__ == "__main__":
    main()
