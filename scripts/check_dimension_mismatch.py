#!/usr/bin/env python3
"""Check if annotation image_size matches what fetch_and_stitch would produce.

This doesn't actually fetch tiles — it reads cached detection images and
compares their dimensions against the annotation's image_size field.
"""

import json
import sys
from pathlib import Path

ANNOTATIONS_DIR = Path("dataset/annotations")
DETECTIONS_DIR = Path("output/detections")
IMAGES_DIR = Path("dataset/images")


def main():
    if not ANNOTATIONS_DIR.exists():
        print("No annotations directory found")
        return

    ann_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    ann_files = [f for f in ann_files if f.name != "manifest.json"]

    if not ann_files:
        print("No annotation files found")
        return

    mismatches = []
    checked = 0

    for ann_path in ann_files:
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        name = ann["block_name"]
        ann_w, ann_h = ann["image_size"]

        # Check cached detection image
        det_image = DETECTIONS_DIR / name / "image.png"
        ds_image = IMAGES_DIR / f"{name}.png"

        for label, img_path in [("detection", det_image), ("dataset", ds_image)]:
            if not img_path.exists():
                continue

            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                checked += 1

                if img_w != ann_w or img_h != ann_h:
                    mismatches.append({
                        "block": name,
                        "source": label,
                        "ann_size": (ann_w, ann_h),
                        "actual_size": (img_w, img_h),
                        "delta_w": img_w - ann_w,
                        "delta_h": img_h - ann_h,
                    })
            except ImportError:
                # No cv2 — try PIL
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    img_w, img_h = img.size
                    checked += 1

                    if img_w != ann_w or img_h != ann_h:
                        mismatches.append({
                            "block": name,
                            "source": label,
                            "ann_size": (ann_w, ann_h),
                            "actual_size": (img_w, img_h),
                            "delta_w": img_w - ann_w,
                            "delta_h": img_h - ann_h,
                        })
                except ImportError:
                    print("Neither cv2 nor PIL available — cannot check image dimensions")
                    sys.exit(1)

    print(f"\nChecked {checked} images across {len(ann_files)} annotations")
    if not mismatches:
        print("NO MISMATCHES — all annotation image_size values match actual images")
    else:
        print(f"\n{len(mismatches)} MISMATCHES FOUND:\n")
        for m in mismatches:
            print(f"  {m['block']} ({m['source']}): "
                  f"annotation says {m['ann_size']}, "
                  f"actual is {m['actual_size']} "
                  f"(delta: {m['delta_w']:+d}w, {m['delta_h']:+d}h)")

        # Quantify impact on perpendicular center
        print("\nCenter offset impact (pixels):")
        for m in mismatches:
            cx_ann = m["ann_size"][0] / 2.0
            cy_ann = m["ann_size"][1] / 2.0
            cx_act = m["actual_size"][0] / 2.0
            cy_act = m["actual_size"][1] / 2.0
            dx = cx_act - cx_ann
            dy = cy_act - cy_ann
            print(f"  {m['block']}: center shifts by ({dx:+.1f}, {dy:+.1f}) pixels")


if __name__ == "__main__":
    main()
