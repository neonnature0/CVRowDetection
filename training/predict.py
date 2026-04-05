"""Full-block inference: stitch patch predictions into a complete likelihood map."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from training.model import create_model

logger = logging.getLogger(__name__)


def predict_block_likelihood(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 192,
    device: str = "cpu",
) -> np.ndarray:
    """Run the U-Net on a full block image, return stitched likelihood map.

    Args:
        model: Trained U-Net model (set to .eval() mode, on device).
        image_bgr: Full block BGR image.
        mask: Binary mask (uint8, 0 or 255).
        patch_size: Patch size matching training.
        stride: Stride between patches (patch_size * (1 - overlap)).
        device: 'cpu' or 'cuda'.

    Returns:
        Likelihood map (float32, 0-1), same spatial shape as image.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    likelihood = np.zeros((h, w), dtype=np.float64)
    weight_map = np.zeros((h, w), dtype=np.float64)

    # Generate patch grid coordinates
    y_positions = list(range(0, max(1, h - patch_size + 1), stride))
    x_positions = list(range(0, max(1, w - patch_size + 1), stride))

    # Add edge patches if the grid doesn't cover the full image
    if y_positions[-1] + patch_size < h:
        y_positions.append(h - patch_size)
    if x_positions[-1] + patch_size < w:
        x_positions.append(w - patch_size)

    n_patches = len(y_positions) * len(x_positions)
    logger.info("ML inference: %d patches (%dx%d grid)", n_patches, len(y_positions), len(x_positions))

    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                patch = image_rgb[y:y + patch_size, x:x + patch_size]

                # Ensure correct size (may be smaller at edges)
                ph, pw = patch.shape[:2]
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                # To tensor: HWC uint8 -> CHW float [0,1]
                tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(device)

                # Forward pass + temperature-scaled sigmoid
                logits = model(tensor)
                if hasattr(model, '_temperature') and model._temperature != 1.0:
                    logits = logits / model._temperature
                pred = torch.sigmoid(logits).squeeze().cpu().numpy()

                # Accumulate (only the valid region)
                pred_crop = pred[:ph, :pw]
                likelihood[y:y + ph, x:x + pw] += pred_crop
                weight_map[y:y + ph, x:x + pw] += 1.0

    # Average overlapping regions
    likelihood = np.where(weight_map > 0, likelihood / weight_map, 0.0).astype(np.float32)

    # Apply mask
    likelihood *= (mask > 0).astype(np.float32)

    # Normalize to [0, 1] within mask
    mask_pixels = mask > 0
    if mask_pixels.any():
        lmax = float(likelihood[mask_pixels].max())
        if lmax > 1e-6:
            likelihood[mask_pixels] = likelihood[mask_pixels] / lmax

    return likelihood


def load_model(checkpoint_path: str, encoder: str = "mobilenet_v2", device: str = "cpu") -> torch.nn.Module:
    """Load a trained checkpoint."""
    model = create_model(encoder_name=encoder, encoder_weights=None)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
