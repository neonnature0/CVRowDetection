"""Block boundary detection head.

Takes the fused FPN feature map from SharedEncoder and produces a
2-channel segmentation mask:
  - Channel 0: block interior (1 inside any vineyard block, 0 outside)
  - Channel 1: block boundary (1 on block edges, 0 elsewhere)

Individual blocks are separated by subtracting boundary from interior
and running connected component analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detection.config import DetectionConfig
from detection.types import BlockDetection

logger = logging.getLogger(__name__)


class BlockDetectionHead(nn.Module):
    """Lightweight decoder for block boundary segmentation.

    Consumes the fused FPN feature map (B, fpn_channels, H/4, W/4) and
    produces 2-channel logits at the original input resolution.
    """

    def __init__(self, in_channels: int = 256, hidden_channels: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        # 2 channels: interior + boundary
        self.output = nn.Conv2d(hidden_channels, 2, 1)

    def forward(self, features: torch.Tensor, input_size: tuple[int, int] | None = None) -> torch.Tensor:
        """Produce block segmentation logits.

        Args:
            features: FPN output (B, C, H/4, W/4).
            input_size: Original (H, W) for upsampling. If None, upsamples 4x.

        Returns:
            Logits tensor (B, 2, H, W). Apply sigmoid for probabilities.
        """
        x = self.decoder(features)
        logits = self.output(x)

        if input_size is not None:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        else:
            logits = F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)

        return logits


class BlockDetector(nn.Module):
    """Combined encoder + block head for training and inference."""

    def __init__(self, encoder: nn.Module, head: BlockDetectionHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: image → 2-channel logits.

        Args:
            x: Input image (B, 3, H, W), float [0, 1].

        Returns:
            Logits (B, 2, H, W).
        """
        input_size = (x.shape[2], x.shape[3])
        features = self.encoder(x)
        return self.head(features, input_size=input_size)


def save_head(head: BlockDetectionHead, path: Path | str) -> None:
    """Save block head state dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), path)
    logger.info("Saved block head to %s", path)


def load_head(
    path: Path | str,
    in_channels: int = 256,
    hidden_channels: int = 64,
    device: str = "cpu",
) -> BlockDetectionHead:
    """Load block head from checkpoint."""
    head = BlockDetectionHead(in_channels=in_channels, hidden_channels=hidden_channels)
    state = torch.load(str(path), map_location=device, weights_only=True)
    head.load_state_dict(state)
    head.to(device)
    logger.info("Loaded block head from %s", path)
    return head


# ---------------------------------------------------------------------------
# Post-processing: masks → polygons
# ---------------------------------------------------------------------------


def masks_to_polygons(
    interior_prob: np.ndarray,
    boundary_prob: np.ndarray,
    config: DetectionConfig | None = None,
) -> list[BlockDetection]:
    """Convert probability maps to individual block polygons.

    Pipeline:
    1. Threshold interior + boundary channels
    2. Subtract boundary from interior to create seed regions
    3. Morphological cleanup
    4. Connected components to separate blocks
    5. Contour extraction + simplification
    6. Filter by minimum area

    Args:
        interior_prob: (H, W) float32 probability map [0, 1].
        boundary_prob: (H, W) float32 probability map [0, 1].
        config: Detection config for thresholds. Uses defaults if None.

    Returns:
        List of BlockDetection objects, one per detected block.
    """
    if config is None:
        config = DetectionConfig()

    # Diagnostic logging
    logger.info(
        "Post-process input: interior max=%.4f mean=%.4f, boundary max=%.4f mean=%.4f",
        interior_prob.max(), interior_prob.mean(), boundary_prob.max(), boundary_prob.mean(),
    )

    # 1. Threshold interior
    interior_bin = (interior_prob > config.interior_threshold).astype(np.uint8)
    logger.info("Interior pixels above %.2f threshold: %d", config.interior_threshold, interior_bin.sum())

    # 2. Boundary subtraction — only if boundary head is producing useful signal
    seeds = interior_bin.copy()
    if boundary_prob.max() > 0.1:
        boundary_bin = (boundary_prob > config.boundary_threshold).astype(np.uint8)
        seeds[boundary_bin > 0] = 0
        logger.info("After boundary subtraction: %d seed pixels", seeds.sum())

        # Fallback: if boundary killed everything, skip subtraction
        if seeds.sum() == 0 and interior_bin.sum() > 0:
            logger.warning("Boundary subtraction collapsed all seeds — falling back to interior only")
            seeds = interior_bin.copy()
    else:
        logger.info("Boundary max=%.4f too low, skipping subtraction", boundary_prob.max())

    # 3. Morphological cleanup — remove small noise, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, kernel)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, kernel)
    logger.info("After morphology: %d seed pixels", seeds.sum())

    # 4. Connected components
    n_labels, labels = cv2.connectedComponents(seeds)
    logger.info("Connected components: %d (excluding background)", n_labels - 1)

    # 5. Extract polygons
    blocks: list[BlockDetection] = []
    for label_id in range(1, n_labels):  # skip background (0)
        component = (labels == label_id).astype(np.uint8)
        area = float(component.sum())

        if area < config.min_block_area_px:
            continue

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Take largest contour (there should be only one per connected component)
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 4:
            continue

        # Simplify with Douglas-Peucker
        epsilon = config.simplify_tolerance_px
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) < 4:
            approx = largest  # keep original if oversimplified

        polygon = [(float(p[0][0]), float(p[0][1])) for p in approx]

        # Confidence = mean interior probability within the component
        confidence = float(interior_prob[component > 0].mean())

        blocks.append(BlockDetection(
            block_id=label_id,
            polygon_px=polygon,
            area_px=area,
            confidence=confidence,
        ))

    # Sort by area descending (largest block first)
    blocks.sort(key=lambda b: b.area_px, reverse=True)

    logger.info(
        "Extracted %d blocks from masks (%d components, %d below min area)",
        len(blocks), n_labels - 1, (n_labels - 1) - len(blocks),
    )

    return blocks
