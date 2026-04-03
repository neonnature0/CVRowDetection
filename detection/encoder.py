"""Shared encoder backbone for all detection heads.

ResNet-50 + FPN producing a fused multi-scale feature map.
All detection heads consume these shared features.

Uses segmentation_models_pytorch internally (same library as training/model.py)
to avoid reimplementing the FPN neck. We instantiate smp.FPN but only expose
its encoder + decoder (the FPN merger), discarding the segmentation head.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from detection.config import DetectionConfig

logger = logging.getLogger(__name__)


class SharedEncoder(nn.Module):
    """ResNet-50 backbone with Feature Pyramid Network.

    Processes an input image and outputs a fused feature map at 1/4 resolution
    (P2 level, 256 channels by default). All task-specific heads consume this
    shared representation.
    """

    def __init__(self, config: DetectionConfig | None = None):
        super().__init__()
        if config is None:
            config = DetectionConfig()

        # Build an smp.FPN model — we use its encoder + decoder (FPN neck),
        # but discard the segmentation head. This gives us a battle-tested FPN
        # implementation with proper lateral connections and top-down pathway.
        self._fpn_model = smp.FPN(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=1,  # dummy — we don't use the segmentation head
            activation=None,
            encoder_depth=5,
            decoder_pyramid_channels=config.fpn_channels,
            decoder_segmentation_channels=config.fpn_channels,
        )

        # Expose encoder and decoder as named attributes for differential LR
        self.encoder = self._fpn_model.encoder
        self.decoder = self._fpn_model.decoder

        # Store the output channels for heads to query
        self.out_channels = config.fpn_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process image through ResNet-50 encoder + FPN decoder.

        Args:
            x: Input tensor (B, C, H, W), expects RGB float [0, 1].

        Returns:
            Fused feature map (B, fpn_channels, H/4, W/4).
        """
        features = self.encoder(x)
        fpn_output = self.decoder(features)
        return fpn_output

    def get_param_groups(self, lr_encoder: float, lr_head: float) -> list[dict]:
        """Return parameter groups with differential learning rates.

        The encoder (pretrained ResNet-50) gets a lower LR to preserve
        pretrained features. The FPN decoder gets the full head LR since
        it's randomly initialized.
        """
        return [
            {"params": self.encoder.parameters(), "lr": lr_encoder},
            {"params": self.decoder.parameters(), "lr": lr_head},
        ]


def save_encoder(encoder: SharedEncoder, path: Path | str) -> None:
    """Save encoder state dict to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path)
    logger.info("Saved encoder to %s", path)


def load_encoder(
    path: Path | str,
    config: DetectionConfig | None = None,
    device: str = "cpu",
    freeze: bool = False,
) -> SharedEncoder:
    """Load encoder from a checkpoint.

    Args:
        path: Path to encoder state dict.
        config: Detection config (uses defaults if None).
        device: Device to load to.
        freeze: If True, freeze all encoder parameters.
    """
    if config is None:
        config = DetectionConfig()
    # Don't load pretrained weights — we'll load from checkpoint
    config_no_weights = DetectionConfig(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        fpn_channels=config.fpn_channels,
        in_channels=config.in_channels,
    )
    encoder = SharedEncoder(config_no_weights)
    state = torch.load(str(path), map_location=device, weights_only=True)
    encoder.load_state_dict(state)

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    encoder.to(device)
    logger.info("Loaded encoder from %s (freeze=%s)", path, freeze)
    return encoder
