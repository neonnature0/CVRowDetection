"""Segmentation model definition for row likelihood prediction."""

from __future__ import annotations

import torch.nn as nn
import segmentation_models_pytorch as smp


def create_model(
    encoder_name: str = "mobilenet_v2",
    encoder_weights: str | None = "imagenet",
    decoder_type: str = "unet",
) -> nn.Module:
    """Create a segmentation model with a lightweight pretrained encoder.

    Args:
        decoder_type: 'unet' or 'fpn'.

    Returns raw logits (no activation) — apply sigmoid at inference time.
    """
    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
    if decoder_type == "fpn":
        return smp.FPN(**kwargs)
    return smp.Unet(**kwargs)
