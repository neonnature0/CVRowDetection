"""U-Net model definition for row likelihood prediction."""

from __future__ import annotations

import segmentation_models_pytorch as smp


def create_model(
    encoder_name: str = "mobilenet_v2",
    encoder_weights: str | None = "imagenet",
) -> smp.Unet:
    """Create a U-Net with a lightweight pretrained encoder.

    Returns raw logits (no activation) — apply sigmoid at inference time.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )
