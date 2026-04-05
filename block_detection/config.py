"""Configuration for the multi-head detection system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectionConfig:
    """All tunable parameters for the detection system.

    Follows the same pattern as vinerow/config.py — single dataclass
    with sensible defaults, override via CLI or constructor kwargs.
    """

    # --- Encoder ---
    encoder_name: str = "mobilenet_v2"
    encoder_weights: str | None = "imagenet"
    fpn_channels: int = 256
    in_channels: int = 3

    # --- Block detection head ---
    block_head_hidden: int = 64
    block_boundary_width_px: int = 3  # width of boundary ring in training masks

    # --- Training ---
    patch_size: int = 256
    batch_size: int = 4
    lr_encoder: float = 1e-5  # 0.1x of head LR
    lr_head: float = 1e-4
    freeze_encoder_epochs: int = 5
    epochs: int = 100
    patience: int = 20
    weight_decay: float = 1e-4
    interior_loss_weight: float = 0.7
    boundary_loss_weight: float = 0.3

    # --- Inference ---
    inference_stride: int = 192  # 256 * 0.75 = 192 (25% overlap)
    interior_threshold: float = 0.4
    boundary_threshold: float = 0.8
    min_block_area_px: int = 500
    simplify_tolerance_px: float = 2.0

    # --- Paths ---
    tile_cache_dir: str = "output/.tile_cache"
    checkpoint_dir: str = "block_detection/checkpoints"
    data_dir: str = "dataset/block_training"
