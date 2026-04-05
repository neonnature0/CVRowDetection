"""ML-based row-likelihood map using a trained segmentation model.

Lazy-loads the model on first call to avoid importing PyTorch
at pipeline startup when using Gabor mode.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from vinerow.config import PipelineConfig
from vinerow.types import PreprocessedChannels

logger = logging.getLogger(__name__)

_cached_model = None
_cached_path: str | None = None
_cached_decoder: str | None = None


def _rotate_image(image: np.ndarray, angle_deg: float, border_value=0) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image by angle_deg, expanding canvas to avoid cropping.

    Returns (rotated_image, rotation_matrix).
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    if image.ndim == 3:
        rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(border_value,) * image.shape[2])
    else:
        rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=border_value)
    return rotated, M


def _rotate_back(likelihood: np.ndarray, angle_deg: float, original_shape: tuple[int, int]) -> np.ndarray:
    """Rotate likelihood map back to original orientation and crop to original size."""
    h_rot, w_rot = likelihood.shape[:2]
    h_orig, w_orig = original_shape
    center = (w_rot / 2.0, h_rot / 2.0)
    # Inverse rotation
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h_rot * sin + w_rot * cos)
    new_h = int(h_rot * cos + w_rot * sin)
    M[0, 2] += (new_w - w_rot) / 2.0
    M[1, 2] += (new_h - h_rot) / 2.0
    unrotated = cv2.warpAffine(likelihood, M, (new_w, new_h), borderValue=0.0)
    # Center-crop to original size
    cy, cx = new_h // 2, new_w // 2
    y0 = cy - h_orig // 2
    x0 = cx - w_orig // 2
    return unrotated[y0:y0 + h_orig, x0:x0 + w_orig]


def compute_ml_likelihood(
    preprocessed: PreprocessedChannels,
    mask: np.ndarray,
    config: PipelineConfig,
    coarse_angle_deg: float | None = None,
) -> np.ndarray:
    """Compute row-likelihood map using the trained segmentation model.

    Returns the same format as compute_row_likelihood() in likelihood.py:
    np.ndarray float32 [0, 1], same shape as mask.
    """
    global _cached_model, _cached_path, _cached_decoder

    model_path = config.ml_model_path
    decoder_type = config.ml_decoder

    # Lazy load model (invalidate cache if path or decoder changed)
    if _cached_model is None or _cached_path != model_path or _cached_decoder != decoder_type:
        import torch
        from training.model import create_model

        logger.info("Loading ML model from %s (decoder=%s)", model_path, decoder_type)
        model = create_model(encoder_name="mobilenet_v2", encoder_weights=None, decoder_type=decoder_type)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        # Support both plain state_dict and dict with temperature
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            T = checkpoint.get("temperature", 1.0)
        else:
            model.load_state_dict(checkpoint)
            T = 1.0
        model.train(False)
        model._temperature = T
        if T != 1.0:
            logger.info("Temperature scaling: T=%.3f", T)
        _cached_model = model
        _cached_path = model_path
        _cached_decoder = decoder_type
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("ML model loaded (%d params)", n_params)

    from training.predict import predict_block_likelihood

    image_bgr = preprocessed.image_bgr
    run_mask = mask

    # Rotate to align rows vertically if configured
    align = config.ml_align_rows and coarse_angle_deg is not None
    if align:
        rotation_deg = 90.0 - coarse_angle_deg
        logger.info("Rotating image by %.1f° to align rows vertically", rotation_deg)
        image_bgr, _ = _rotate_image(image_bgr, rotation_deg, border_value=0)
        run_mask, _ = _rotate_image(mask, rotation_deg, border_value=0)

    likelihood = predict_block_likelihood(
        model=_cached_model,
        image_bgr=image_bgr,
        mask=run_mask,
        patch_size=256,
        stride=192,
        device="cpu",
    )

    if align:
        original_shape = mask.shape[:2]
        likelihood = _rotate_back(likelihood, rotation_deg, original_shape)
        # Re-apply original mask (rotation can introduce artifacts at edges)
        likelihood *= (mask > 0).astype(np.float32)

    return likelihood
