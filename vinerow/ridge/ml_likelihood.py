"""ML-based row-likelihood map using a trained U-Net.

Lazy-loads the model on first call to avoid importing PyTorch
at pipeline startup when using Gabor mode.
"""

from __future__ import annotations

import logging

import numpy as np

from vinerow.config import PipelineConfig
from vinerow.types import PreprocessedChannels

logger = logging.getLogger(__name__)

_cached_model = None
_cached_path: str | None = None


def compute_ml_likelihood(
    preprocessed: PreprocessedChannels,
    mask: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Compute row-likelihood map using the trained U-Net.

    Returns the same format as compute_row_likelihood() in likelihood.py:
    np.ndarray float32 [0, 1], same shape as mask.
    """
    global _cached_model, _cached_path

    model_path = config.ml_model_path

    # Lazy load model only when needed
    if _cached_model is None or _cached_path != model_path:
        import torch
        from training.model import create_model

        logger.info("Loading ML model from %s", model_path)
        model = create_model(encoder_name="mobilenet_v2", encoder_weights=None)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.train(False)  # set to inference mode
        _cached_model = model
        _cached_path = model_path
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("ML model loaded (%d params)", n_params)

    from training.predict import predict_block_likelihood

    likelihood = predict_block_likelihood(
        model=_cached_model,
        image_bgr=preprocessed.image_bgr,
        mask=mask,
        patch_size=256,
        stride=192,
        device="cpu",
    )

    return likelihood
