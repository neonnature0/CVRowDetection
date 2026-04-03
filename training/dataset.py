"""PyTorch Dataset for vineyard row likelihood training."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT, p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ToTensorV2(),
    ])


def get_aligned_train_transform() -> A.Compose:
    """Transform for pre-aligned patches: rows are vertical, reduced rotation."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # No RandomRotate90 — rows are pre-aligned vertical
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT, p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    return A.Compose([ToTensorV2()])


class RowLikelihoodDataset(Dataset):
    """Loads patch/target pairs for row likelihood training.

    Patches are RGB uint8 PNGs. Targets are float32 [0,1] heatmaps (.npy).
    """

    def __init__(
        self,
        patch_files: list[Path],
        target_files: list[Path],
        transform: A.Compose | None = None,
    ):
        self.patch_files = patch_files
        self.target_files = target_files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image (BGR from cv2, convert to RGB)
        img_bgr = cv2.imread(str(self.patch_files[idx]))
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # H×W×3 uint8

        target = np.load(str(self.target_files[idx]))  # H×W float32

        if self.transform:
            augmented = self.transform(image=image, mask=target)
            image = augmented["image"]  # C×H×W uint8 tensor after ToTensorV2
            target = augmented["mask"]  # H×W float32 tensor

        # Normalize image to [0, 1]
        image = image.float() / 255.0

        # Ensure target is float tensor with channel dim
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        target = target.unsqueeze(0)  # 1×H×W

        return image, target
