"""Training loop for block boundary detection.

Follows the same structure as training/train.py with key additions:
  - Differential learning rates (encoder 0.1x, head 1x)
  - Encoder freezing for first N epochs
  - 2-channel loss (interior + boundary)
  - Separate encoder/head checkpoints

Usage:
    python -m detection.train_blocks
    python -m detection.train_blocks --epochs 50 --batch-size 4 --freeze-epochs 10
    python -m detection.train_blocks --run-test --checkpoint-dir detection/checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from detection.config import DetectionConfig
from detection.encoder import SharedEncoder, save_encoder
from detection.heads.block_head import BlockDetectionHead, BlockDetector, save_head


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def get_train_transform() -> A.Compose:
    """Block training augmentations — more aggressive rotation than rows."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3,
            ),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ToTensorV2(),
        ],
        additional_targets={"mask_boundary": "mask"},
    )


def get_val_transform() -> A.Compose:
    return A.Compose(
        [ToTensorV2()],
        additional_targets={"mask_boundary": "mask"},
    )


class BlockDataset(Dataset):
    """Loads patch/target pairs for block detection training.

    Patches are RGB uint8 PNGs. Targets are (H, W, 2) float32 .npy
    files — channel 0 is interior mask, channel 1 is boundary mask.
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
        img_bgr = cv2.imread(str(self.patch_files[idx]))
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # H*W*3 uint8

        target = np.load(str(self.target_files[idx]))  # H*W*2 float32
        interior = target[:, :, 0]
        boundary = target[:, :, 1]

        if self.transform:
            augmented = self.transform(image=image, mask=interior, mask_boundary=boundary)
            image = augmented["image"]  # C*H*W uint8 tensor
            interior = augmented["mask"]  # H*W float32 tensor
            boundary = augmented["mask_boundary"]  # H*W float32 tensor

        # Normalize image to [0, 1]
        image = image.float() / 255.0

        # Stack targets: 2*H*W
        if not isinstance(interior, torch.Tensor):
            interior = torch.from_numpy(interior).float()
        if not isinstance(boundary, torch.Tensor):
            boundary = torch.from_numpy(boundary).float()
        target_tensor = torch.stack([interior, boundary], dim=0)  # 2*H*W

        return image, target_tensor


def load_split_files(
    data_dir: Path,
    split_names: list[str],
) -> tuple[list[Path], list[Path]]:
    """Load patch and target file paths for given split property names."""
    patches_dir = data_dir / "patches"
    targets_dir = data_dir / "targets"
    patch_files = []
    target_files = []

    for f in sorted(patches_dir.glob("*.png")):
        stem = f.stem
        for prop_name in split_names:
            safe_name = prop_name.replace(" ", "_").replace("/", "_")
            if stem.startswith(safe_name + "_"):
                target_path = targets_dir / (stem + ".npy")
                if target_path.exists():
                    patch_files.append(f)
                    target_files.append(target_path)
                break

    return patch_files, target_files


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def compute_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    bce_loss: nn.Module,
    dice_loss_fn: nn.Module,
    interior_weight: float = 0.7,
    boundary_weight: float = 0.3,
) -> torch.Tensor:
    """Compute weighted 2-channel BCE + Dice loss.

    Args:
        preds: (B, 2, H, W) raw logits.
        targets: (B, 2, H, W) float targets [0, 1].
        interior_weight: weight for interior channel loss.
        boundary_weight: weight for boundary channel loss.
    """
    # Interior loss (channel 0)
    pred_int = preds[:, 0:1]
    tgt_int = targets[:, 0:1]
    loss_int = 0.5 * bce_loss(pred_int, tgt_int) + 0.5 * dice_loss_fn(pred_int, tgt_int)

    # Boundary loss (channel 1)
    pred_bnd = preds[:, 1:2]
    tgt_bnd = targets[:, 1:2]
    loss_bnd = 0.5 * bce_loss(pred_bnd, tgt_bnd) + 0.5 * dice_loss_fn(pred_bnd, tgt_bnd)

    return interior_weight * loss_int + boundary_weight * loss_bnd


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice score for monitoring (interior channel only)."""
    pred_bin = (torch.sigmoid(pred[:, 0:1]) > threshold).float()
    tgt = target[:, 0:1]
    intersection = (pred_bin * tgt).sum()
    union = pred_bin.sum() + tgt.sum()
    if union < 1e-6:
        return 1.0
    return float(2 * intersection / union)


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    dice_loss_fn: nn.Module,
    config: DetectionConfig,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch. Returns (mean_loss, mean_dice)."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = compute_loss(
            preds, targets, bce_loss, dice_loss_fn,
            config.interior_loss_weight, config.boundary_loss_weight,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(preds, targets)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    bce_loss: nn.Module,
    dice_loss_fn: nn.Module,
    config: DetectionConfig,
    device: str,
) -> tuple[float, float]:
    """Run validation. Returns (mean_loss, mean_dice)."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = compute_loss(
            preds, targets, bce_loss, dice_loss_fn,
            config.interior_loss_weight, config.boundary_loss_weight,
        )
        total_loss += loss.item()
        total_dice += dice_score(preds, targets)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


def save_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_dices: list[float],
    output_path: Path,
):
    """Save training curves plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss (Interior + Boundary)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_dices, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Validation Interior Dice")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train block detection head")
    parser.add_argument("--data-dir", type=str, default="dataset/block_training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Head learning rate")
    parser.add_argument("--lr-encoder", type=float, default=None, help="Encoder LR (default: lr * 0.1)")
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Epochs to freeze encoder")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--encoder", type=str, default="mobilenet_v2")
    parser.add_argument("--output-dir", type=str, default="detection/checkpoints")
    parser.add_argument("--max-patches", type=int, default=None, help="Limit training patches (faster iteration)")
    parser.add_argument("--run-test", action="store_true", help="Test mode only")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint dir for test mode")
    args = parser.parse_args()

    config = DetectionConfig(
        encoder_name=args.encoder,
        lr_head=args.lr,
        lr_encoder=args.lr_encoder if args.lr_encoder else args.lr * 0.1,
        freeze_encoder_epochs=args.freeze_epochs,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    # Load splits
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    print(f"Splits: train={len(splits['train'])} props, val={len(splits['val'])} props, test={len(splits['test'])} props")

    train_patches, train_targets = load_split_files(data_dir, splits["train"])
    val_patches, val_targets = load_split_files(data_dir, splits["val"])
    test_patches, test_targets = load_split_files(data_dir, splits["test"])

    # Limit patches for faster iteration
    if args.max_patches and len(train_patches) > args.max_patches:
        train_patches = train_patches[:args.max_patches]
        train_targets = train_targets[:args.max_patches]
    if args.max_patches and len(val_patches) > args.max_patches // 4:
        val_patches = val_patches[:args.max_patches // 4]
        val_targets = val_targets[:args.max_patches // 4]

    print(f"Patches: train={len(train_patches)}, val={len(val_patches)}, test={len(test_patches)}")

    if len(train_patches) == 0 and not args.run_test:
        print("ERROR: No training patches found!")
        sys.exit(1)

    # Create model
    encoder = SharedEncoder(config)
    head = BlockDetectionHead(
        in_channels=config.fpn_channels,
        hidden_channels=config.block_head_hidden,
    )
    model = BlockDetector(encoder, head)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"Model: {config.encoder_name} encoder + block head, {n_params / 1e6:.1f}M params ({n_enc / 1e6:.1f}M encoder, {n_head / 1e6:.1f}M head)")

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

    # Test-only mode
    if args.run_test:
        ckpt_dir = Path(args.checkpoint_dir or args.output_dir)
        encoder.load_state_dict(torch.load(str(ckpt_dir / "encoder.pth"), map_location=device, weights_only=True))
        head.load_state_dict(torch.load(str(ckpt_dir / "block_head.pth"), map_location=device, weights_only=True))
        model.eval()

        test_ds = BlockDataset(test_patches, test_targets, transform=get_val_transform())
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
        test_loss, test_dice = run_validation(model, test_loader, bce_loss, dice_loss_fn, config, device)
        print(f"\nTest Loss: {test_loss:.4f} | Test Interior Dice: {test_dice:.4f}")
        return

    # Datasets
    train_ds = BlockDataset(train_patches, train_targets, transform=get_train_transform())
    val_ds = BlockDataset(val_patches, val_targets, transform=get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer with differential learning rates
    param_groups = encoder.get_param_groups(config.lr_encoder, config.lr_head)
    param_groups.append({"params": head.parameters(), "lr": config.lr_head})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Freeze encoder for initial epochs
    encoder_frozen = config.freeze_encoder_epochs > 0
    if encoder_frozen:
        for param in encoder.parameters():
            param.requires_grad = False
        print(f"Encoder frozen for first {config.freeze_encoder_epochs} epochs")

    # Training loop
    best_dice = 0.0
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_dices: list[float] = []

    print(f"\nTraining for {config.epochs} epochs (patience={config.patience})...\n")

    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()

        # Unfreeze encoder after freeze period
        if encoder_frozen and epoch > config.freeze_encoder_epochs:
            for param in encoder.parameters():
                param.requires_grad = True
            encoder_frozen = False
            print(f"--- Encoder unfrozen at epoch {epoch} ---")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, bce_loss, dice_loss_fn, config, device,
        )
        val_loss, val_dice = run_validation(
            model, val_loader, bce_loss, dice_loss_fn, config, device,
        )
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[-1]["lr"]  # head LR

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        is_best = val_dice > best_dice or best_dice == 0.0
        if is_best:
            best_dice = val_dice
            patience_counter = 0
            save_encoder(encoder, output_dir / "encoder.pth")
            save_head(head, output_dir / "block_head.pth")
        else:
            patience_counter += 1

        marker = " *" if is_best else ""
        frozen_tag = " [enc frozen]" if encoder_frozen else ""
        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | Best: {best_dice:.4f}{marker} | "
            f"LR: {lr:.1e}{frozen_tag} | {elapsed:.0f}s"
        )

        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {config.patience} epochs)")
            break

    # Save last model and curves
    save_encoder(encoder, output_dir / "encoder_last.pth")
    save_head(head, output_dir / "block_head_last.pth")
    save_training_curves(train_losses, val_losses, val_dices, output_dir / "training_curves.png")

    # Save training config
    train_config = {
        "encoder": config.encoder_name,
        "fpn_channels": config.fpn_channels,
        "head_hidden": config.block_head_hidden,
        "epochs_trained": len(train_losses),
        "batch_size": config.batch_size,
        "lr_head": config.lr_head,
        "lr_encoder": config.lr_encoder,
        "freeze_epochs": config.freeze_encoder_epochs,
        "best_val_dice": best_dice,
        "n_train_patches": len(train_patches),
        "n_val_patches": len(val_patches),
        "patch_size": config.patch_size,
        "interior_loss_weight": config.interior_loss_weight,
        "boundary_loss_weight": config.boundary_loss_weight,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"\nTraining complete. Best val Interior Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")

    # Final test with best model
    if test_patches and (output_dir / "encoder.pth").exists():
        encoder.load_state_dict(torch.load(str(output_dir / "encoder.pth"), map_location=device, weights_only=True))
        head.load_state_dict(torch.load(str(output_dir / "block_head.pth"), map_location=device, weights_only=True))
        model_for_test = BlockDetector(encoder, head)
        model_for_test.eval()
        test_ds = BlockDataset(test_patches, test_targets, transform=get_val_transform())
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
        test_loss, test_dice = run_validation(model_for_test, test_loader, bce_loss, dice_loss_fn, config, device)
        print(f"Test Loss: {test_loss:.4f} | Test Interior Dice: {test_dice:.4f}")


if __name__ == "__main__":
    main()
