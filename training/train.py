"""Training loop for vineyard row likelihood U-Net.

Usage:
    python -m training.train
    python -m training.train --epochs 50 --batch-size 4 --lr 3e-4
    python -m training.train --run-test --checkpoint training/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from training.dataset import RowLikelihoodDataset, get_train_transform, get_val_transform
from training.model import create_model


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice score (for monitoring, not loss)."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    if union < 1e-6:
        return 1.0
    return float(2 * intersection / union)


def load_split_files(
    data_dir: Path,
    split_names: list[str],
) -> tuple[list[Path], list[Path]]:
    """Load patch and target file paths for given split block names."""
    patches_dir = data_dir / "patches"
    targets_dir = data_dir / "targets"
    patch_files = []
    target_files = []

    for f in sorted(patches_dir.glob("*.png")):
        stem = f.stem
        for block_name in split_names:
            if stem.startswith(block_name + "_"):
                target_path = targets_dir / (stem + ".npy")
                if target_path.exists():
                    patch_files.append(f)
                    target_files.append(target_path)
                break

    return patch_files, target_files


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce_loss: nn.Module,
    dice_loss: nn.Module,
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
        loss = 0.5 * bce_loss(preds, targets) + 0.5 * dice_loss(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(preds, targets)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    bce_loss: nn.Module,
    dice_loss: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Run evaluation on val/test set. Returns (mean_loss, mean_dice)."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = 0.5 * bce_loss(preds, targets) + 0.5 * dice_loss(preds, targets)
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
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_dices, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Validation Dice")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train row likelihood U-Net")
    parser.add_argument("--data-dir", type=str, default="dataset/training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder", type=str, default="mobilenet_v2")
    parser.add_argument("--output-dir", type=str, default="training/checkpoints")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--run-test", action="store_true", help="Run test evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint for test evaluation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    # Load splits
    with open(data_dir / "splits.json") as f:
        splits = json.load(f)

    print(f"Splits: train={len(splits['train'])} blocks, val={len(splits['val'])} blocks, test={len(splits['test'])} blocks")

    # Load file lists
    train_patches, train_targets = load_split_files(data_dir, splits["train"])
    val_patches, val_targets = load_split_files(data_dir, splits["val"])
    test_patches, test_targets = load_split_files(data_dir, splits["test"])
    print(f"Patches: train={len(train_patches)}, val={len(val_patches)}, test={len(test_patches)}")

    if len(train_patches) == 0:
        print("ERROR: No training patches found!")
        sys.exit(1)

    # Create model
    model = create_model(encoder_name=args.encoder)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.encoder} encoder, {n_params / 1e6:.1f}M parameters")

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

    # Test-only mode
    if args.run_test and args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)

        test_ds = RowLikelihoodDataset(test_patches, test_targets, transform=get_val_transform())
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        test_loss, test_dice = run_evaluation(model, test_loader, bce_loss, dice_loss, device)
        print(f"\nTest Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")
        return

    # Datasets
    train_ds = RowLikelihoodDataset(train_patches, train_targets, transform=get_train_transform())
    val_ds = RowLikelihoodDataset(val_patches, val_targets, transform=get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_dice = 0.0
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_dices: list[float] = []

    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device)
        val_loss, val_dice = run_evaluation(model, val_loader, bce_loss, dice_loss, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_counter += 1

        marker = " *" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | Best: {best_dice:.4f}{marker} | "
            f"LR: {lr:.1e} | {elapsed:.0f}s"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Save last model and curves
    torch.save(model.state_dict(), output_dir / "last_model.pth")
    save_training_curves(train_losses, val_losses, val_dices, output_dir / "training_curves.png")

    # Save training config
    config = {
        "encoder": args.encoder,
        "epochs_trained": len(train_losses),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_dice": best_dice,
        "n_train_patches": len(train_patches),
        "n_val_patches": len(val_patches),
        "patch_size": 256,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")

    # Final test evaluation with best model
    if test_patches:
        model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device, weights_only=True))
        test_ds = RowLikelihoodDataset(test_patches, test_targets, transform=get_val_transform())
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loss, test_dice = run_evaluation(model, test_loader, bce_loss, dice_loss, device)
        print(f"Test Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")


if __name__ == "__main__":
    main()
