"""
Training Script for LP-ASRN

Implements the training loop with stability features from the papers:
- StepLR learning rate schedule (reduce by 0.9 every 5 epochs)
- Monitoring recognition rate instead of loss for LR scheduling
- Gradient clipping to prevent exploding gradients
- Frozen OCR discriminator for stability
- Early stopping based on recognition rate

Based on:
- Nascimento et al. "Enhancing License Plate Super-Resolution" (2024)
"""

import argparse
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.lp_dataset import create_dataloaders
from src.models.generator import Generator
from src.losses.lcofl import LCOFL
from src.losses.basic import L1Loss
from src.ocr.parseq_wrapper import ParseqOCR
from src.ocr.confusion_tracker import ConfusionTracker, MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train LP-ASRN")
    parser.add_argument("--config", type=str, default="configs/lp_asrn.yaml")
    parser.add_argument("--data-root", type=str, default="data/train")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def train_epoch(
    generator,
    ocr,
    train_loader,
    lcofl_loss,
    l1_loss,
    optimizer,
    device,
    epoch,
    args,
):
    """Train for one epoch."""
    generator.train()

    total_loss = 0.0
    total_lcofl = 0.0
    total_l1 = 0.0

    pred_texts_all = []
    gt_texts_all = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        lr_images = batch["lr"].to(device)
        hr_images = batch["hr"].to(device)
        gt_texts = batch["plate_text"]

        # Forward pass through generator
        sr_images = generator(lr_images)

        # Get OCR predictions
        with torch.no_grad():
            pred_logits = ocr(sr_images, return_logits=True)
            pred_texts = ocr.predict(sr_images)

        # Collect for confusion tracking
        pred_texts_all.extend(pred_texts)
        gt_texts_all.extend(gt_texts)

        # Compute LCOFL loss
        lcofl, lcofl_info = lcofl_loss(
            sr_images=sr_images,
            hr_images=hr_images,
            pred_logits=pred_logits,
            gt_texts=gt_texts,
            pred_texts=pred_texts,
        )

        # Compute L1 loss for pixel-level constraint
        l1 = l1_loss(sr_images, hr_images)

        # Total loss
        loss = lcofl + 0.1 * l1

        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability (from papers)
        if hasattr(args, 'gradient_clip') and args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), max_norm=args.gradient_clip
            )

        optimizer.step()

        total_loss += loss.item()
        total_lcofl += lcofl.item()
        total_l1 += l1.item()

        pbar.set_postfix({
            "loss": loss.item(),
            "lcofl": lcofl.item(),
            "l1": l1.item(),
        })

    avg_loss = total_loss / len(train_loader)
    avg_lcofl = total_lcofl / len(train_loader)
    avg_l1 = total_l1 / len(train_loader)

    return avg_loss, avg_lcofl, avg_l1, pred_texts_all, gt_texts_all


def validate(generator, ocr, val_loader, device):
    """Validate the model."""
    generator.eval()

    total_loss = 0.0
    word_correct = 0
    char_correct = 0
    total_chars = 0

    pred_texts_all = []
    gt_texts_all = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)
            gt_texts = batch["plate_text"]

            # Forward pass
            sr_images = generator(lr_images)

            # Get OCR predictions
            pred_logits = ocr(sr_images, return_logits=True)
            pred_texts = ocr.predict(sr_images)

            # Collect for confusion tracking
            pred_texts_all.extend(pred_texts)
            gt_texts_all.extend(gt_texts)

            # Compute accuracy
            for pred, gt in zip(pred_texts, gt_texts):
                if pred == gt:
                    word_correct += 1

                # Character accuracy
                for p_char, g_char in zip(pred, gt):
                    if p_char == g_char:
                        char_correct += 1
                    total_chars += 1

    word_acc = word_correct / len(val_loader.dataset)
    char_acc = char_correct / total_chars if total_chars > 0 else 0

    return word_acc, char_acc, pred_texts_all, gt_texts_all


def main(args):
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Extract config values
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    ocr_config = config.get("ocr", {})

    # Create save directory
    save_dir = Path(training_config.get("save_dir", "outputs/run_default"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        root_dir=args.data_root,
        batch_size=data_config.get("batch_size", 16),
        num_workers=data_config.get("num_workers", 4),
        image_size=tuple(data_config.get("lr_size", [17, 31])),
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create generator
    print("Creating generator...")
    generator = Generator(
        in_channels=3,
        out_channels=3,
        num_features=model_config.get("num_filters", 64),
        num_blocks=model_config.get("num_rrdb_blocks", 16),
        upscale_factor=model_config.get("upscale_factor", 2),
        use_deformable=model_config.get("use_deformable", True),
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {total_params:,}")

    # Create OCR model (frozen)
    print("Creating OCR model...")
    ocr = ParseqOCR(
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        max_length=ocr_config.get("max_length", 7),
        frozen=ocr_config.get("freeze_ocr", True),
    )

    # Load fine-tuned OCR if available
    ocr_path = ocr_config.get("finetuned_path")
    if ocr_path and Path(ocr_path).exists():
        print(f"Loading fine-tuned OCR from {ocr_path}")
        checkpoint = torch.load(ocr_path, map_location="cpu")
        ocr.load_state_dict(checkpoint["model_state_dict"])

    ocr = ocr.to(device)
    ocr.eval()  # Keep in eval mode (frozen)

    # Create loss functions
    lcofl_loss = LCOFL(
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        lambda_layout=loss_config.get("lambda_layout", 0.5),
        lambda_ssim=loss_config.get("lambda_ssim", 0.2),
        alpha=loss_config.get("alpha", 0.1),
        beta=loss_config.get("beta", 1.0),
    )

    l1_loss = L1Loss()

    # Setup optimizer
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=training_config.get("lr", 1e-4),
        betas=(training_config.get("beta1", 0.9), training_config.get("beta2", 0.999)),
    )

    # StepLR scheduler based on recognition rate (Paper 2 innovation)
    # Reduce LR by 0.9 every 5 epochs if no improvement in recognition rate
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config.get("lr_step_size", 5),
        gamma=training_config.get("lr_gamma", 0.9),
    )

    # Confusion tracker
    confusion_tracker = ConfusionTracker(
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    )

    # Metrics tracker
    metrics = MetricsTracker()

    # Resume from checkpoint
    start_epoch = 0
    best_word_acc = 0.0
    epochs_without_improvement = 0

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        generator.load_state_dict(checkpoint["generator_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_word_acc = checkpoint.get("best_word_acc", 0.0)

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, training_config.get("epochs", 100)):
        # Train
        train_loss, train_lcofl, train_l1, pred_texts, gt_texts = train_epoch(
            generator, ocr, train_loader, lcofl_loss, l1_loss,
            optimizer, device, epoch + 1, args
        )

        # Validate
        val_word_acc, val_char_acc, val_pred_texts, val_gt_texts = validate(
            generator, ocr, val_loader, device
        )

        # Update confusion tracker with validation predictions
        confusion_tracker.update(val_pred_texts, val_gt_texts)

        # Update LCOFL weights based on confusion
        confusion_matrix = confusion_tracker.confusion_matrix
        lcofl_loss.update_weights(confusion_matrix)

        # Log metrics
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}, LCOFL: {train_lcofl:.4f}, L1: {train_l1:.4f}")
        print(f"  Val Word Acc: {val_word_acc:.4f}, Val Char Acc: {val_char_acc:.4f}")

        # Update metrics
        metrics.update("train_loss", train_loss)
        metrics.update("val_word_acc", val_word_acc)
        metrics.update("val_char_acc", val_char_acc)

        # StepLR (every 5 epochs based on recognition rate)
        if (epoch + 1) % training_config.get("lr_step_size", 5) == 0:
            # Check if recognition rate improved
            if not metrics.is_improving("val_word_acc", patience=5, maximize=True):
                scheduler.step()
                print(f"  Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_word_acc > best_word_acc:
            best_word_acc = val_word_acc
            epochs_without_improvement = 0

            save_path = save_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_word_acc": val_word_acc,
                "val_char_acc": val_char_acc,
                "best_word_acc": best_word_acc,
            }, save_path)
            print(f"  Saved best model with val_word_acc={val_word_acc:.4f}")
        else:
            epochs_without_improvement += 1

        # Save latest checkpoint
        save_path = save_dir / "latest.pth"
        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_word_acc": val_word_acc,
            "val_char_acc": val_char_acc,
            "best_word_acc": best_word_acc,
        }, save_path)

        # Early stopping (from Paper 1)
        if epochs_without_improvement >= training_config.get("early_stop_patience", 20):
            print(f"No improvement for {epochs_without_improvement} epochs. Early stopping.")
            break

        # Save confusion report
        if (epoch + 1) % 5 == 0:
            with open(save_dir / f"confusion_epoch_{epoch+1}.txt", "w") as f:
                f.write(confusion_tracker.get_report())

    # Save final metrics
    metrics.save(save_dir / "metrics.json")

    # Save final confusion report
    with open(save_dir / "confusion_final.txt", "w") as f:
        f.write(confusion_tracker.get_report())

    print(f"Training complete! Best val_word_acc: {best_word_acc:.4f}")


if __name__ == "__main__":
    args = parse_args()

    # Add config values to args
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            training_config = config.get("training", {})
            for key, value in training_config.items():
                if not hasattr(args, key):
                    setattr(args, key, value)

    main(args)
