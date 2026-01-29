"""
Fine-tune Parseq OCR on License Plate Data

This script fine-tunes the Parseq model on the high-resolution
license plate images before using it for super-resolution training.
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.lp_dataset import create_dataloaders
from src.ocr.parseq_wrapper import ParseqOCR
from src.ocr.confusion_tracker import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Parseq on license plate data")
    parser.add_argument("--config", type=str, default="configs/lacd_srnn.yaml")
    parser.add_argument("--data-root", type=str, default="data/train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="checkpoints/parseq")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def finetune(args):
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load config if available
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        ocr_config = config.get("ocr", {})
        vocab = ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        max_length = ocr_config.get("max_length", 7)
    else:
        vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        max_length = 7

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(32, 64),  # HR image size
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("Creating Parseq model...")
    ocr = ParseqOCR(
        vocab=vocab,
        max_length=max_length,
        frozen=False,  # Unfreeze for fine-tuning
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        ocr.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    ocr = ocr.to(args.device)

    # Setup training
    optimizer = torch.optim.Adam(ocr.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Metrics tracker
    metrics = MetricsTracker()

    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        ocr.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            hr_images = batch["hr"].to(args.device)
            gt_texts = batch["plate_text"]

            # Forward pass
            logits = ocr(hr_images, return_logits=True)

            # Encode ground truth
            gt_indices = ocr.tokenizer.encode_batch(gt_texts).to(args.device)

            # Compute loss
            B, K, C = logits.shape
            logits_flat = logits.reshape(-1, C)
            targets_flat = gt_indices.reshape(-1)

            # Filter padding
            mask = targets_flat > 0
            if mask.sum() > 0:
                loss = criterion(logits_flat[mask], targets_flat[mask])
            else:
                loss = torch.tensor(0.0, device=args.device)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ocr.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            # Accuracy (character-level)
            pred_indices = logits.argmax(dim=-1)
            train_correct += (pred_indices == gt_indices).sum().item()
            train_total += gt_indices.numel()

            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0

        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation phase
        ocr.eval()
        val_correct = 0
        val_total = 0
        val_word_correct = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                hr_images = batch["hr"].to(args.device)
                gt_texts = batch["plate_text"]

                pred_texts = ocr.predict(hr_images)

                for pred, gt in zip(pred_texts, gt_texts):
                    # Character-level accuracy
                    for p_char, g_char in zip(pred, gt):
                        if p_char == g_char:
                            val_correct += 1
                        val_total += 1

                    # Word-level accuracy
                    if pred == gt:
                        val_word_correct += 1

        val_char_acc = val_correct / val_total if val_total > 0 else 0
        val_word_acc = val_word_correct / len(val_loader.dataset) if val_loader else 0

        print(f"Val Char Acc: {val_char_acc:.4f}, Val Word Acc: {val_word_acc:.4f}")

        # Update scheduler
        scheduler.step(avg_train_loss)

        # Track metrics
        metrics.update("train_loss", avg_train_loss)
        metrics.update("train_char_acc", train_acc)
        metrics.update("val_char_acc", val_char_acc)
        metrics.update("val_word_acc", val_word_acc)

        # Save checkpoint
        if val_word_acc > best_val_acc:
            best_val_acc = val_word_acc
            save_path = save_dir / "best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": ocr.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_word_acc": val_word_acc,
                "val_char_acc": val_char_acc,
            }, save_path)
            print(f"Saved best model with val_word_acc={val_word_acc:.4f}")

        # Save latest checkpoint
        save_path = save_dir / "latest.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": ocr.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_word_acc": val_word_acc,
            "val_char_acc": val_char_acc,
        }, save_path)

    # Save metrics
    metrics.save(save_dir / "metrics.json")

    print(f"Training complete! Best val_word_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    args = parse_args()
    finetune(args)
