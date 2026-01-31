"""
Progressive Training for LP-ASRN

Implements multi-stage training approach:
1. Stage 1: Warm-up with L1 loss only
2. Stage 2: LCOFL training with frozen OCR
3. Stage 3: Fine-tuning with unfrozen OCR
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.models.generator import Generator
from src.ocr.parseq_wrapper import ParseqOCR
from src.losses.lcofl import LCOFL
from src.losses.basic import L1Loss
from src.ocr.confusion_tracker import ConfusionTracker, MetricsTracker
from src.utils.logger import TensorBoardLogger
from src.utils.visualizer import create_comparison_grid


class TrainingStage(Enum):
    """Training stages for progressive training."""
    PRETRAIN = "pretrain"
    WARMUP = "warmup"
    LCOFL = "lcofl"
    FINETUNE = "finetune"


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    name: str
    epochs: int
    lr: float
    loss_components: list  # ["l1"] or ["l1", "lcofl"]
    freeze_ocr: bool
    update_confusion: bool = False


class ProgressiveTrainer:
    """
    Progressive Trainer for LP-ASRN.

    Implements three-stage training:
    1. Warm-up: Stabilize network with L1 loss only
    2. LCOFL: Main training with character-driven loss
    3. Fine-tune: Joint optimization with OCR

    This approach improves training stability and final recognition accuracy.
    """

    def __init__(
        self,
        generator: Generator,
        ocr: ParseqOCR,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: Optional[TensorBoardLogger] = None,
        device: str = "cuda",
    ):
        """
        Initialize Progressive Trainer.

        Args:
            generator: Generator network
            ocr: OCR model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            logger: TensorBoard logger
            device: Device to use
        """
        self.generator = generator.to(device)
        self.ocr = ocr.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = device

        # Extract progressive training config
        self.progressive_config = config.get("progressive_training", {})
        self.tensorboard_config = config.get("tensorboard", {})

        # Initialize loss functions
        self.l1_loss = L1Loss()
        self.lcofl_loss = LCOFL(
            vocab=config.get("ocr", {}).get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            lambda_layout=config.get("loss", {}).get("lambda_layout", 0.5),
            lambda_ssim=config.get("loss", {}).get("lambda_ssim", 0.2),
            alpha=config.get("loss", {}).get("alpha", 0.1),
            beta=config.get("loss", {}).get("beta", 1.0),
        )

        # Stage configurations
        self.stage_configs = {
            TrainingStage.PRETRAIN: StageConfig(
                name="pretrain",
                epochs=self.progressive_config.get("stage0", {}).get("epochs", 20),
                lr=self.progressive_config.get("stage0", {}).get("lr", 1e-4),
                loss_components=["ocr"],
                freeze_ocr=False,
                update_confusion=False,
            ),
            TrainingStage.WARMUP: StageConfig(
                name="warmup",
                epochs=self.progressive_config.get("stage1", {}).get("epochs", 10),
                lr=self.progressive_config.get("stage1", {}).get("lr", 1e-4),
                loss_components=["l1"],
                freeze_ocr=True,
            ),
            TrainingStage.LCOFL: StageConfig(
                name="lcofl",
                epochs=self.progressive_config.get("stage2", {}).get("epochs", 50),
                lr=self.progressive_config.get("stage2", {}).get("lr", 1e-4),
                loss_components=["l1", "lcofl"],
                freeze_ocr=True,
                update_confusion=True,
            ),
            TrainingStage.FINETUNE: StageConfig(
                name="finetune",
                epochs=self.progressive_config.get("stage3", {}).get("epochs", 20),
                lr=self.progressive_config.get("stage3", {}).get("lr", 1e-5),
                loss_components=["l1", "lcofl"],
                freeze_ocr=False,
                update_confusion=True,
            ),
        }

        # Tracking
        self.current_stage = TrainingStage.WARMUP
        self.global_epoch = 0
        self.global_step = 0
        self.best_word_acc = 0.0
        self.epochs_without_improvement = 0

        # Confusion tracking
        self.confusion_tracker = ConfusionTracker(
            vocab=config.get("ocr", {}).get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        )
        self.metrics_tracker = MetricsTracker()

        # Save directory
        self.save_dir = Path(config.get("training", {}).get("save_dir", "checkpoints/lp_asrn"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def set_stage(self, stage: TrainingStage):
        """Set the current training stage."""
        self.current_stage = stage
        config = self.stage_configs[stage]

        # Update OCR frozen state
        for param in self.ocr.parameters():
            param.requires_grad = not config.freeze_ocr

        print(f"\n{'='*60}")
        print(f"STAGE: {stage.value.upper()}")
        print(f"{'='*60}")
        print(f"Epochs: {config.epochs}")
        print(f"Learning Rate: {config.lr}")
        print(f"Loss Components: {config.loss_components}")
        print(f"OCR Frozen: {config.freeze_ocr}")
        print(f"{'='*60}\n")

    def train_epoch(self, stage_config: StageConfig) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_lcofl = 0.0

        pred_texts_all = []
        gt_texts_all = []

        pbar = tqdm(self.train_loader, desc=f"Stage {self.current_stage.value}")
        for batch in pbar:
            lr_images = batch["lr"].to(self.device)
            hr_images = batch["hr"].to(self.device)
            gt_texts = batch["plate_text"]

            # Forward pass
            sr_images = self.generator(lr_images)

            # Compute losses based on stage
            l1 = self.l1_loss(sr_images, hr_images)
            loss = l1

            if "lcofl" in stage_config.loss_components:
                # Get OCR predictions
                with torch.no_grad():
                    pred_logits = self.ocr(sr_images, return_logits=True)
                    pred_texts = self.ocr.predict(sr_images)

                # Compute LCOFL
                lcofl, lcofl_info = self.lcofl_loss(
                    sr_images, hr_images, pred_logits, gt_texts, pred_texts
                )
                # Use configurable lambda_lcofl weight (default 1.0 instead of 0.1)
                lambda_lcofl = self.config.get("loss", {}).get("lambda_lcofl", 1.0)
                loss = loss + lambda_lcofl * lcofl
                total_lcofl += lcofl.item()

                pred_texts_all.extend(pred_texts)
                gt_texts_all.extend(gt_texts)

            total_loss += loss.item()
            total_l1 += l1.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get("training", {}).get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(),
                    max_norm=self.config["training"]["gradient_clip"],
                )

            self.optimizer.step()

            # Update progress
            if self.logger and self.global_step % 10 == 0:
                pbar.set_postfix({"loss": loss.item()})

            self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        avg_l1 = total_l1 / len(self.train_loader)
        avg_lcofl = total_lcofl / len(self.train_loader) if "lcofl" in stage_config.loss_components else 0

        return {
            "loss": avg_loss,
            "l1": avg_l1,
            "lcofl": avg_lcofl,
            "pred_texts": pred_texts_all,
            "gt_texts": gt_texts_all,
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.generator.eval()

        total_psnr = 0.0
        total_ssim = 0.0
        word_correct = 0
        char_correct = 0
        total_chars = 0

        pred_texts_all = []
        gt_texts_all = []

        sample_batch = None

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                lr_images = batch["lr"].to(self.device)
                hr_images = batch["hr"].to(self.device)
                gt_texts = batch["plate_text"]

                # Forward pass
                sr_images = self.generator(lr_images)

                # Store first batch for visualization
                if sample_batch is None and self.logger:
                    # Get predictions for this batch
                    pred_texts_batch = self.ocr.predict(sr_images)

                    sample_batch = {
                        "lr": lr_images,
                        "sr": sr_images,
                        "hr": hr_images,
                        "gt_texts": gt_texts,
                        "pred_texts": pred_texts_batch,
                    }

                # Compute PSNR/SSIM
                for i in range(sr_images.shape[0]):
                    mse = torch.mean((sr_images[i] - hr_images[i]) ** 2)
                    # Images are in [-1, 1] range, so max_value = 2.0
                    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse + 1e-10))
                    total_psnr += psnr.item()

                    # SSIM (simplified) - normalize MAE from [-1, 1] to [0, 1]
                    mae = torch.mean(torch.abs(sr_images[i] - hr_images[i]))
                    ssim = 1.0 - (mae / 2.0)
                    total_ssim += ssim.item()

                # OCR predictions
                pred_texts = self.ocr.predict(sr_images)
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

        avg_psnr = total_psnr / len(self.val_loader.dataset)
        avg_ssim = total_ssim / len(self.val_loader.dataset)
        word_acc = word_correct / len(self.val_loader.dataset)
        char_acc = char_correct / total_chars if total_chars > 0 else 0

        return {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "word_acc": word_acc,
            "char_acc": char_acc,
            "pred_texts": pred_texts_all,
            "gt_texts": gt_texts_all,
            "sample_batch": sample_batch,
        }

    def save_ocr_checkpoint(self, epoch: int, stage: TrainingStage):
        """Save OCR model checkpoint."""
        ocr_path = self.save_dir / f"ocr_stage_{stage.value}_epoch_{epoch}.pth"
        self.ocr.save(str(ocr_path))
        # Also save as default for later loading
        default_path = self.save_dir / "ocr_best.pth"
        self.ocr.save(str(default_path))

    def train_pretrain_stage(self, stage_config: StageConfig) -> float:
        """
        Train OCR model on license plate data (Stage 0: Pretraining).

        This stage trains only the OCR model on high-resolution license plate
        images to establish a baseline recognition capability before the
        super-resolution training begins.

        Args:
            stage_config: Configuration for this training stage

        Returns:
            Best word accuracy achieved during pretraining
        """
        import torch.nn as nn

        # Unfreeze OCR for training
        for param in self.ocr.parameters():
            param.requires_grad = True

        self.ocr.train()

        # Create optimizer for OCR only
        self.optimizer = optim.Adam(self.ocr.parameters(), lr=stage_config.lr)

        best_word_acc = 0.0
        self.epochs_without_improvement = 0

        print(f"\n{'='*60}")
        print(f"OCR PRETRAINING STAGE")
        print(f"{'='*60}")
        print(f"Epochs: {stage_config.epochs}")
        print(f"Learning Rate: {stage_config.lr}")
        print(f"Using CTC Loss: Yes")
        print(f"{'='*60}\n")

        for epoch in range(stage_config.epochs):
            total_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"OCR Pretrain Epoch {epoch+1}/{stage_config.epochs}")
            for batch in pbar:
                hr_images = batch["hr"].to(self.device)
                gt_texts = batch["plate_text"]

                # Forward pass
                logits = self.ocr(hr_images, return_logits=True)

                # Compute CTC loss
                loss = self.ocr.compute_ctc_loss(logits, gt_texts, device=self.device)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ocr.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(self.train_loader)

            # Validation
            val_metrics = self.validate()

            # Print results
            print(f"\nOCR Pretrain Epoch {epoch+1}/{stage_config.epochs}:")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Val Word Acc: {val_metrics['word_acc']:.4f}")
            print(f"  Val Char Acc: {val_metrics['char_acc']:.4f}")

            # Save best model
            if val_metrics['word_acc'] > best_word_acc:
                best_word_acc = val_metrics['word_acc']
                self.epochs_without_improvement = 0
                self.save_ocr_checkpoint(epoch, TrainingStage.PRETRAIN)
                print(f"  ✓ New best OCR: word_acc={val_metrics['word_acc']:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            patience = self.config.get("training", {}).get("early_stop_patience", 20)
            if self.epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

        # Freeze OCR after pretraining for subsequent stages
        for param in self.ocr.parameters():
            param.requires_grad = False

        print(f"\nOCR Pretraining complete. Best word accuracy: {best_word_acc:.4f}\n")

        return best_word_acc

    def log_to_tensorboard(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        epoch: int,
    ):
        """Log metrics to TensorBoard."""
        if not self.logger:
            return

        # Log scalars
        self.logger.log_scalars(train_metrics, epoch, "train")
        self.logger.log_scalars(val_metrics, epoch, "val")

        # Log model weights
        if epoch % 10 == 0:
            self.logger.log_model_weights(self.generator, epoch)

        # Log comparison images
        if val_metrics.get("sample_batch") and epoch % 5 == 0:
            batch = val_metrics["sample_batch"]
            try:
                grid = create_comparison_grid(
                    batch["lr"],
                    batch["sr"],
                    batch["hr"],
                    batch["gt_texts"],
                    batch["pred_texts"],  # Use sample_batch predictions, not all validation predictions
                    max_images=8,
                )
                self.logger.log_image_grid(f"comparison/epoch_{epoch}", grid, epoch)
            except Exception as e:
                # Log detailed error information with tensor shapes
                lr_shape = getattr(batch.get("lr"), "shape", "N/A")
                sr_shape = getattr(batch.get("sr"), "shape", "N/A")
                hr_shape = getattr(batch.get("hr"), "shape", "N/A")
                gt_count = len(batch.get("gt_texts", []))
                pred_count = len(batch.get("pred_texts", []))
                print(f"Warning: Could not log images: {e}")
                print(f"  Tensor shapes - LR: {lr_shape}, SR: {sr_shape}, HR: {hr_shape}")
                print(f"  Text counts - GT: {gt_count}, Pred: {pred_count}")

        # Log confusion matrix
        if epoch % 5 == 0 and self.current_stage in [TrainingStage.LCOFL, TrainingStage.FINETUNE]:
            self.confusion_tracker.update(val_metrics["pred_texts"], val_metrics["gt_texts"])
            try:
                labels = list(self.config["ocr"]["vocab"])
                self.logger.log_confusion_matrix(
                    self.confusion_tracker.confusion_matrix,
                    labels,
                    epoch,
                )
            except Exception as e:
                print(f"Warning: Could not log confusion matrix: {e}")

    def train_stage(self, stage: TrainingStage) -> float:
        """Train a specific stage."""
        self.set_stage(stage)
        config = self.stage_configs[stage]

        # Create optimizer for this stage
        if stage == TrainingStage.FINETUNE:
            # Train both generator and OCR
            params = list(self.generator.parameters()) + list(self.ocr.parameters())
        else:
            # Train only generator
            params = self.generator.parameters()

        self.optimizer = optim.Adam(params, lr=config.lr)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config.get("training", {}).get("lr_step_size", 5),
            gamma=self.config.get("training", {}).get("lr_gamma", 0.9),
        )

        start_epoch = self.global_epoch

        for epoch in range(config.epochs):
            current_global_epoch = start_epoch + epoch

            # Train
            train_metrics = self.train_epoch(config)

            # Validate
            val_metrics = self.validate()

            # Update confusion if needed
            if config.update_confusion:
                self.confusion_tracker.update(val_metrics["pred_texts"], val_metrics["gt_texts"])
                self.lcofl_loss.update_weights(self.confusion_tracker.confusion_matrix)

            # Log to TensorBoard
            self.log_to_tensorboard(train_metrics, val_metrics, epoch)

            # Print results
            print(f"Epoch {current_global_epoch + 1}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Val Word Acc: {val_metrics['word_acc']:.4f}")
            print(f"  Val Char Acc: {val_metrics['char_acc']:.4f}")

            # Update best model
            if val_metrics['word_acc'] > self.best_word_acc:
                self.best_word_acc = val_metrics['word_acc']
                self.epochs_without_improvement = 0

                self.save_checkpoint(current_global_epoch, stage)
                print(f"  ✓ New best model: word_acc={val_metrics['word_acc']:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping
            patience = self.config.get("training", {}).get("early_stop_patience", 20)
            if self.epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

        self.global_epoch = start_epoch + config.epochs

        return self.best_word_acc

    def save_checkpoint(self, epoch: int, stage: TrainingStage):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "stage": stage.value,
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_word_acc": self.best_word_acc,
            "global_epoch": self.global_epoch,
        }

        if not self.config.get("ocr", {}).get("freeze_ocr", True):
            checkpoint["ocr_state_dict"] = self.ocr.state_dict()

        save_path = self.save_dir / f"stage_{stage.value}_epoch_{epoch}.pth"
        torch.save(checkpoint, save_path)

        # Save best
        save_path = self.save_dir / "best.pth"
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])

        if "ocr_state_dict" in checkpoint:
            self.ocr.load_state_dict(checkpoint["ocr_state_dict"])

        self.global_epoch = checkpoint.get("global_epoch", 0)
        self.best_word_acc = checkpoint.get("best_word_acc", 0.0)

    def train_full_progressive(self) -> Dict[str, Any]:
        """Train all stages sequentially."""
        results = {}

        print("\n" + "=" * 60)
        print("PROGRESSIVE TRAINING START")
        print("=" * 60 + "\n")

        # Stage 0: OCR Pretraining
        print("\nStage 0: OCR Pretraining")
        print("Purpose: Train OCR model on license plate data")
        print("This ensures OCR can provide meaningful guidance to the generator")
        best_acc_stage0 = self.train_pretrain_stage(
            self.stage_configs[TrainingStage.PRETRAIN]
        )
        results["stage0_best_acc"] = best_acc_stage0

        # Stage 1: Warm-up
        print("\nStage 1: Warm-up (L1 loss only)")
        print("Purpose: Stabilize network with simple reconstruction loss")
        best_acc_stage1 = self.train_stage(TrainingStage.WARMUP)
        results["stage1_best_acc"] = best_acc_stage1

        # Stage 2: LCOFL
        print("\nStage 2: LCOFL Training (Character-driven loss)")
        print("Purpose: Optimize for character recognition with frozen OCR")
        best_acc_stage2 = self.train_stage(TrainingStage.LCOFL)
        results["stage2_best_acc"] = best_acc_stage2

        # Stage 3: Fine-tuning
        print("\nStage 3: Fine-tuning (Joint optimization)")
        print("Purpose: Refine with unfrozen OCR for joint training")
        best_acc_stage3 = self.train_stage(TrainingStage.FINETUNE)
        results["stage3_best_acc"] = best_acc_stage3

        # Save final results
        results["final_best_acc"] = self.best_word_acc

        with open(self.save_dir / "progressive_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 60)
        print("PROGRESSIVE TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nFinal Best Word Accuracy: {self.best_word_acc:.4f}")
        print(f"Results saved to: {self.save_dir}")

        return results


if __name__ == "__main__":
    # Test progressive trainer
    print("Testing ProgressiveTrainer...")

    from src.data.lp_dataset import create_dataloaders

    # Create small test config
    config = {
        "model": {"num_filters": 32, "num_rrdb_blocks": 2},
        "data": {"batch_size": 2, "num_workers": 0},
        "ocr": {"vocab": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        "loss": {"lambda_layout": 0.5},
        "training": {
            "save_dir": "checkpoints/test_progressive",
            "gradient_clip": 1.0,
            "lr_step_size": 5,
            "lr_gamma": 0.9,
            "early_stop_patience": 5,
        },
        "progressive_training": {
            "stage1": {"epochs": 2, "lr": 1e-4},
            "stage2": {"epochs": 3, "lr": 1e-4},
            "stage3": {"epochs": 2, "lr": 1e-5},
        },
        "tensorboard": {
            "enabled": True,
            "log_dir": "logs/test",
        },
    }

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        root_dir="data/train",
        batch_size=2,
        num_workers=0,
        image_size=(17, 31),
    )

    # Create models
    generator = Generator(num_filters=32, num_blocks=2)
    ocr = ParseqOCR(frozen=True)

    # Create logger
    logger = TensorBoardLogger(log_dir="logs/test")

    # Create trainer
    trainer = ProgressiveTrainer(
        generator=generator,
        ocr=ocr,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        device="cpu",
    )

    print("ProgressiveTrainer created successfully!")
