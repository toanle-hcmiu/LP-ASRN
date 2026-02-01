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
import torch.distributed as dist
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
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize Progressive Trainer.

        Args:
            generator: Generator network (may be wrapped in DDP)
            ocr: OCR model (may be wrapped in DDP)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            logger: TensorBoard logger
            device: Device to use
            distributed: Whether using DDP
            rank: Rank for DDP
            world_size: World size for DDP
        """
        # For DDP, models are already wrapped and moved to device
        self.generator = generator
        self.ocr = ocr
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.text_logger = None  # Optional TextLogger
        self.device = device

        # DDP settings
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.is_main = not distributed or rank == 0

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

        # Save directory (single output folder for checkpoints + logs)
        self.save_dir = Path(config.get("training", {}).get("save_dir", "outputs/run_default"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _unwrap_model(self, model):
        """Unwrap DDP model to access underlying model."""
        if self.distributed and isinstance(model, nn.parallel.DistributedDataParallel):
            return model.module
        return model

    def set_stage(self, stage: TrainingStage):
        """Set the current training stage."""
        # Synchronize all ranks before changing stage configuration
        # This ensures all ranks have the same model structure (OCR frozen/unfrozen)
        if self.distributed:
            dist.barrier()

        self.current_stage = stage
        config = self.stage_configs[stage]

        # Update OCR frozen state
        for param in self.ocr.parameters():
            param.requires_grad = not config.freeze_ocr

    def set_text_logger(self, text_logger):
        """Set the text logger for file logging."""
        self.text_logger = text_logger

    def _log(self, message: str, level: str = "info"):
        """Log message to both console and text file (main process only)."""
        if self.distributed and not self.is_main:
            return  # Only main process logs
        if self.text_logger:
            if level == "info":
                self.text_logger.info(message)
            elif level == "debug":
                self.text_logger.debug(message)
            elif level == "warning":
                self.text_logger.warning(message)
            elif level == "error":
                self.text_logger.error(message)
        else:
            print(message)

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
                    ocr_unwrapped = self._unwrap_model(self.ocr)
                    pred_texts = ocr_unwrapped.predict(sr_images)

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
        """Validate the model (only on main process for DDP)."""
        if self.distributed and not self.is_main:
            return {
                "psnr": 0.0,
                "ssim": 0.0,
                "word_acc": 0.0,
                "char_acc": 0.0,
            }

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

                # OCR predictions (ONCE per batch - greedy decoding for speed)
                ocr_unwrapped = self._unwrap_model(self.ocr)
                pred_texts = ocr_unwrapped.predict(sr_images, beam_width=5)

                # Store first batch for visualization
                if sample_batch is None and self.logger:
                    sample_batch = {
                        "lr": lr_images,
                        "sr": sr_images,
                        "hr": hr_images,
                        "gt_texts": gt_texts,
                        "pred_texts": pred_texts,
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

                # Store predictions for metrics
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
        """Save OCR model checkpoint (main process only for DDP)."""
        if self.distributed and not self.is_main:
            return

        ocr_unwrapped = self._unwrap_model(self.ocr)
        ocr_path = self.save_dir / f"ocr_stage_{stage.value}_epoch_{epoch}.pth"
        ocr_unwrapped.save(str(ocr_path))
        # Also save as default for later loading
        default_path = self.save_dir / "ocr_best.pth"
        ocr_unwrapped.save(str(default_path))

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

        # Add LR scheduler for OCR pretraining (reduces LR when accuracy plateaus)
        ocr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True if self.is_main else False,
        )

        # Use char_acc for early stopping (more granular than word_acc)
        best_char_acc = 0.0
        self.epochs_without_improvement = 0

        if self.is_main:
            print(f"\n{'='*60}")
            print(f"OCR PRETRAINING STAGE")
            print(f"{'='*60}")
            print(f"Epochs: {stage_config.epochs}")
            print(f"Learning Rate: {stage_config.lr}")
            print(f"Using CTC Loss: Yes")
            print(f"{'='*60}\n")

        # Track step counter for OCR pretraining
        ocr_step = 0

        for epoch in range(stage_config.epochs):
            total_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"OCR Pretrain Epoch {epoch+1}/{stage_config.epochs}",
                       disable=not self.is_main)
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

                # Log training loss to TensorBoard (every 10 steps)
                if self.logger and self.is_main and ocr_step % 10 == 0:
                    self.logger.log_scalar("stage0/train_loss", loss.item(), ocr_step)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.log_scalar("stage0/lr", current_lr, ocr_step)

                ocr_step += 1

            avg_loss = total_loss / len(self.train_loader)

            # Synchronize all ranks before validation
            # Ensures both ranks reach validation at the same time
            if self.distributed:
                dist.barrier()

            # Validation
            val_metrics = self.validate()

            # Synchronize all ranks after validation (critical for DDP)
            # Without this, Rank 1 continues while Rank 0 is still validating, causing NCCL timeout
            if self.distributed:
                dist.barrier()

            # Log validation metrics to TensorBoard
            if self.logger and self.is_main:
                stage_prefix = "stage0"
                self.logger.log_scalar(f"{stage_prefix}/val_loss", avg_loss, epoch)
                self.logger.log_scalar(f"{stage_prefix}/val_char_acc", val_metrics['char_acc'], epoch)
                self.logger.log_scalar(f"{stage_prefix}/val_word_acc", val_metrics['word_acc'], epoch)
                self.logger.log_scalar(f"{stage_prefix}/val_psnr", val_metrics['psnr'], epoch)
                self.logger.log_scalar(f"{stage_prefix}/val_ssim", val_metrics['ssim'], epoch)

                # Log OCR predictions visualization (every 5 epochs)
                if epoch % 5 == 0 and val_metrics.get("sample_batch"):
                    self._log_ocr_predictions(val_metrics["sample_batch"], epoch)

            # Print results (only rank 0)
            if self.is_main:
                print(f"\nOCR Pretrain Epoch {epoch+1}/{stage_config.epochs}:")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
                print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
                print(f"  Val Word Acc: {val_metrics['word_acc']:.4f}")
                print(f"  Val Char Acc: {val_metrics['char_acc']:.4f}")

            # Step LR scheduler based on validation char_acc
            ocr_scheduler.step(val_metrics['char_acc'])

            # Save best model (using char_acc for early stopping)
            if val_metrics['char_acc'] > best_char_acc:
                best_char_acc = val_metrics['char_acc']
                self.epochs_without_improvement = 0
                self.save_ocr_checkpoint(epoch, TrainingStage.PRETRAIN)
                if self.is_main:
                    print(f"  ✓ New best OCR: char_acc={val_metrics['char_acc']:.4f}, word_acc={val_metrics['word_acc']:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            patience = self.config.get("training", {}).get("early_stop_patience", 20)
            if self.epochs_without_improvement >= patience:
                if self.is_main:
                    print(f"Early stopping after {patience} epochs without improvement")
                break

        # Freeze OCR after pretraining for subsequent stages
        for param in self.ocr.parameters():
            param.requires_grad = False

        if self.is_main:
            print(f"\nOCR Pretraining complete. Best char accuracy: {best_char_acc:.4f}\n")

        return best_char_acc

    def _log_ocr_predictions(
        self,
        batch: Dict,
        epoch: int,
        max_images: int = 8,
    ):
        """
        Log OCR predictions visualization to TensorBoard.

        Shows HR images with ground truth and predicted text overlays.

        Args:
            batch: Batch containing hr_images, gt_texts, pred_texts
            epoch: Current epoch number
            max_images: Maximum number of images to log
        """
        if not self.logger:
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from torchvision.utils import make_grid

            hr_images = batch["hr"]
            gt_texts = batch["gt_texts"]
            pred_texts = batch["pred_texts"]

            # Limit number of images
            n_images = min(len(gt_texts), max_images)

            # Create figure with subplots
            fig, axes = plt.subplots(n_images, 1, figsize=(12, 2 * n_images))
            if n_images == 1:
                axes = [axes]

            for i in range(n_images):
                # Get image and convert to numpy
                img = hr_images[i].cpu()
                img = img.permute(1, 2, 0).numpy()

                # Ensure range [0, 1]
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                img = img.clip(0, 1)

                # Display image
                axes[i].imshow(img, cmap='gray' if img.shape[-1] == 1 else None)
                axes[i].axis('off')

                # Add text annotations
                gt = gt_texts[i] if i < len(gt_texts) else ""
                pred = pred_texts[i] if i < len(pred_texts) else ""

                # Color code: green if correct, red if wrong
                color = 'green' if gt == pred else 'red'

                title = f"GT: {gt} | Pred: {pred}"
                axes[i].set_title(title, color=color, fontsize=12, fontweight='bold')

            plt.tight_layout()
            self.logger.log_figure(f"stage0/ocr_predictions_epoch_{epoch}", fig, epoch)
            plt.close(fig)

        except Exception as e:
            if self.is_main:
                print(f"Warning: Could not log OCR predictions: {e}")

    def _log_sr_ocr_predictions(
        self,
        batch: Dict,
        epoch: int,
        max_images: int = 4,
    ):
        """
        Log SR+OCR predictions visualization to TensorBoard.

        Shows LR -> SR -> HR progression with OCR predictions.

        Args:
            batch: Batch containing lr, sr, hr, gt_texts, pred_texts
            epoch: Current epoch number
            max_images: Maximum number of images to log
        """
        if not self.logger:
            return

        try:
            import matplotlib.pyplot as plt

            lr_images = batch["lr"]
            sr_images = batch["sr"]
            hr_images = batch["hr"]
            gt_texts = batch["gt_texts"]
            pred_texts = batch["pred_texts"]

            # Limit number of images
            n_images = min(len(gt_texts), max_images)

            # Create figure: each row shows LR -> SR -> HR
            fig, axes = plt.subplots(n_images, 3, figsize=(15, 4 * n_images))
            if n_images == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_images):
                # Get images and convert to numpy
                lr = lr_images[i].cpu().permute(1, 2, 0).numpy()
                sr = sr_images[i].cpu().permute(1, 2, 0).numpy()
                hr = hr_images[i].cpu().permute(1, 2, 0).numpy()

                # Ensure range [0, 1]
                for img in [lr, sr, hr]:
                    if img.min() < 0:
                        img = (img + 1.0) / 2.0
                    img.clip(0, 1, out=img)

                # Display LR
                axes[i, 0].imshow(lr, cmap='gray' if lr.shape[-1] == 1 else None)
                axes[i, 0].set_title("LR Input", fontsize=10)
                axes[i, 0].axis('off')

                # Display SR with OCR prediction
                axes[i, 1].imshow(sr, cmap='gray' if sr.shape[-1] == 1 else None)
                pred = pred_texts[i] if i < len(pred_texts) else ""
                axes[i, 1].set_title(f"SR Output\nOCR: {pred}", fontsize=10, color='blue')
                axes[i, 1].axis('off')

                # Display HR with ground truth
                axes[i, 2].imshow(hr, cmap='gray' if hr.shape[-1] == 1 else None)
                gt = gt_texts[i] if i < len(gt_texts) else ""
                color = 'green' if gt == pred else 'red'
                axes[i, 2].set_title(f"HR Ground Truth\nGT: {gt}", fontsize=10, color=color)
                axes[i, 2].axis('off')

            stage_to_name = {
                TrainingStage.WARMUP: "Stage1_Warmup",
                TrainingStage.LCOFL: "Stage2_LCOFL",
                TrainingStage.FINETUNE: "Stage3_Finetune",
            }
            stage_name = stage_to_name.get(self.current_stage, "SR")

            plt.suptitle(f"{stage_name} - SR+OCR Results (Epoch {epoch})", fontsize=14, fontweight='bold')
            plt.tight_layout()
            self.logger.log_figure(f"{stage_name.lower()}/sr_ocr_predictions_epoch_{epoch}", fig, epoch)
            plt.close(fig)

        except Exception as e:
            if self.is_main:
                print(f"Warning: Could not log SR+OCR predictions: {e}")

    def log_to_tensorboard(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        epoch: int,
    ):
        """Log metrics to TensorBoard with stage-prefixed names."""
        if not self.logger:
            return

        # Get stage prefix for organized logging
        stage_to_prefix = {
            TrainingStage.PRETRAIN: "stage0_ocr",
            TrainingStage.WARMUP: "stage1_warmup",
            TrainingStage.LCOFL: "stage2_lcofl",
            TrainingStage.FINETUNE: "stage3_finetune",
        }
        stage_prefix = stage_to_prefix.get(self.current_stage, f"stage_{self.current_stage.value}")

        # Log training metrics with stage prefix
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.log_scalar(f"{stage_prefix}/train_{key}", value, epoch)

        # Log validation metrics with stage prefix
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.log_scalar(f"{stage_prefix}/val_{key}", value, epoch)

        # Log learning rate
        if hasattr(self, 'optimizer') and self.optimizer:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_scalar(f"{stage_prefix}/learning_rate", current_lr, epoch)

        # Log model weights (less frequently)
        if epoch % 10 == 0:
            self.logger.log_model_weights(self.generator, epoch, prefix=stage_prefix)

        # Log comparison images for SR stages (not OCR pretraining)
        if val_metrics.get("sample_batch") and epoch % 5 == 0 and self.current_stage != TrainingStage.PRETRAIN:
            batch = val_metrics["sample_batch"]
            try:
                grid = create_comparison_grid(
                    batch["lr"],
                    batch["sr"],
                    batch["hr"],
                    batch["gt_texts"],
                    batch["pred_texts"],
                    max_images=8,
                )
                self.logger.log_image_grid(f"{stage_prefix}/comparison_epoch_{epoch}", grid, epoch)
            except Exception as e:
                if self.is_main:
                    print(f"Warning: Could not log comparison images: {e}")

        # Log confusion matrix for LCOFL and finetune stages
        if epoch % 5 == 0 and self.current_stage in [TrainingStage.LCOFL, TrainingStage.FINETUNE] and self.is_main:
            self.confusion_tracker.update(val_metrics["pred_texts"], val_metrics["gt_texts"])
            try:
                labels = list(self.config["ocr"]["vocab"])
                self.logger.log_confusion_matrix(
                    self.confusion_tracker.confusion_matrix,
                    labels,
                    epoch,
                    tag=f"{stage_prefix}/confusion_matrix",
                )
            except Exception as e:
                if self.is_main:
                    print(f"Warning: Could not log confusion matrix: {e}")

        # Log OCR predictions for LCOFL and finetune stages
        if epoch % 5 == 0 and self.current_stage in [TrainingStage.LCOFL, TrainingStage.FINETUNE] and self.is_main:
            if val_metrics.get("sample_batch"):
                self._log_sr_ocr_predictions(val_metrics["sample_batch"], epoch)

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

            # Synchronize all ranks before validation
            # Ensures both ranks reach validation at the same time
            if self.distributed:
                dist.barrier()

            # Validate
            val_metrics = self.validate()

            # Synchronize all ranks after validation (critical for DDP)
            # Without this, Rank 1 continues while Rank 0 is still validating, causing NCCL timeout
            if self.distributed:
                dist.barrier()

            # Update confusion if needed (only main rank has pred_texts)
            if config.update_confusion and self.is_main:
                self.confusion_tracker.update(val_metrics["pred_texts"], val_metrics["gt_texts"])
                self.lcofl_loss.update_weights(self.confusion_tracker.confusion_matrix)

            # Log to TensorBoard
            self.log_to_tensorboard(train_metrics, val_metrics, epoch)

            # Print results (only rank 0)
            if self.is_main:
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
                if self.is_main:
                    print(f"  ✓ New best model: word_acc={val_metrics['word_acc']:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping
            patience = self.config.get("training", {}).get("early_stop_patience", 20)
            if self.epochs_without_improvement >= patience:
                if self.is_main:
                    print(f"Early stopping after {patience} epochs without improvement")
                break

        self.global_epoch = start_epoch + config.epochs

        return self.best_word_acc

    def save_checkpoint(self, epoch: int, stage: TrainingStage):
        """Save a checkpoint (main process only for DDP)."""
        if self.distributed and not self.is_main:
            return

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

        if self.distributed:
            # Synchronize all processes after loading checkpoint
            dist.barrier()

    def train_full_progressive(self) -> Dict[str, Any]:
        """Train all stages sequentially."""
        results = {}

        if self.is_main:
            print("\n" + "=" * 60)
            print("PROGRESSIVE TRAINING START")
            print("=" * 60 + "\n")

        # Stage 0: OCR Pretraining
        if self.is_main:
            print("\nStage 0: OCR Pretraining")
            print("Purpose: Train OCR model on license plate data")
            print("This ensures OCR can provide meaningful guidance to the generator")
        best_acc_stage0 = self.train_pretrain_stage(
            self.stage_configs[TrainingStage.PRETRAIN]
        )
        results["stage0_best_acc"] = best_acc_stage0

        # Synchronize all ranks between stages (critical for DDP)
        # Ensures all ranks complete stage 0 before stage 1 starts
        if self.distributed:
            dist.barrier()

        # Stage 1: Warm-up
        if self.is_main:
            print("\nStage 1: Warm-up (L1 loss only)")
            print("Purpose: Stabilize network with simple reconstruction loss")
        best_acc_stage1 = self.train_stage(TrainingStage.WARMUP)
        results["stage1_best_acc"] = best_acc_stage1

        # Synchronize all ranks between stages
        if self.distributed:
            dist.barrier()

        # Stage 2: LCOFL
        if self.is_main:
            print("\nStage 2: LCOFL Training (Character-driven loss)")
            print("Purpose: Optimize for character recognition with frozen OCR")
        best_acc_stage2 = self.train_stage(TrainingStage.LCOFL)
        results["stage2_best_acc"] = best_acc_stage2

        # Synchronize all ranks between stages
        if self.distributed:
            dist.barrier()

        # Stage 3: Fine-tuning
        if self.is_main:
            print("\nStage 3: Fine-tuning (Joint optimization)")
            print("Purpose: Refine with unfrozen OCR for joint training")
        best_acc_stage3 = self.train_stage(TrainingStage.FINETUNE)
        results["stage3_best_acc"] = best_acc_stage3

        # Save final results
        results["final_best_acc"] = self.best_word_acc

        if self.is_main:
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
            "save_dir": "outputs/test_progressive",
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
            "log_dir": "outputs/test_progressive/logs",
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
    logger = TensorBoardLogger(log_dir="outputs/test_progressive/logs")

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
