"""
Progressive Training for LP-ASRN

Implements multi-stage training approach:
1. Stage 1: Warm-up with L1 loss only
2. Stage 2: LCOFL training with frozen OCR
3. Stage 3: Fine-tuning with unfrozen OCR
"""

import os
import json
import time
import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm

from src.models.generator import Generator
from src.ocr.ocr_model import OCRModel
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
    HARD_MINING = "hard_mining"  # Stage 4: OCR-driven hard example mining


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
        ocr: OCRModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: Optional[TensorBoardLogger] = None,
        device: str = "cuda",
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        train_sampler: Optional["DistributedSampler"] = None,
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
            train_sampler: DistributedSampler for training data (for set_epoch calls)
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
        self.train_sampler = train_sampler

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
            TrainingStage.HARD_MINING: StageConfig(
                name="hard_mining",
                epochs=self.progressive_config.get("stage4", {}).get("epochs", 20),
                lr=self.progressive_config.get("stage4", {}).get("lr", 5e-6),
                loss_components=["l1", "lcofl", "embedding"],
                freeze_ocr=True,
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

        # DDP synchronization tracking
        self._last_sync_step = 0
        self._sync_interval = 50  # Sync check every 50 steps

    def _safe_barrier(self, timeout_seconds: int = 300, description: str = "barrier"):
        """
        Execute a barrier with appropriate timeout handling.
        
        Note: NCCL backend doesn't support monitored_barrier, so we use regular
        barrier for NCCL. The timeout is handled at init_process_group level.
        For GLOO backend, we use monitored_barrier with explicit timeout.
        
        Args:
            timeout_seconds: Max seconds to wait for barrier (GLOO only)
            description: Description for logging
        """
        if not self.distributed:
            return True
        
        try:
            # Get current backend
            backend = dist.get_backend()
            
            # NCCL doesn't support monitored_barrier, use regular barrier
            # The timeout is set at init_process_group level (60 min)
            if backend == "nccl":
                dist.barrier()
            elif hasattr(dist, 'monitored_barrier'):
                # GLOO supports monitored_barrier with explicit timeout
                dist.monitored_barrier(timeout=datetime.timedelta(seconds=timeout_seconds))
            else:
                dist.barrier()
            return True
        except Exception as e:
            if self.is_main:
                self._log(f"WARNING: Barrier '{description}' failed: {e}", "warning")
            return False

    def _sync_all_ranks(self, tensor_to_sync: torch.Tensor = None):
        """
        Synchronize a tensor across all ranks (for consistent state).
        
        Args:
            tensor_to_sync: Optional tensor to sync (broadcasts from rank 0)
        """
        if not self.distributed:
            return tensor_to_sync
            
        if tensor_to_sync is not None:
            dist.broadcast(tensor_to_sync, src=0)
        return tensor_to_sync

    def _ddp_sync_check(self, step: int, force: bool = False):
        """
        Periodic DDP sync check to prevent rank divergence.
        
        This is a lightweight check that ensures all ranks are at the same step.
        
        Args:
            step: Current training step
            force: Force sync even if not at interval
        """
        if not self.distributed:
            return
            
        # Only check at intervals to minimize overhead
        if not force and (step - self._last_sync_step) < self._sync_interval:
            return
            
        self._last_sync_step = step
        
        # Create a tensor with current step on each rank
        step_tensor = torch.tensor([step], device=self.device, dtype=torch.long)
        
        # Gather all steps to rank 0
        if self.world_size > 1:
            gathered = [torch.zeros(1, device=self.device, dtype=torch.long) for _ in range(self.world_size)]
            dist.all_gather(gathered, step_tensor)
            
            # Check for divergence (only main rank logs)
            if self.is_main:
                steps = [t.item() for t in gathered]
                max_diff = max(steps) - min(steps)
                if max_diff > 10:  # Significant divergence
                    self._log(f"WARNING: Rank step divergence detected! Steps: {steps}", "warning")

    def _unwrap_model(self, model):
        """Unwrap DDP model to access underlying model."""
        if self.distributed and isinstance(model, nn.parallel.DistributedDataParallel):
            return model.module
        return model

    def _apply_ocr_augmentation(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply aggressive data augmentation for OCR pretraining.
        
        Includes: Gaussian blur, noise, brightness/contrast, random erasing,
        affine transforms, and mixup.
        
        Args:
            images: Input images (B, C, H, W) in range [-1, 1] or [0, 1]
            
        Returns:
            Augmented images
        """
        import random
        import torch.nn.functional as F
        
        B, C, H, W = images.shape
        augmented = images.clone()
        
        for i in range(B):
            # 70% chance to apply augmentation (more aggressive)
            if random.random() > 0.7:
                continue
                
            img = augmented[i:i+1]
            
            # 1. Gaussian blur (40% chance)
            if random.random() < 0.4:
                kernel_size = random.choice([3, 5])
                padding = kernel_size // 2
                img = F.avg_pool2d(img, kernel_size, stride=1, padding=padding)
            
            # 2. Gaussian noise (50% chance)
            if random.random() < 0.5:
                noise_std = random.uniform(0.03, 0.12)
                noise = torch.randn_like(img) * noise_std
                img = img + noise
            
            # 3. Brightness/contrast jitter (50% chance)
            if random.random() < 0.5:
                brightness = random.uniform(-0.2, 0.2)
                contrast = random.uniform(0.7, 1.3)
                img = img * contrast + brightness
            
            # 4. Random erasing (30% chance) - simulates occlusion
            if random.random() < 0.3:
                erase_h = random.randint(2, max(3, H // 5))
                erase_w = random.randint(4, max(5, W // 3))
                erase_y = random.randint(0, H - erase_h)
                erase_x = random.randint(0, W - erase_w)
                # Use random value instead of 0 for more variety
                erase_val = random.uniform(-1, 1)
                img[:, :, erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = erase_val
            
            # 5. Affine transform (25% chance) - rotation and shear
            if random.random() < 0.25:
                angle = random.uniform(-5, 5)  # Small rotation
                shear = random.uniform(-0.05, 0.05)
                # Create affine matrix
                theta = torch.tensor([
                    [1, shear, 0],
                    [0, 1, 0]
                ], dtype=img.dtype, device=img.device).unsqueeze(0)
                grid = F.affine_grid(theta, img.size(), align_corners=False)
                img = F.grid_sample(img, grid, align_corners=False, padding_mode='border')
            
            # Clamp to valid range
            augmented[i:i+1] = torch.clamp(img, -1.0, 1.0)
        
        # 6. Mixup augmentation (20% of batches)
        if random.random() < 0.2 and B >= 2:
            # Random shuffle indices
            perm = torch.randperm(B, device=images.device)
            alpha = random.uniform(0.1, 0.3)
            augmented = (1 - alpha) * augmented + alpha * augmented[perm]
        
        return augmented

    def set_stage(self, stage: TrainingStage):
        """Set the current training stage."""
        # Synchronize all ranks before changing stage configuration
        # This ensures all ranks have the same model structure (OCR frozen/unfrozen)
        self._safe_barrier(description="set_stage")

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

    def _check_gradients(self) -> bool:
        """Check if gradients contain NaN or Inf values (lightweight check)."""
        # Only check a subset of parameters to avoid overhead
        # Check first few layers where NaN typically propagates from
        for i, param in enumerate(self.generator.parameters()):
            if param.grad is not None and i < 5:  # Only check first 5 parameter layers
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return False
        return True

    def train_epoch(self, stage_config: StageConfig) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_lcofl = 0.0

        pred_texts_all = []
        gt_texts_all = []

        pbar = tqdm(self.train_loader, desc=f"Stage {self.current_stage.value} Epoch {self.global_epoch}/{stage_config.epochs}")
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
                # Get OCR logits (single forward pass — predict() is not needed)
                with torch.no_grad():
                    pred_logits = self.ocr(sr_images, return_logits=True)

                # Decode texts from logits for confusion tracking (no extra forward pass)
                ocr_unwrapped = self._unwrap_model(self.ocr)
                with torch.no_grad():
                    if hasattr(ocr_unwrapped.model, 'use_ctc') and ocr_unwrapped.model.use_ctc:
                        decoded_lists = ocr_unwrapped.model.ctc_decode_greedy(pred_logits)
                        pred_texts = []
                        for indices in decoded_lists:
                            text = ""
                            for idx in indices:
                                if 0 <= idx < ocr_unwrapped.blank_idx:
                                    char = ocr_unwrapped.tokenizer.idx_to_char.get(idx, "")
                                    if char in ocr_unwrapped.tokenizer.vocab:
                                        text += char
                            pred_texts.append(text)
                    else:
                        pred_indices = pred_logits.argmax(dim=-1)
                        pred_texts = ocr_unwrapped.tokenizer.decode_batch(pred_indices)

                # Compute LCOFL (layout penalty now uses logits directly)
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

            # Periodic GPU memory cleanup
            if self.global_step % 100 == 0:
                torch.cuda.empty_cache()

        # DDP sync: Ensure all ranks complete training epoch before returning
        self._safe_barrier(description="train_epoch_end")

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

    def validate(self, beam_width: int = 5) -> Dict[str, float]:
        """
        Validate the model (only on main process for DDP).

        Args:
            beam_width: Beam width for OCR decoding (higher = more accurate).

        Returns:
            Dictionary with validation metrics.
        """
        if self.distributed and not self.is_main:
            return {
                "psnr": 0.0,
                "ssim": 0.0,
                "word_acc": 0.0,
                "char_acc": 0.0,
            }

        # Handle None val_loader (non-main ranks in DDP)
        if self.val_loader is None:
            return {
                "psnr": 0.0,
                "ssim": 0.0,
                "word_acc": 0.0,
                "char_acc": 0.0,
                "pred_texts": [],
                "gt_texts": [],
            }

        self.generator.eval()
        self.ocr.eval()  # Critical: ensure batch norm uses eval statistics

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

                # OCR predictions (ONCE per batch - use configurable beam width)
                ocr_unwrapped = self._unwrap_model(self.ocr)
                pred_texts = ocr_unwrapped.predict(sr_images, beam_width=beam_width)

                # Store first batch for visualization (move to CPU to save GPU memory)
                if sample_batch is None and self.logger:
                    sample_batch = {
                        "lr": lr_images.cpu(),
                        "sr": sr_images.detach().cpu(),
                        "hr": hr_images.cpu(),
                        "gt_texts": gt_texts,
                        "pred_texts": pred_texts,
                    }

                # Compute PSNR/SSIM (vectorized - much faster!)
                # MSE per image in batch
                mse_per_img = torch.mean((sr_images - hr_images) ** 2, dim=(1, 2, 3))
                psnr_per_img = 20 * torch.log10(2.0 / torch.sqrt(mse_per_img + 1e-10))
                total_psnr += psnr_per_img.sum().item()

                # SSIM (simplified) - batch computation
                mae_per_img = torch.mean(torch.abs(sr_images - hr_images), dim=(1, 2, 3))
                ssim_per_img = 1.0 - (mae_per_img / 2.0)
                total_ssim += ssim_per_img.sum().item()

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

        # Restore training mode
        self.generator.train()
        self.ocr.train()

        return {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "word_acc": word_acc,
            "char_acc": char_acc,
            "pred_texts": pred_texts_all,
            "gt_texts": gt_texts_all,
            "sample_batch": sample_batch,
        }

    def validate_ocr_only(self, beam_width: int = 5, max_batches: int = None) -> Dict[str, float]:
        """
        Validate OCR model on HR images (for OCR pretraining stage).

        Unlike validate() which runs SR generator, this validates OCR directly
        on HR images since OCR pretraining doesn't use the generator.

        Args:
            beam_width: Beam width for OCR decoding
            max_batches: Limit validation to this many batches (None = all). 
                         Use to prevent OOM during long validation runs.

        Returns:
            Dictionary with word_acc and char_acc (no PSNR/SSIM for OCR-only validation)
        """
        if self.distributed and not self.is_main:
            return {"word_acc": 0.0, "char_acc": 0.0}

        if self.val_loader is None:
            return {"word_acc": 0.0, "char_acc": 0.0}

        self.ocr.eval()

        word_correct = 0
        char_correct = 0
        char_total = 0
        total_samples = 0

        # Limit validation batches to prevent OOM
        total_batches = len(self.val_loader)
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="OCR Validation", total=total_batches)):
                # Stop if we've reached max_batches
                if max_batches is not None and batch_idx >= max_batches:
                    break

                hr_images = batch["hr"].to(self.device)
                gt_texts = batch["plate_text"]

                # OCR predictions on HR images (not SR!)
                ocr_unwrapped = self._unwrap_model(self.ocr)
                pred_texts = ocr_unwrapped.predict(hr_images, beam_width=beam_width)

                # Aggressive GPU memory cleanup every 10 batches to prevent OOM
                # (Reduced from 20 to 10 for memory-constrained scenarios)
                if batch_idx % 10 == 0 and batch_idx > 0:
                    del hr_images  # Explicitly delete tensors
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                for pred, gt in zip(pred_texts, gt_texts):
                    total_samples += 1
                    if pred == gt:
                        word_correct += 1

                    # Fixed char accuracy: count ALL GT chars, not just matched positions
                    for i, g_char in enumerate(gt):
                        char_total += 1
                        if i < len(pred) and pred[i] == g_char:
                            char_correct += 1
                    # Extra predicted chars count as errors
                    if len(pred) > len(gt):
                        char_total += len(pred) - len(gt)

        # Final memory cleanup
        torch.cuda.empty_cache()
        self.ocr.train()  # Restore training mode

        return {
            "word_acc": word_correct / total_samples if total_samples > 0 else 0,
            "char_acc": char_correct / char_total if char_total > 0 else 0,
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
        # Unfreeze OCR for training
        ocr_unwrapped = self._unwrap_model(self.ocr)
        for param in ocr_unwrapped.parameters():
            param.requires_grad = True

        # In DDP mode, wrap OCR with DDP if not already wrapped
        # (OCR was initially frozen, so may not have been wrapped)
        if self.distributed and not isinstance(self.ocr, nn.parallel.DistributedDataParallel):
            self.ocr = nn.parallel.DistributedDataParallel(
                self.ocr.to(self.device),
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,  # All params used in pretraining
            )
            if self.is_main:
                print("OCR wrapped with DDP for pretraining")

        self.ocr.train()

        # Create optimizer for OCR only (AdamW with weight decay for better generalization)
        # Fixed: Use configured LR directly (removed 0.1 multiplier that was too conservative)
        self.optimizer = optim.AdamW(
            self.ocr.parameters(),
            lr=stage_config.lr,  # Use configured LR directly (0.001)
            weight_decay=0.05,  # Increased from 0.01 for better regularization
        )

        # Add warmup scheduler for first 5 epochs to prevent early instability
        warmup_epochs = 5
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
        )

        # Main scheduler: ReduceLROnPlateau for stable convergence
        # Reduces LR only when char_acc stops improving (no sudden LR spikes)
        ocr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',      # Maximize char_acc
            factor=0.5,      # Halve LR when stuck
            patience=10,     # Wait 10 epochs before reducing
            min_lr=1e-6,
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

        # Get validation config (outside loop for efficiency)
        val_interval = self.config.get("training", {}).get("val_interval", 30)
        val_beam_width = self.config.get("training", {}).get("val_beam_width", 5)

        for epoch in range(stage_config.epochs):
            # Set epoch for proper shuffling in DDP
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            total_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"OCR Pretrain Epoch {epoch+1}/{stage_config.epochs}",
                       disable=not self.is_main)
            for batch in pbar:
                hr_images = batch["hr"].to(self.device)
                gt_texts = batch["plate_text"]

                # Apply data augmentation for OCR pretraining (prevents overfitting)
                if self.ocr.training:
                    hr_images = self._apply_ocr_augmentation(hr_images)

                # Forward pass
                logits = self.ocr(hr_images, return_logits=True)

                # Compute loss (uses CTC for SimpleCRNN)
                ocr_unwrapped = self._unwrap_model(self.ocr)
                label_smoothing = self.config.get("ocr", {}).get("label_smoothing", 0.1)
                loss = ocr_unwrapped.compute_loss(logits, gt_texts, device=self.device,
                                                   label_smoothing=label_smoothing)

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

            # Only validate every val_interval epochs (or first/last epoch)
            should_validate = (epoch % val_interval == 0) or (epoch == 0) or (epoch == stage_config.epochs - 1)

            # DDP sync: all ranks pause before validation
            self._safe_barrier(timeout_seconds=600, description="pretrain_pre_validation")

            if should_validate:
                # Limit validation batches to prevent OOM (configurable, default 100 batches)
                max_val_batches = self.config.get("training", {}).get("max_val_batches", 100)
                val_metrics = self.validate_ocr_only(beam_width=val_beam_width, max_batches=max_val_batches)

                # Log validation metrics to TensorBoard
                if self.logger and self.is_main:
                    stage_prefix = "stage0"
                    self.logger.log_scalar(f"{stage_prefix}/val_loss", avg_loss, epoch)
                    self.logger.log_scalar(f"{stage_prefix}/val_char_acc", val_metrics['char_acc'], epoch)
                    self.logger.log_scalar(f"{stage_prefix}/val_word_acc", val_metrics['word_acc'], epoch)
                    # No PSNR/SSIM for OCR-only validation (not applicable without SR generator)

                # Print results
                if self.is_main:
                    print(f"\nOCR Pretrain Epoch {epoch+1}/{stage_config.epochs}:")
                    print(f"  Train Loss: {avg_loss:.4f}")
                    print(f"  Val Word Acc: {val_metrics['word_acc']:.4f}")
                    print(f"  Val Char Acc: {val_metrics['char_acc']:.4f}")

                # Save best model (using char_acc for early stopping)
                if val_metrics['char_acc'] > best_char_acc:
                    best_char_acc = val_metrics['char_acc']
                    self.epochs_without_improvement = 0
                    self.save_ocr_checkpoint(epoch, TrainingStage.PRETRAIN)
                    if self.is_main:
                        print(f"  ✓ New best OCR: char_acc={val_metrics['char_acc']:.4f}, word_acc={val_metrics['word_acc']:.4f}")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping (scale patience by validation interval)
                base_patience = self.config.get("training", {}).get("early_stop_patience", 20)
                patience = base_patience * max(1, val_interval // 10)
                if self.epochs_without_improvement >= patience:
                    if self.is_main:
                        print(f"Early stopping after {patience} validations without improvement")
                    break

            else:
                # Skip validation - just print training progress
                if self.is_main:
                    print(f"\nOCR Pretrain Epoch {epoch+1}/{stage_config.epochs}: Train Loss: {avg_loss:.4f}")

            # DDP sync: all ranks wait for validation to complete
            self._safe_barrier(timeout_seconds=900, description="pretrain_post_validation")

            # Step warmup scheduler for first 5 epochs
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            elif should_validate:
                # ReduceLROnPlateau needs the metric value
                ocr_scheduler.step(val_metrics.get('char_acc', 0.0))

        # CRITICAL: Sync all ranks before modifying model state
        self._safe_barrier(timeout_seconds=300, description="pretrain_before_freeze")

        # Freeze OCR after pretraining for subsequent stages
        for param in self.ocr.parameters():
            param.requires_grad = False

        # CRITICAL: Sync all ranks after pretrain stage completes
        self._safe_barrier(timeout_seconds=300, description="pretrain_complete")

        if self.is_main:
            print(f"\nOCR Pretraining complete. Best char accuracy: {best_char_acc:.4f}\n")

        return best_char_acc

    def train_hard_mining_stage(self, stage_config: StageConfig) -> float:
        """
        Train with hard example mining (Stage 4).

        This stage focuses on samples that OCR struggles with, using
        weighted sampling and per-character loss adjustments.

        Args:
            stage_config: Configuration for this training stage

        Returns:
            Best word accuracy achieved during hard mining
        """
        from src.training.hard_example_miner import HardExampleMiner

        # Initialize hard example miner
        self.hard_example_miner = HardExampleMiner(
            dataset_size=len(self.train_loader.dataset),
            alpha=self.config.get("progressive_training", {}).get("stage4", {})
                .get("hard_mining", {}).get("difficulty_alpha", 2.0),
        )

        # Get Stage 4 configuration
        stage4_config = self.config.get("progressive_training", {}).get("stage4", {})
        reweight_interval = stage4_config.get("hard_mining", {}).get("reweight_interval", 5)
        min_samples_seen = stage4_config.get("hard_mining", {}).get("min_samples_seen", 100)

        if self.is_main:
            print(f"\n{'='*60}")
            print(f"HARD EXAMPLE MINING STAGE (Stage 4)")
            print(f"{'='*60}")
            print(f"Epochs: {stage_config.epochs}")
            print(f"Learning Rate: {stage_config.lr}")
            print(f"Loss Components: {stage_config.loss_components}")
            print(f"{'='*60}\n")

        # Create optimizer
        params = list(self.generator.parameters())
        if not stage_config.freeze_ocr:
            params.extend(list(self.ocr.parameters()))

        self.optimizer = optim.Adam(params, lr=stage_config.lr)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config.get("training", {}).get("lr_step_size", 5),
            gamma=self.config.get("training", {}).get("lr_gamma", 0.9),
        )

        # Get validation config
        val_interval = self.config.get("training", {}).get("val_interval", 10)
        val_beam_width = self.config.get("training", {}).get("val_beam_width", 5)

        best_word_acc = 0.0
        self.epochs_without_improvement = 0

        # Track batch indices for hard example mining
        # In practice, you'd need to modify the dataset to provide indices
        batch_indices = torch.arange(len(self.train_loader.dataset))

        for epoch in range(stage_config.epochs):
            # Set epoch for proper shuffling in DDP
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.generator.train()
            if stage_config.freeze_ocr:
                self.ocr.eval()
            else:
                self.ocr.train()

            total_loss = 0.0
            total_l1 = 0.0
            total_lcofl = 0.0
            total_embed = 0.0

            pred_texts_all = []
            gt_texts_all = []

            pbar = tqdm(self.train_loader, desc=f"Hard Mining Epoch {epoch+1}/{stage_config.epochs}",
                       disable=not self.is_main)

            for batch_idx, batch in enumerate(pbar):
                lr_images = batch["lr"].to(self.device)
                hr_images = batch["hr"].to(self.device)
                gt_texts = batch["plate_text"]

                # Forward pass
                sr_images = self.generator(lr_images)

                # Compute L1 loss
                l1 = self.l1_loss(sr_images, hr_images)
                loss = l1

                # Get OCR predictions
                with torch.no_grad():
                    pred_logits = self.ocr(sr_images, return_logits=True)
                    ocr_unwrapped = self._unwrap_model(self.ocr)
                    pred_texts = ocr_unwrapped.predict(sr_images)

                # Compute LCOFL
                lcofl, lcofl_info = self.lcofl_loss(
                    sr_images, hr_images, pred_logits, gt_texts, pred_texts
                )
                lambda_lcofl = self.config.get("loss", {}).get("lambda_lcofl", 1.0)
                loss = loss + lambda_lcofl * lcofl
                total_lcofl += lcofl.item()

                # Compute embedding loss if enabled
                if "embedding" in stage_config.loss_components:
                    sr_emb, hr_emb = self.lcofl_loss.get_embeddings(sr_images, hr_images)
                    if sr_emb is not None and hr_emb is not None:
                        embed_loss, _ = self.lcofl_loss.embedding_loss_fn(sr_emb, hr_emb)
                        # Get current embedding weight
                        lambda_embed = self.config.get("loss", {}).get("lambda_embed", 0.3)
                        loss = loss + lambda_embed * embed_loss
                        total_embed += embed_loss.item()

                total_loss += loss.item()
                total_l1 += l1.item()

                # Compute character accuracy per sample
                char_accs = self._compute_char_accuracy(pred_texts, gt_texts)

                # Update hard example miner
                # Note: In a real implementation, you'd track which samples
                # correspond to which batch indices
                self.hard_example_miner.update(
                    torch.arange(len(pred_texts))[:len(char_accs)],
                    char_accs[:len(char_accs)],
                )

                pred_texts_all.extend(pred_texts)
                gt_texts_all.extend(gt_texts)

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
                self.global_step += 1

                pbar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(self.train_loader)
            avg_l1 = total_l1 / len(self.train_loader)
            avg_lcofl = total_lcofl / len(self.train_loader)
            avg_embed = total_embed / len(self.train_loader) if "embedding" in stage_config.loss_components else 0

            # Validate
            should_validate = (epoch % val_interval == 0) or (epoch == 0) or (epoch == stage_config.epochs - 1)

            if should_validate:
                self._safe_barrier(timeout_seconds=600, description="hard_mining_validation")

                val_metrics = self.validate(beam_width=val_beam_width)

                # Log to TensorBoard
                train_metrics = {
                    "loss": avg_loss,
                    "l1": avg_l1,
                    "lcofl": avg_lcofl,
                    "embed": avg_embed,
                }
                self.log_to_tensorboard(train_metrics, val_metrics, epoch)

                if self.is_main:
                    print(f"\nHard Mining Epoch {epoch+1}/{stage_config.epochs}:")
                    print(f"  Train Loss: {avg_loss:.4f}")
                    print(f"  Val Word Acc: {val_metrics['word_acc']:.4f}")
                    print(f"  Val Char Acc: {val_metrics['char_acc']:.4f}")

                    # Check for best model
                    if val_metrics['word_acc'] > self.best_word_acc:
                        self.best_word_acc = val_metrics['word_acc']
                        self.epochs_without_improvement = 0
                        self.save_checkpoint(epoch, TrainingStage.HARD_MINING)
                    else:
                        self.epochs_without_improvement += 1

            # Update confusion if needed
            if stage_config.update_confusion and self.is_main:
                self.confusion_tracker.update(pred_texts_all, gt_texts_all)
                self.lcofl_loss.update_weights(self.confusion_tracker.confusion_matrix)

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping
            patience = self.config.get("training", {}).get("early_stop_patience", 20)
            if self.epochs_without_improvement >= patience:
                if self.is_main:
                    print(f"Early stopping after {patience} epochs without improvement")
                break

        best_acc = self.best_word_acc

        if self.is_main:
            print(f"\nHard Mining Stage complete. Best word accuracy: {best_acc:.4f}\n")

        return best_acc

    def _compute_char_accuracy(
        self,
        pred_texts: List[str],
        gt_texts: List[str],
    ) -> torch.Tensor:
        """
        Compute character-level accuracy for each sample.

        Args:
            pred_texts: List of predicted texts
            gt_texts: List of ground truth texts

        Returns:
            Tensor of character accuracies
        """
        char_accs = []
        for pred, gt in zip(pred_texts, gt_texts):
            min_len = min(len(pred), len(gt))
            correct = sum(1 for i in range(min_len) if pred[i] == gt[i])
            acc = correct / max(len(gt), 1)
            char_accs.append(acc)

        return torch.tensor(char_accs)

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
                TrainingStage.HARD_MINING: "Stage4_HardMining",
            }
            stage_name = stage_to_name.get(self.current_stage, "SR")

            plt.suptitle(f"{stage_name} - SR+OCR Results (Epoch {epoch})", fontsize=14, fontweight='bold')
            plt.tight_layout()
            # Use consistent tag name (step parameter tracks progression)
            self.logger.log_figure(f"{stage_name.lower()}/sr_ocr_predictions", fig, epoch)
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
            TrainingStage.HARD_MINING: "stage4_hard_mining",
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

        # Log model weights (very infrequently - creates many histogram cards)
        # Only log at epoch 0 and every 50 epochs to reduce TensorBoard clutter
        if epoch == 0 or epoch % 50 == 0:
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
                # Live comparison card (uses slider to see progression)
                self.logger.log_image_grid(f"{stage_prefix}/comparison", grid, epoch)
                
                # Snapshots for easy side-by-side comparison (every 5 epochs)
                self.logger.log_image_grid(f"{stage_prefix}/epoch_{epoch:03d}", grid, epoch)
                
                del grid  # Free memory immediately
            except Exception as e:
                if self.is_main:
                    print(f"Warning: Could not log comparison images: {e}")
            finally:
                # Clean up to prevent memory accumulation over long runs
                torch.cuda.empty_cache()

        # Log confusion matrix for LCOFL and finetune stages
        if epoch % 5 == 0 and self.current_stage in [TrainingStage.LCOFL, TrainingStage.FINETUNE] and self.is_main:
            self.confusion_tracker.update(val_metrics["pred_texts"], val_metrics["gt_texts"])
            try:
                labels = list(self.config["ocr"]["vocab"])
                # Use consistent tag name to avoid creating new card every epoch
                self.logger.log_confusion_matrix(
                    self.confusion_tracker.confusion_matrix,
                    labels,
                    epoch,
                    tag=f"{stage_prefix}/confusion",
                )
            except Exception as e:
                if self.is_main:
                    print(f"Warning: Could not log confusion matrix: {e}")

        # Log OCR predictions less frequently (only every 20 epochs to reduce clutter)
        if epoch % 20 == 0 and self.current_stage in [TrainingStage.LCOFL, TrainingStage.FINETUNE] and self.is_main:
            if val_metrics.get("sample_batch"):
                self._log_sr_ocr_predictions(val_metrics["sample_batch"], epoch)

    def train_stage(self, stage: TrainingStage) -> float:
        """Train a specific stage."""
        self.set_stage(stage)
        config = self.stage_configs[stage]

        # Handle Stage 4 (Hard Mining) separately
        if stage == TrainingStage.HARD_MINING:
            return self.train_hard_mining_stage(config)

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

        # Load optimizer state if resuming from checkpoint
        self._load_optimizer_state()

        # Log stage start
        if self.text_logger and self.is_main:
            stage_descriptions = {
                TrainingStage.WARMUP: "Stabilize network with L1 loss only",
                TrainingStage.LCOFL: "Character-driven training with frozen OCR",
                TrainingStage.FINETUNE: "Joint optimization with unfrozen OCR",
            }
            self.text_logger.log_stage_start(
                stage_name=stage.value,
                epochs=config.epochs,
                lr=config.lr,
                description=stage_descriptions.get(stage, "")
            )

        # Calculate starting epoch within this stage
        # The checkpoint stores the stage-specific epoch, not global epoch
        stage_start_epoch = 0
        if hasattr(self, '_checkpoint_state') and 'stage' in self._checkpoint_state:
            checkpoint_stage = self._checkpoint_state.get('stage', '')
            if checkpoint_stage == stage.value:
                # Same stage - continue from saved epoch
                checkpoint_epoch = self._checkpoint_state.get('epoch', 0)
                stage_start_epoch = checkpoint_epoch + 1  # Resume from NEXT epoch

                # If checkpoint epoch >= config.epochs, the stage was already done.
                # When user passes e.g. --lcofl-epochs 100 to extend, interpret
                # config.epochs as ADDITIONAL epochs beyond the checkpoint.
                if stage_start_epoch >= config.epochs:
                    if self.is_main:
                        self._log(f"Checkpoint at epoch {checkpoint_epoch} >= configured {config.epochs} epochs. "
                                  f"Interpreting as {config.epochs} ADDITIONAL epochs.")
                    config = StageConfig(
                        name=config.name,
                        epochs=stage_start_epoch + config.epochs,
                        lr=config.lr,
                        loss_components=config.loss_components,
                        freeze_ocr=config.freeze_ocr,
                        update_confusion=config.update_confusion,
                    )
            else:
                # Different stage - start from 0
                stage_start_epoch = 0
                self.global_epoch = 0  # Reset for new stage

        if self.is_main:
            self._log(f"Starting from epoch {stage_start_epoch}, running to epoch {config.epochs}")

        for epoch in range(stage_start_epoch, config.epochs):
            # epoch is stage-specific (0 to config.epochs-1)
            # Update global_epoch after each epoch for tracking
            self.global_epoch = epoch + 1

            # Log epoch start (for timing)
            if self.text_logger and self.is_main:
                self.text_logger.log_epoch_start(epoch + 1, config.epochs)

            # Train
            train_metrics = self.train_epoch(config)

            # Get validation interval
            val_interval = self.config.get("training", {}).get("val_interval", 10)
            val_beam_width = self.config.get("training", {}).get("val_beam_width", 5)

            # Only validate every val_interval epochs (or first/last epoch)
            should_validate = (epoch % val_interval == 0) or (epoch == 0) or (epoch == config.epochs - 1)

            # DDP sync: ALL ranks must synchronize BEFORE validation
            # This prevents NCCL timeout when main rank takes longer due to validation
            self._safe_barrier(timeout_seconds=600, description="pre_validation")

            if should_validate:
                val_metrics = self.validate(beam_width=val_beam_width)

                # Update confusion if needed (only main rank has pred_texts)
                if config.update_confusion and self.is_main:
                    self.confusion_tracker.update(val_metrics["pred_texts"], val_metrics["gt_texts"])
                    self.lcofl_loss.update_weights(self.confusion_tracker.confusion_matrix)

                # Log to TensorBoard
                self.log_to_tensorboard(train_metrics, val_metrics, epoch)
            else:
                # Skip validation - provide minimal metrics for early stopping
                val_metrics = {"word_acc": 0.0, "char_acc": 0.0, "psnr": 0.0, "ssim": 0.0, "pred_texts": [], "gt_texts": []}

            # DDP sync: ALL ranks must synchronize AFTER validation (before early stopping check)
            # Use longer timeout since validation can take 15+ minutes
            self._safe_barrier(timeout_seconds=900, description="post_validation")

            # Log and print results (only when validation was performed)
            is_best = False
            if should_validate:
                # Check for best model
                if val_metrics['word_acc'] > self.best_word_acc:
                    self.best_word_acc = val_metrics['word_acc']
                    self.epochs_without_improvement = 0
                    is_best = True

                    self.save_checkpoint(epoch, stage)
                    save_path = str(self.save_dir / "best.pth")

                    if self.text_logger and self.is_main:
                        self.text_logger.log_best_model(
                            path=save_path,
                            metric_name="word_acc",
                            metric_value=val_metrics['word_acc'],
                            epoch=epoch + 1
                        )
                else:
                    self.epochs_without_improvement += 1

                # Log epoch metrics to text file
                if self.text_logger and self.is_main:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.text_logger.log_epoch_metrics(
                        epoch=epoch + 1,
                        total_epochs=config.epochs,
                        train_metrics={
                            "loss": train_metrics['loss'],
                            "l1": train_metrics.get('l1', 0.0),
                            "lcofl": train_metrics.get('lcofl', 0.0),
                        },
                        val_metrics={
                            "psnr": val_metrics['psnr'],
                            "ssim": val_metrics['ssim'],
                            "word_acc": val_metrics['word_acc'],
                            "char_acc": val_metrics['char_acc'],
                        },
                        lr=current_lr,
                        is_best=is_best
                    )

            # Learning rate scheduling
            self.scheduler.step()

            # Periodic checkpoint save (every 25 epochs) to avoid losing progress
            checkpoint_interval = self.config.get("training", {}).get("checkpoint_interval", 25)
            if (epoch + 1) % checkpoint_interval == 0 and self.is_main:
                self.save_checkpoint(epoch, stage, is_best=False)
                self._log(f"Periodic checkpoint saved at epoch {epoch + 1}")

            # Periodic memory cleanup to prevent OOM in long stages (Stage 2: 300 epochs, Stage 3: 150 epochs)
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            # Early stopping
            patience = self.config.get("training", {}).get("early_stop_patience", 20)
            if self.epochs_without_improvement >= patience:
                if self.text_logger and self.is_main:
                    self.text_logger.log_early_stopping(self.epochs_without_improvement, patience)
                break

        # Log stage end
        if self.text_logger and self.is_main:
            self.text_logger.log_stage_end(
                stage_name=stage.value,
                best_metric=self.best_word_acc,
                metric_name="word_acc"
            )

        self.global_epoch = config.epochs

        return self.best_word_acc

    def save_checkpoint(self, epoch: int, stage: TrainingStage, emergency: bool = False, is_best: bool = True):
        """Save a checkpoint (main process only for DDP).

        Args:
            epoch: Current epoch number
            stage: Current training stage
            emergency: If True, save as emergency checkpoint with timestamp
            is_best: If True, also save as best.pth. Set False for periodic saves.
        """
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

        # Always save OCR state_dict (even when frozen) for proper resume
        # The trained OCR from stage0/pretraining is critical for validation
        checkpoint["ocr_state_dict"] = self.ocr.state_dict()

        if emergency:
            # Emergency checkpoint - add timestamp and save to special location
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.save_dir / f"emergency_{timestamp}_stage_{stage.value}_epoch_{epoch}.pth"
            torch.save(checkpoint, save_path)
            self._log(f"Emergency checkpoint saved: {save_path}", "error")
            # Also save as latest emergency for easy resume
            save_path = self.save_dir / "emergency_latest.pth"
            torch.save(checkpoint, save_path)
        else:
            # Regular checkpoint — always save epoch-specific file
            save_path = self.save_dir / f"stage_{stage.value}_epoch_{epoch}.pth"
            torch.save(checkpoint, save_path)

            # Also save as latest.pth for easy resume
            latest_path = self.save_dir / "latest.pth"
            torch.save(checkpoint, latest_path)

            # Only overwrite best.pth when this is actually the best model
            if is_best:
                best_path = self.save_dir / "best.pth"
                torch.save(checkpoint, best_path)

    def _unwrap_model(self, model):
        """Unwrap DDP or DataParallel wrapper to get the underlying model."""
        if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
            return model.module
        return model

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        if self.is_main:
            self._log(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Store checkpoint state for later use (after optimizer is created)
        self._checkpoint_state = checkpoint

        # Handle DDP wrapper mismatch - checkpoint might have "module." prefix
        generator_state = checkpoint["generator_state_dict"]

        # Check if state dict has "module." prefix (from DDP save)
        has_module_prefix = any(k.startswith("module.") for k in generator_state.keys())
        is_ddp_wrapped = isinstance(self.generator, nn.parallel.DistributedDataParallel)

        # For DDP: load directly into wrapped model (expects "module." prefix)
        # For non-DDP: load into model directly (no prefix expected)
        if has_module_prefix and not is_ddp_wrapped:
            # Remove "module." prefix for non-DDP model
            generator_state = {k.replace("module.", ""): v for k, v in generator_state.items()}
            self.generator.load_state_dict(generator_state)
        elif not has_module_prefix and is_ddp_wrapped:
            # Add "module." prefix for DDP model
            generator_state = {"module." + k: v for k, v in generator_state.items()}
            self.generator.load_state_dict(generator_state)
        elif has_module_prefix and is_ddp_wrapped:
            # Both have prefix - load directly into wrapped model
            self.generator.load_state_dict(generator_state)
        else:
            # Neither has prefix - load directly
            self.generator.load_state_dict(generator_state)

        if "ocr_state_dict" in checkpoint:
            ocr_state = checkpoint["ocr_state_dict"]

            # Check if state dict has "module." prefix
            has_module_prefix = any(k.startswith("module.") for k in ocr_state.keys())
            is_ddp_wrapped = isinstance(self.ocr, nn.parallel.DistributedDataParallel)

            # Same logic for OCR
            if has_module_prefix and not is_ddp_wrapped:
                ocr_state = {k.replace("module.", ""): v for k, v in ocr_state.items()}
                self.ocr.load_state_dict(ocr_state)
            elif not has_module_prefix and is_ddp_wrapped:
                ocr_state = {"module." + k: v for k, v in ocr_state.items()}
                self.ocr.load_state_dict(ocr_state)
            elif has_module_prefix and is_ddp_wrapped:
                self.ocr.load_state_dict(ocr_state)
            else:
                self.ocr.load_state_dict(ocr_state)
        else:
            # OCR not in checkpoint - try loading from separate ocr_best.pth file
            # This handles old checkpoints where OCR wasn't saved when freeze_ocr=true
            checkpoint_dir = Path(checkpoint_path).parent
            ocr_best_path = checkpoint_dir / "ocr_best.pth"

            if ocr_best_path.exists():
                if self.is_main:
                    self._log(f"OCR not in checkpoint, loading from {ocr_best_path}...")

                # Use OCRModel.load() method which properly handles model_state_dict format
                # This loads the state_dict, vocab, and max_length correctly
                self.ocr.load(str(ocr_best_path))

                if self.is_main:
                    self._log(f"OCR state loaded from {ocr_best_path}")

        self.global_epoch = checkpoint.get("global_epoch", 0)
        self.best_word_acc = checkpoint.get("best_word_acc", 0.0)

        # Load the stage that was saved
        checkpoint_stage = checkpoint.get("stage", None)
        if checkpoint_stage:
            try:
                self.current_stage = TrainingStage(checkpoint_stage)
            except ValueError:
                pass  # Keep default if stage name not recognized

        if self.is_main:
            self._log(f"Checkpoint loaded: epoch={self.global_epoch}, stage={checkpoint_stage}, best_acc={self.best_word_acc:.4f}")

        # Note: No barrier here - the caller (train_progressive.py) handles synchronization

    def _load_optimizer_state(self):
        """Load optimizer state from checkpoint (must be called after optimizer is created)."""
        if hasattr(self, '_checkpoint_state') and 'optimizer_state_dict' in self._checkpoint_state:
            self.optimizer.load_state_dict(self._checkpoint_state['optimizer_state_dict'])
            if self.is_main:
                self._log("Optimizer state loaded from checkpoint")

    def train_full_progressive(self) -> Dict[str, Any]:
        """Train all stages sequentially."""
        results = {}

        if self.is_main:
            self._log("\n" + "=" * 60)
            self._log("PROGRESSIVE TRAINING START")
            self._log("=" * 60 + "\n")

        # Stage 0: OCR Pretraining
        if self.text_logger and self.is_main:
            self.text_logger.log_stage_start(
                stage_name="OCR Pretrain",
                epochs=self.stage_configs[TrainingStage.PRETRAIN].epochs,
                lr=self.stage_configs[TrainingStage.PRETRAIN].lr,
                description="Train OCR model on license plate data for meaningful generator guidance"
            )

        best_acc_stage0 = self.train_pretrain_stage(
            self.stage_configs[TrainingStage.PRETRAIN]
        )
        results["stage0_best_acc"] = best_acc_stage0

        if self.text_logger and self.is_main:
            self.text_logger.log_stage_end(
                stage_name="OCR Pretrain",
                best_metric=best_acc_stage0,
                metric_name="char_acc"
            )

        # Synchronize all ranks between stages (critical for DDP)
        self._safe_barrier(timeout_seconds=300, description="stage0_to_stage1")

        # Stage 1: Warm-up
        best_acc_stage1 = self.train_stage(TrainingStage.WARMUP)
        results["stage1_best_acc"] = best_acc_stage1

        # Synchronize all ranks between stages
        self._safe_barrier(timeout_seconds=300, description="stage1_to_stage2")

        # Stage 2: LCOFL
        best_acc_stage2 = self.train_stage(TrainingStage.LCOFL)
        results["stage2_best_acc"] = best_acc_stage2

        # Synchronize all ranks between stages
        self._safe_barrier(timeout_seconds=300, description="stage2_to_stage3")

        # Stage 3: Fine-tuning
        best_acc_stage3 = self.train_stage(TrainingStage.FINETUNE)
        results["stage3_best_acc"] = best_acc_stage3

        # Synchronize all ranks between stages
        self._safe_barrier(timeout_seconds=300, description="stage3_to_stage4")

        # Stage 4: Hard Example Mining (optional, check if configured)
        stage4_epochs = self.progressive_config.get("stage4", {}).get("epochs", 0)
        if stage4_epochs > 0:
            best_acc_stage4 = self.train_stage(TrainingStage.HARD_MINING)
            results["stage4_best_acc"] = best_acc_stage4
        else:
            results["stage4_best_acc"] = self.best_word_acc

        # Save final results
        results["final_best_acc"] = self.best_word_acc

        if self.is_main:
            with open(self.save_dir / "progressive_results.json", "w") as f:
                json.dump(results, f, indent=2)

            # Log final summary
            if self.text_logger:
                self.text_logger._write_raw("")
                self.text_logger._write_raw("╔" + "═" * 78 + "╗")
                self.text_logger._write_raw("║" + " PROGRESSIVE TRAINING SUMMARY ".center(78) + "║")
                self.text_logger._write_raw("╠" + "═" * 78 + "╣")
                self.text_logger._write_raw(f"║  Stage 0 (OCR Pretrain): {results['stage0_best_acc']:.4f} char_acc".ljust(79) + "║")
                self.text_logger._write_raw(f"║  Stage 1 (Warmup): {results['stage1_best_acc']:.4f} word_acc".ljust(79) + "║")
                self.text_logger._write_raw(f"║  Stage 2 (LCOFL): {results['stage2_best_acc']:.4f} word_acc".ljust(79) + "║")
                self.text_logger._write_raw(f"║  Stage 3 (Finetune): {results['stage3_best_acc']:.4f} word_acc".ljust(79) + "║")
                if stage4_epochs > 0:
                    self.text_logger._write_raw(f"║  Stage 4 (Hard Mining): {results['stage4_best_acc']:.4f} word_acc".ljust(79) + "║")
                self.text_logger._write_raw("╠" + "═" * 78 + "╣")
                self.text_logger._write_raw(f"║  Final Best Word Accuracy: {self.best_word_acc:.4f}".ljust(79) + "║")
                self.text_logger._write_raw(f"║  Results saved to: {str(self.save_dir)[:50]}".ljust(79) + "║")
                self.text_logger._write_raw("╚" + "═" * 78 + "╝")
                self.text_logger._write_raw("")

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
    ocr = OCRModel(frozen=True)

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
