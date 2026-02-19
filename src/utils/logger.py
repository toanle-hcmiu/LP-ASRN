"""
TensorBoard Logger for LP-ASRN Training

Provides comprehensive logging for training metrics, images, histograms,
and other visualizations for monitoring the training process.
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Optional, Union, List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Figure logging disabled.")


class TextLogger:
    """
    Enhanced text file logger for comprehensive training output.

    Provides rich logging including:
    - System information (GPU, memory, PyTorch version)
    - Model architecture summaries
    - Per-epoch metrics with tabular formatting
    - Stage transitions with visual separators
    - Best checkpoint tracking
    - Memory usage monitoring
    - Timing information
    """

    def __init__(
        self,
        log_dir: str = None,
        filename: str = "training.log",
        also_console: bool = True,
    ):
        """
        Initialize Text Logger.

        Args:
            log_dir: Directory for log file. If None, auto-generates path.
            filename: Name of the log file.
            also_console: If True, also print to console.
        """
        if log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"outputs/run_{timestamp}/logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / filename
        self.also_console = also_console

        # Tracking
        self.start_time = datetime.datetime.now()
        self.stage_start_time = None
        self.epoch_start_time = None
        self.current_stage = None
        self.best_metrics = {}

        # Create log file
        self.log_file.touch()

        # Write enhanced header
        self._write_header()

    def _write_header(self):
        """Write enhanced header with system information."""
        self._write_raw("")
        self._write_raw("╔" + "═" * 78 + "╗")
        self._write_raw("║" + " LP-ASRN TRAINING LOG ".center(78) + "║")
        self._write_raw("╠" + "═" * 78 + "╣")
        self._write_raw(f"║  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}".ljust(79) + "║")
        self._write_raw(f"║  Log file: {str(self.log_file)[:60]}...".ljust(79) + "║") if len(str(self.log_file)) > 60 else self._write_raw(f"║  Log file: {self.log_file}".ljust(79) + "║")
        self._write_raw("╚" + "═" * 78 + "╝")
        self._write_raw("")

    def _write_raw(self, message: str):
        """Write raw message without timestamp."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        if self.also_console:
            try:
                print(message)
            except UnicodeEncodeError:
                # Fallback for Windows console that doesn't support Unicode
                ascii_message = message.encode('ascii', 'replace').decode('ascii')
                print(ascii_message)

    def _write(self, message: str):
        """Write message to log file with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def info(self, message: str):
        """Log an info message."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self._write(message)
        if self.also_console:
            print(message)

    def debug(self, message: str):
        """Log a debug message."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [DEBUG] {message}"
        self._write(f"[DEBUG] {message}")
        if self.also_console:
            print(f"[DEBUG] {message}")

    def warning(self, message: str):
        """Log a warning message."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [WARNING] {message}"
        self._write(f"[WARNING] {message}")
        if self.also_console:
            print(f"[WARNING] {message}")

    def error(self, message: str):
        """Log an error message."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [ERROR] {message}"
        self._write(f"[ERROR] {message}")
        if self.also_console:
            print(f"[ERROR] {message}", file=sys.stderr)

    def log_system_info(self):
        """Log comprehensive system information."""
        self._write_raw("")
        self._write_raw("┌" + "─" * 78 + "┐")
        self._write_raw("│" + " SYSTEM INFORMATION ".center(78) + "│")
        self._write_raw("├" + "─" * 78 + "┤")

        # Python version
        self._write_raw(f"│  Python: {sys.version.split()[0]}".ljust(79) + "│")

        # PyTorch version
        self._write_raw(f"│  PyTorch: {torch.__version__}".ljust(79) + "│")

        # CUDA information
        if torch.cuda.is_available():
            self._write_raw(f"│  CUDA: {torch.version.cuda}".ljust(79) + "│")
            self._write_raw(f"│  cuDNN: {torch.backends.cudnn.version()}".ljust(79) + "│")

            # GPU information
            gpu_count = torch.cuda.device_count()
            self._write_raw(f"│  GPU Count: {gpu_count}".ljust(79) + "│")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self._write_raw(f"│    GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)".ljust(79) + "│")
        else:
            self._write_raw("│  CUDA: Not available".ljust(79) + "│")

        self._write_raw("└" + "─" * 78 + "┘")
        self._write_raw("")

    def log_model_summary(self, model_name: str, model: nn.Module, input_shape: tuple = None):
        """Log model architecture summary."""
        self._write_raw("")
        self._write_raw("┌" + "─" * 78 + "┐")
        self._write_raw(f"│ MODEL: {model_name} ".ljust(79) + "│")
        self._write_raw("├" + "─" * 78 + "┤")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        self._write_raw(f"│  Total Parameters: {total_params:,}".ljust(79) + "│")
        self._write_raw(f"│  Trainable: {trainable_params:,}".ljust(79) + "│")
        self._write_raw(f"│  Frozen: {frozen_params:,}".ljust(79) + "│")

        # Estimate model size
        param_size_mb = total_params * 4 / (1024**2)  # Assuming float32
        self._write_raw(f"│  Model Size: {param_size_mb:.2f} MB (float32)".ljust(79) + "│")

        self._write_raw("└" + "─" * 78 + "┘")
        self._write_raw("")

    def log_training_config(self, config: dict):
        """Log training configuration in a formatted table."""
        self._write_raw("")
        self._write_raw("┌" + "─" * 78 + "┐")
        self._write_raw("│" + " TRAINING CONFIGURATION ".center(78) + "│")
        self._write_raw("├" + "─" * 78 + "┤")

        # Progressive training stages
        prog = config.get("progressive_training", {})
        stages = [
            ("Stage 0 (Pretrain)", prog.get("stage0", {})),
            ("Stage 1 (Warmup)", prog.get("stage1", {})),
            ("Stage 2 (LCOFL)", prog.get("stage2", {})),
            ("Stage 3 (Finetune)", prog.get("stage3", {})),
        ]

        for stage_name, stage_config in stages:
            epochs = stage_config.get("epochs", "?")
            lr = stage_config.get("lr", "?")
            self._write_raw(f"│  {stage_name}: {epochs} epochs @ lr={lr}".ljust(79) + "│")

        self._write_raw("├" + "─" * 78 + "┤")

        # Data configuration
        data = config.get("data", {})
        self._write_raw(f"│  Batch Size: {data.get('batch_size', '?')}".ljust(79) + "│")
        self._write_raw(f"│  Num Workers: {data.get('num_workers', '?')}".ljust(79) + "│")
        self._write_raw(f"│  LR Size: {data.get('lr_size', '?')}".ljust(79) + "│")

        self._write_raw("├" + "─" * 78 + "┤")

        # Loss configuration
        loss = config.get("loss", {})
        self._write_raw(f"│  Lambda LCOFL: {loss.get('lambda_lcofl', 1.0)}".ljust(79) + "│")
        self._write_raw(f"│  Lambda Layout: {loss.get('lambda_layout', 0.5)}".ljust(79) + "│")
        self._write_raw(f"│  Lambda SSIM: {loss.get('lambda_ssim', 0.2)}".ljust(79) + "│")

        self._write_raw("└" + "─" * 78 + "┘")
        self._write_raw("")

    def log_stage_start(self, stage_name: str, epochs: int, lr: float, description: str = ""):
        """Log the start of a training stage with visual separator."""
        self.current_stage = stage_name
        self.stage_start_time = datetime.datetime.now()

        self._write_raw("")
        self._write_raw("█" * 80)
        self._write_raw(f"█  STAGE: {stage_name.upper()} ".ljust(79) + "█")
        self._write_raw("█" * 80)
        self._write_raw(f"│  Epochs: {epochs}")
        self._write_raw(f"│  Learning Rate: {lr}")
        if description:
            self._write_raw(f"│  Description: {description}")
        self._write_raw(f"│  Started: {self.stage_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_raw("─" * 80)
        self._write_raw("")

    def log_stage_end(self, stage_name: str, best_metric: float = None, metric_name: str = ""):
        """Log the end of a training stage."""
        if self.stage_start_time:
            duration = datetime.datetime.now() - self.stage_start_time
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = "N/A"

        self._write_raw("")
        self._write_raw("─" * 80)
        self._write_raw(f"│  Stage {stage_name} Complete")
        self._write_raw(f"│  Duration: {duration_str}")
        if best_metric is not None:
            self._write_raw(f"│  Best {metric_name}: {best_metric:.4f}")
        self._write_raw("─" * 80)
        self._write_raw("")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log the start of an epoch."""
        self.epoch_start_time = datetime.datetime.now()

    def log_epoch_metrics(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: dict,
        val_metrics: dict = None,
        lr: float = None,
        is_best: bool = False,
    ):
        """Log epoch metrics in a clean, tabular format."""
        # Calculate epoch duration
        if self.epoch_start_time:
            epoch_duration = datetime.datetime.now() - self.epoch_start_time
            duration_str = str(epoch_duration).split('.')[0]
        else:
            duration_str = "N/A"

        # Build epoch header
        stage_prefix = f"[{self.current_stage}] " if self.current_stage else ""
        best_marker = " ★ NEW BEST" if is_best else ""
        self._write_raw(f"\n{stage_prefix}Epoch {epoch}/{total_epochs} ({duration_str}){best_marker}")

        # Training metrics
        train_str = "  Train: "
        train_parts = []
        for key, value in train_metrics.items():
            if isinstance(value, float):
                train_parts.append(f"{key}={value:.4f}")
            elif isinstance(value, (int, str)):
                train_parts.append(f"{key}={value}")
        train_str += " | ".join(train_parts)
        self._write_raw(train_str)

        # Validation metrics
        if val_metrics:
            val_str = "  Val:   "
            val_parts = []
            for key, value in val_metrics.items():
                if isinstance(value, float):
                    val_parts.append(f"{key}={value:.4f}")
                elif isinstance(value, (int, str)):
                    val_parts.append(f"{key}={value}")
            val_str += " | ".join(val_parts)
            self._write_raw(val_str)

        # Learning rate
        if lr is not None:
            self._write_raw(f"  LR: {lr:.2e}")

        # GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            self._write_raw(f"  GPU Mem: {allocated:.2f}/{reserved:.2f} GB (allocated/reserved)")

    def log_checkpoint(self, path: str, metrics: dict):
        """Log checkpoint save event."""
        self._write_raw("")
        self._write_raw("┌" + "─" * 78 + "┐")
        self._write_raw("│  ✓ CHECKPOINT SAVED".ljust(79) + "│")
        self._write_raw(f"│  Path: {path[:65]}...".ljust(79) + "│") if len(path) > 65 else self._write_raw(f"│  Path: {path}".ljust(79) + "│")
        for key, value in metrics.items():
            if isinstance(value, float):
                self._write_raw(f"│  {key}: {value:.4f}".ljust(79) + "│")
        self._write_raw("└" + "─" * 78 + "┘")
        self._write_raw("")

    def log_best_model(self, path: str, metric_name: str, metric_value: float, epoch: int):
        """Log when a new best model is saved."""
        self.best_metrics[metric_name] = metric_value

        self._write_raw("")
        self._write_raw("╔" + "═" * 78 + "╗")
        self._write_raw("║  ★ NEW BEST MODEL SAVED ★".ljust(79) + "║")
        self._write_raw("╠" + "═" * 78 + "╣")
        self._write_raw(f"║  Epoch: {epoch}".ljust(79) + "║")
        self._write_raw(f"║  {metric_name}: {metric_value:.4f}".ljust(79) + "║")
        self._write_raw(f"║  Path: {path[:65]}...".ljust(79) + "║") if len(path) > 65 else self._write_raw(f"║  Path: {path}".ljust(79) + "║")
        self._write_raw("╚" + "═" * 78 + "╝")
        self._write_raw("")

    def log_validation_summary(self, metrics: dict, sample_predictions: list = None):
        """Log detailed validation summary."""
        self._write_raw("")
        self._write_raw("┌" + "─" * 78 + "┐")
        self._write_raw("│" + " VALIDATION SUMMARY ".center(78) + "│")
        self._write_raw("├" + "─" * 78 + "┤")

        for key, value in metrics.items():
            if isinstance(value, float):
                self._write_raw(f"│  {key}: {value:.4f}".ljust(79) + "│")
            elif isinstance(value, (int, str)):
                self._write_raw(f"│  {key}: {value}".ljust(79) + "│")

        if sample_predictions:
            self._write_raw("├" + "─" * 78 + "┤")
            self._write_raw("│  Sample Predictions:".ljust(79) + "│")
            for i, (pred, gt) in enumerate(sample_predictions[:5]):
                status = "✓" if pred == gt else "✗"
                self._write_raw(f"│    {status} GT: {gt:8s} | Pred: {pred:8s}".ljust(79) + "│")

        self._write_raw("└" + "─" * 78 + "┘")
        self._write_raw("")

    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics dictionary."""
        step_str = f"Step {step}" if step is not None else "Metrics"
        self.info(f"{step_str}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")

    def log_epoch(self, epoch: int, metrics: dict, stage: str = None):
        """Log epoch summary (legacy method for compatibility)."""
        stage_str = f"[{stage}] " if stage else ""
        self.info(f"{stage_str}Epoch {epoch}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")

    def log_early_stopping(self, epochs_without_improvement: int, patience: int):
        """Log early stopping event."""
        self._write_raw("")
        self._write_raw("┌" + "─" * 78 + "┐")
        self._write_raw("│  ⚠ EARLY STOPPING TRIGGERED".ljust(79) + "│")
        self._write_raw(f"│  Epochs without improvement: {epochs_without_improvement}/{patience}".ljust(79) + "│")
        self._write_raw("└" + "─" * 78 + "┘")
        self._write_raw("")

    def close(self):
        """Close the logger with enhanced footer."""
        end_time = datetime.datetime.now()
        total_duration = end_time - self.start_time
        duration_str = str(total_duration).split('.')[0]

        self._write_raw("")
        self._write_raw("╔" + "═" * 78 + "╗")
        self._write_raw("║" + " TRAINING COMPLETE ".center(78) + "║")
        self._write_raw("╠" + "═" * 78 + "╣")
        self._write_raw(f"║  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}".ljust(79) + "║")
        self._write_raw(f"║  Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}".ljust(79) + "║")
        self._write_raw(f"║  Total Duration: {duration_str}".ljust(79) + "║")
        self._write_raw("╠" + "═" * 78 + "╣")

        if self.best_metrics:
            self._write_raw("║  Best Metrics:".ljust(79) + "║")
            for key, value in self.best_metrics.items():
                self._write_raw(f"║    {key}: {value:.4f}".ljust(79) + "║")

        self._write_raw("╚" + "═" * 78 + "╝")
        self._write_raw("")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class TensorBoardLogger:
    """
    TensorBoard Logger for LP-ASRN training.

    Provides:
    - Scalar logging (losses, metrics, learning rates)
    - Image logging (LR, SR, HR comparisons)
    - Histogram logging (weights, gradients)
    - Figure logging (confusion matrices, etc.)
    - Text logging (custom messages)
    """

    def __init__(
        self,
        log_dir: str = None,
        comment: str = "",
        purge_step: Optional[int] = None,
    ):
        """
        Initialize TensorBoard Logger.

        Args:
            log_dir: Directory for TensorBoard logs. If None, auto-generates timestamped path.
            comment: Comment suffix for log directory
            purge_step: Step from which to purge old logs
        """
        # Auto-generate timestamped directory if not provided
        if log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"outputs/run_{timestamp}/logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            purge_step=purge_step,
        )

        self.global_step = 0

    def log_scalar(
        self,
        name: str,
        value: Union[float, int, torch.Tensor],
        step: Optional[int] = None,
    ):
        """
        Log a scalar value.

        Args:
            name: Metric name (e.g., "train/loss")
            value: Scalar value
            step: Global step (uses self.global_step if None)
        """
        if step is None:
            step = self.global_step

        self.writer.add_scalar(name, value, step)

    def log_scalars(
        self,
        scalar_dict: dict,
        step: Optional[int] = None,
        main_tag: str = "",
    ):
        """
        Log multiple scalars at once.

        Args:
            scalar_dict: Dictionary of metric names to values
            step: Global step
            main_tag: Prefix for all metric names
        """
        if step is None:
            step = self.global_step

        for name, value in scalar_dict.items():
            # Skip non-scalar values (lists, dicts, tensors, etc.)
            if isinstance(value, (list, dict, tuple)) or (hasattr(value, 'shape') and len(value.shape) > 0):
                continue
            if not isinstance(value, (int, float)):
                continue

            full_name = f"{main_tag}/{name}" if main_tag else name
            self.writer.add_scalar(full_name, value, step)

    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: Optional[int] = None,
        dataformats: str = "NCHW",
        max_images: int = 16,
    ):
        """
        Log a batch of images.

        Args:
            tag: Image tag (e.g., "train/lr_input")
            images: Image tensor of shape (B, C, H, W) in range [-1, 1] or [0, 1]
            step: Global step
            dataformats: Format of images ("NCHW" or "NHWC")
            max_images: Maximum number of images to log
        """
        if step is None:
            step = self.global_step

        # Limit number of images
        if images.shape[0] > max_images:
            images = images[:max_images]

        # Convert from [-1, 1] to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Clamp to valid range
        images = torch.clamp(images, 0, 1)

        self.writer.add_images(tag, images, step, dataformats=dataformats)

    def log_image_grid(
        self,
        tag: str,
        image_grid: torch.Tensor,
        step: Optional[int] = None,
    ):
        """
        Log an image grid (comparison visualization).

        Args:
            tag: Image tag
            image_grid: Grid tensor of shape (C, H, W) or (1, C, H, W)
            step: Global step
        """
        if step is None:
            step = self.global_step

        # Add channel dimension if needed
        if image_grid.dim() == 3:
            image_grid = image_grid.unsqueeze(0)

        # Ensure range [0, 1]
        if image_grid.min() < 0:
            image_grid = (image_grid + 1.0) / 2.0

        image_grid = torch.clamp(image_grid, 0, 1)

        self.writer.add_image(tag, image_grid, step, dataformats="NCHW")

    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: Optional[int] = None,
        bins: int = 100,
    ):
        """
        Log a histogram of values.

        Args:
            tag: Histogram tag
            values: Values to histogram
            step: Global step
            bins: Number of histogram bins
        """
        if step is None:
            step = self.global_step

        self.writer.add_histogram(tag, values, step, bins=bins)

    def log_model_weights(
        self,
        model: nn.Module,
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """
        Log all model weights as histograms.

        Args:
            model: PyTorch model
            step: Global step
            prefix: Prefix for histogram names (e.g., stage name)
        """
        if step is None:
            step = self.global_step

        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                tag = f"{prefix}/weights/{name}" if prefix else f"weights/{name}"
                self.writer.add_histogram(tag, param.data, step)

    def log_gradients(
        self,
        model: nn.Module,
        step: Optional[int] = None,
    ):
        """
        Log gradient norms for all model parameters.

        Args:
            model: PyTorch model
            step: Global step
        """
        if step is None:
            step = self.global_step

        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                self.writer.add_histogram(f"gradients/{name}", param.grad.data, step)

        total_norm = total_norm ** 0.5
        self.log_scalar("gradients/total_norm", total_norm, step)

    def log_figure(
        self,
        tag: str,
        figure: "Figure",
        step: Optional[int] = None,
        close: bool = True,
    ):
        """
        Log a matplotlib figure.

        Args:
            tag: Figure tag
            figure: Matplotlib figure
            step: Global step
            close: Whether to close the figure after logging
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if step is None:
            step = self.global_step

        self.writer.add_figure(tag, figure, step, close=close)

    def log_confusion_matrix(
        self,
        confusion_matrix: torch.Tensor,
        labels: List[str],
        step: Optional[int] = None,
        tag: str = "confusion_matrix",
    ):
        """
        Log a confusion matrix as a figure.

        Args:
            confusion_matrix: Confusion matrix tensor (C, C)
            labels: Character labels
            step: Global step
            tag: Figure tag (supports prefix like "stage2_lcofl/confusion_matrix")
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if step is None:
            step = self.global_step

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Calculate appropriate figure size based on number of classes
        n_classes = len(labels)
        fig_size = max(16, n_classes * 0.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

        # Convert to numpy for plotting
        if isinstance(confusion_matrix, torch.Tensor):
            cm = confusion_matrix.cpu().numpy()
        else:
            cm = confusion_matrix

        # Normalize each row
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)

        # Use integer annotations for readability, or fewer decimals
        # If values are very small, use scientific notation
        max_val = cm_normalized.max()
        if max_val < 0.01:
            fmt = ".1e"
        elif max_val < 0.1:
            fmt = ".3f"
        else:
            fmt = ".2f"

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"size": 8 if n_classes > 20 else 10},
            cbar_kws={"shrink": 0.8},
        )

        # Rotate labels for better readability
        ax.set_xlabel("Predicted", fontsize=12, fontweight='bold')
        ax.set_ylabel("Ground Truth", fontsize=12, fontweight='bold')
        ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')

        # Rotate tick labels if there are many classes
        if n_classes > 15:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        else:
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)

        plt.tight_layout()

        self.log_figure(tag, fig, step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ):
        """
        Log text as a markdown-formatted message.

        Args:
            tag: Text tag
            text: Markdown text
            step: Global step
        """
        if step is None:
            step = self.global_step

        self.writer.add_text(tag, text, step)

    def log_hparams(
        self,
        hparam_dict: dict,
        metric_dict: dict,
    ):
        """
        Log hyperparameters and final metrics.

        Args:
            hparam_dict: Hyperparameter dictionary
            metric_dict: Final metrics dictionary
        """
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_embedding(
        self,
        embedding_matrix: torch.Tensor,
        metadata: Optional[List[str]] = None,
        tag: str = "embedding",
        step: Optional[int] = None,
    ):
        """
        Log embeddings for visualization in TensorBoard projector.

        Args:
            embedding_matrix: Embedding matrix (N, D)
            metadata: Optional list of labels
            tag: Embedding tag
            step: Global step
        """
        if step is None:
            step = self.global_step

        self.writer.add_embedding(
            embedding_matrix,
            metadata=metadata,
            metadata_header=["label"] if metadata else None,
            tag=tag,
            global_step=step,
        )

    def increment_step(self, amount: int = 1):
        """
        Increment the global step counter.

        Args:
            amount: Amount to increment
        """
        self.global_step += amount

    def set_step(self, step: int):
        """
        Set the global step counter.

        Args:
            step: New step value
        """
        self.global_step = step

    def get_step(self) -> int:
        """Get the current global step."""
        return self.global_step

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Test the logger
    logger = TensorBoardLogger(log_dir="logs/test")

    # Test scalar logging
    logger.log_scalar("test/loss", 0.5, 0)
    logger.log_scalar("test/accuracy", 0.9, 0)

    # Test image logging
    images = torch.rand(4, 3, 32, 64) * 2 - 1
    logger.log_images("test/images", images, 0)

    # Test histogram logging
    weights = torch.randn(100)
    logger.log_histogram("test/weights", weights, 0)

    print("TensorBoard logger test passed!")
    print(f"Logs saved to: {logger.log_dir}")

    logger.close()
