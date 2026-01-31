"""
TensorBoard Logger for LP-ASRN Training

Provides comprehensive logging for training metrics, images, histograms,
and other visualizations for monitoring the training process.
"""

import os
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
            log_dir = f"logs/tensorboard/run_{timestamp}"

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
    ):
        """
        Log all model weights as histograms.

        Args:
            model: PyTorch model
            step: Global step
        """
        if step is None:
            step = self.global_step

        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                self.writer.add_histogram(f"weights/{name}", param.data, step)

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
            tag: Figure tag
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if step is None:
            step = self.global_step

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))

        # Convert to numpy for plotting
        if isinstance(confusion_matrix, torch.Tensor):
            cm = confusion_matrix.cpu().numpy()
        else:
            cm = confusion_matrix

        # Normalize each row
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Confusion Matrix")

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
