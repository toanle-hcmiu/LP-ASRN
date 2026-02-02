"""
Image Visualization Utilities for LP-ASRN

Provides functions to create visualizations for TensorBoard logging,
including comparison grids, attention visualizations, and confusion matrices.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization disabled.")


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image from [-1, 1] to [0, 1].

    Args:
        image: Image tensor in range [-1, 1]

    Returns:
        Image tensor in range [0, 1]
    """
    return (image + 1.0) / 2.0


def create_comparison_grid(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    gt_texts: Optional[List[str]] = None,
    pred_texts: Optional[List[str]] = None,
    max_images: int = 8,
    padding: int = 4,
    text_color: Tuple[int, int, int] = (1, 1, 1),
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """
    Create a comparison grid showing LR, SR, and HR images side by side.

    Layout for each sample:
        [LR | SR | HR]
         GT  GT  GT

    Args:
        lr_images: Low-resolution images (B, 3, H, W) in range [-1, 1]
        sr_images: Super-resolved images (B, 3, H*2, W*2) in range [-1, 1]
        hr_images: High-resolution images (B, 3, H*2, W*2) in range [-1, 1]
        gt_texts: Ground truth texts (B,)
        pred_texts: Predicted texts (B,)
        max_images: Maximum number of samples to show
        padding: Padding between images
        text_color: RGB color for text
        background_color: RGB color for text background

    Returns:
        Grid image tensor of shape (3, H_grid, W_grid)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for create_comparison_grid")

    # Input validation
    if lr_images.dim() != 4 or sr_images.dim() != 4 or hr_images.dim() != 4:
        raise ValueError(
            f"Expected 4D tensors (B, C, H, W), got shapes: "
            f"lr={lr_images.shape}, sr={sr_images.shape}, hr={hr_images.shape}"
        )

    if lr_images.shape[0] == 0 or sr_images.shape[0] == 0 or hr_images.shape[0] == 0:
        raise ValueError(
            f"Empty batch detected: lr={lr_images.shape}, sr={sr_images.shape}, hr={hr_images.shape}"
        )

    B = min(lr_images.shape[0], max_images)

    if B <= 0:
        raise ValueError(f"Batch size must be positive, got B={B}")

    # Get dimensions
    _, _, H_lr, W_lr = lr_images.shape
    _, _, H_hr, W_hr = hr_images.shape

    # Validate dimensions - skip if any image has zero-dimension
    if H_lr <= 0 or W_lr <= 0 or H_hr <= 0 or W_hr <= 0:
        raise ValueError(f"Invalid image dimensions: lr=({H_lr}, {W_lr}), hr=({H_hr}, {W_hr})")

    # Upscale LR for display
    lr_upscaled = nn.functional.interpolate(
        lr_images[:B], size=(H_hr, W_hr), mode='bilinear', align_corners=False
    )

    # Calculate grid dimensions
    # 3 columns (LR, SR, HR) x B rows, with padding on left and between images
    H_grid = B * H_hr + (B + 1) * padding
    W_grid = 3 * W_hr + 4 * padding  # padding | LR | padding | SR | padding | HR | padding

    # Create blank canvas (channel by channel since torch.full requires scalar fill_value)
    grid = torch.zeros((3, H_grid, W_grid), dtype=torch.float32)
    if background_color != (0, 0, 0):
        for c in range(3):
            grid[c].fill_(background_color[c])

    # Denormalize images
    lr_disp = denormalize_image(lr_upscaled)
    sr_disp = denormalize_image(sr_images[:B])
    hr_disp = denormalize_image(hr_images[:B])

    for i in range(B):
        # Calculate positions
        row_y = i * (H_hr + padding) + padding

        # LR image
        grid[:, row_y:row_y+H_hr, padding:padding+W_hr] = lr_disp[i]

        # SR image
        grid[:, row_y:row_y+H_hr, W_hr+2*padding:2*W_hr+2*padding] = sr_disp[i]

        # HR image
        grid[:, row_y:row_y+H_hr, 2*W_hr+3*padding:3*W_hr+3*padding] = hr_disp[i]

    # Add text if available
    if gt_texts is not None or pred_texts is not None:
        try:
            # Validate that gt_texts and pred_texts have at least B elements
            gt_slice = gt_texts[:B] if gt_texts and len(gt_texts) >= B else None
            pred_slice = pred_texts[:B] if pred_texts and len(pred_texts) >= B else None

            if gt_slice is not None or pred_slice is not None:
                # Create text overlay using matplotlib
                grid = add_text_to_grid(
                    grid,
                    gt_slice,
                    pred_slice,
                    H_hr,
                    W_hr,
                    padding,
                )
        except Exception as e:
            # If text overlay fails, return grid without text
            print(f"Warning: Could not add text overlay to grid: {e}. Returning grid without text.")

    return grid


def add_text_to_grid(
    grid: torch.Tensor,
    gt_texts: Optional[List[str]],
    pred_texts: Optional[List[str]],
    H: int,
    W: int,
    padding: int,
) -> torch.Tensor:
    """Add text overlay to the comparison grid."""
    # Validate input grid dimensions
    if grid.dim() != 3:
        raise ValueError(f"Expected 3D grid tensor (C, H, W), got shape: {grid.shape}")

    grid_h, grid_w = grid.shape[1], grid.shape[2]

    # Ensure grid dimensions are positive and valid for matplotlib
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError(f"Invalid grid dimensions: H={grid_h}, W={grid_w}")

    # Calculate figure size with minimum bounds to avoid invalid dimensions
    # figsize is in inches, dpi determines the pixel dimensions
    dpi = 100
    figsize_w = max(grid_w / dpi, 1.0)  # Minimum 1 inch width
    figsize_h = max(grid_h / dpi, 1.0)  # Minimum 1 inch height

    fig = plt.figure(figsize=(figsize_w, figsize_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Display grid
    ax.imshow(grid.permute(1, 2, 0).numpy())

    # Add text labels
    B = min(len(gt_texts) if gt_texts else 0, len(pred_texts) if pred_texts else 0)

    for i in range(B):
        row_y = i * (H + padding) + padding

        # GT text (below LR)
        if gt_texts:
            text = f"GT: {gt_texts[i]}"
            color = "green" if pred_texts and pred_texts[i] == gt_texts[i] else "white"
            ax.text(padding, row_y + H + 2, text, color=color, fontsize=8, weight='bold')

        # Pred text (below SR)
        if pred_texts:
            text = f"Pred: {pred_texts[i]}"
            match_symbol = "✓" if gt_texts and pred_texts[i] == gt_texts[i] else "✗"
            color = "green" if gt_texts and pred_texts[i] == gt_texts[i] else "red"
            ax.text(W + 2 * padding, row_y + H + 2, f"{text} {match_symbol}",
                   color=color, fontsize=8, weight='bold')

    # Convert back to tensor
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    # Validate canvas dimensions before conversion
    if width <= 0 or height <= 0:
        plt.close(fig)
        raise ValueError(f"Invalid canvas dimensions: width={width}, height={height}")

    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    # Validate buffer size
    expected_size = height * width * 3
    if image_array.size != expected_size:
        plt.close(fig)
        raise ValueError(
            f"Buffer size mismatch: got {image_array.size}, expected {expected_size} "
            f"(height={height}, width={width})"
        )

    image_array = image_array.reshape(height, width, 3)
    grid_with_text = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

    # Validate output tensor dimensions
    if grid_with_text.shape[1] <= 0 or grid_with_text.shape[2] <= 0:
        plt.close(fig)
        raise ValueError(f"Invalid output tensor dimensions: {grid_with_text.shape}")

    plt.close(fig)
    return grid_with_text


def create_attention_visualization(
    attention_maps: torch.Tensor,
    input_image: torch.Tensor,
    num_heads: int = 4,
) -> torch.Tensor:
    """
    Visualize attention maps overlaid on input image.

    Args:
        attention_maps: Attention maps (B, H, W) or (B, num_heads, H, W)
        input_image: Input image (B, 3, H, W) in range [-1, 1]
        num_heads: Number of attention heads

    Returns:
        Visualization grid tensor (3, H_vis, W_vis)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for create_attention_visualization")

    B = min(attention_maps.shape[0], 4)  # Show max 4 samples

    # Normalize attention maps
    if attention_maps.dim() == 4:
        # Average over heads: (B, num_heads, H, W) -> (B, H, W)
        attention_maps = attention_maps.mean(dim=1)

    # Upscale attention if needed
    _, H_attn, W_attn = attention_maps.shape
    _, _, H_img, W_img = input_image.shape

    if H_attn != H_img or W_attn != W_img:
        attention_maps = nn.functional.interpolate(
            attention_maps.unsqueeze(1),
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

    # Normalize input image
    img_disp = denormalize_image(input_image[:B])

    # Create visualization grid
    # Each row: [original | attention | overlay]
    H_vis = B * (input_image.shape[2] + 8)
    W_vis = input_image.shape[3] * 3

    grid = torch.zeros(3, H_vis, W_vis)

    fig = plt.figure(figsize=(W_vis / 50, H_vis / 50), dpi=50)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # We'll create a simple grid manually
    row_h = input_image.shape[2]
    col_w = input_image.shape[3]

    for i in range(B):
        y_start = i * (row_h + 2)

        # Original image
        grid[:, y_start:y_start+row_h, 0:col_w] = img_disp[i]

        # Attention map (grayscale)
        attn_map = attention_maps[i].unsqueeze(0).repeat(3, 1, 1)
        grid[:, y_start:y_start+row_h, col_w:2*col_w] = attn_map

        # Overlay
        overlay = 0.6 * img_disp[i] + 0.4 * attn_map
        grid[:, y_start:y_start+row_h, 2*col_w:3*col_w] = overlay

    plt.close(fig)
    return grid


def create_confusion_figure(
    confusion_matrix: torch.Tensor,
    labels: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
) -> "Figure":
    """
    Create a confusion matrix figure.

    Args:
        confusion_matrix: Confusion matrix (C, C)
        labels: Character labels
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for create_confusion_figure")

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy
    if isinstance(confusion_matrix, torch.Tensor):
        cm = confusion_matrix.cpu().numpy()
    else:
        cm = confusion_matrix

    # Normalize rows
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    cm_norm = np.nan_to_num(cm_norm)

    # Plot heatmap
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{cm[i, j]:.0f}",
                          ha="center", va="center", color="black",
                          fontsize=6)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Normalized Count")

    plt.tight_layout()

    return fig


def create_loss_curves_figure(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 5),
) -> "Figure":
    """
    Create training curves figure.

    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Optional training accuracy
        val_accs: Optional validation accuracy
        title: Figure title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for create_loss_curves_figure")

    import matplotlib.pyplot as plt

    if train_accs is not None or val_accs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
        ax2 = None

    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    ax1.plot(epochs, train_losses, label="Train Loss", marker='o')
    ax1.plot(epochs, val_losses, label="Val Loss", marker='s')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracies if provided
    if ax2 is not None and train_accs is not None and val_accs is not None:
        ax2.plot(epochs, train_accs, label="Train Acc", marker='o')
        ax2.plot(epochs, val_accs, label="Val Acc", marker='s')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def make_image_grid(
    images: torch.Tensor,
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Create a simple grid from a batch of images.

    Args:
        images: Batch of images (B, C, H, W)
        nrow: Number of rows
        padding: Padding between images
        normalize: Whether to normalize from [-1, 1] to [0, 1]

    Returns:
        Grid image (C, H_grid, W_grid)
    """
    if normalize:
        images = denormalize_image(images)

    B, C, H, W = images.shape
    ncol = B // nrow

    H_grid = nrow * H + (nrow + 1) * padding
    W_grid = ncol * W + (ncol + 1) * padding

    grid = torch.zeros(C, H_grid, W_grid)
    grid.fill_(0.5)  # Gray padding

    for i in range(B):
        row = i // ncol
        col = i % ncol

        row_start = row * (H + padding) + padding
        col_start = col * (W + padding) + padding

        grid[:, row_start:row_start+H, col_start:col_start+W] = images[i]

    return grid


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")

    # Test sample data
    B, C, H, W = 4, 3, 17, 31
    lr_images = torch.randn(B, C, H, W) * 2 - 1
    hr_images = torch.randn(B, C, H*2, W*2) * 2 - 1
    sr_images = torch.randn(B, C, H*2, W*2) * 2 - 1

    gt_texts = ["ABC123", "XYZ789", "DEF456", "GHI012"]
    pred_texts = ["ABC123", "XY2789", "DEF456", "GHI012"]

    # Test comparison grid
    grid = create_comparison_grid(
        lr_images, sr_images, hr_images,
        gt_texts, pred_texts, max_images=4
    )
    print(f"Comparison grid shape: {grid.shape}")

    # Test confusion figure
    confusion = torch.randn(36, 36).abs()
    labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    fig = create_confusion_figure(confusion, labels)
    print("Confusion figure created")

    print("Visualization tests passed!")
