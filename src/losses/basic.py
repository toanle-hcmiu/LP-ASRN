"""
Basic Loss Functions for Super-Resolution

Includes L1 loss, SSIM loss, and perceptual loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) Loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            L1 loss scalar
        """
        return F.l1_loss(pred, target)


class MSELoss(nn.Module):
    """MSE (Mean Squared Error) Loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            MSE loss scalar
        """
        return F.mse_loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.

    SSIM measures structural similarity between images.
    Using 1 - SSIM as the loss.
    """

    def __init__(
        self,
        window_size: int = 11,
        C1: float = 0.01**2,
        C2: float = 0.03**2,
    ):
        """
        Initialize SSIM Loss.

        Args:
            window_size: Size of the Gaussian window
            C1, C2: Stability constants
        """
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

    def ssim(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM between two tensors.

        Args:
            x: First tensor (B, C, H, W), assumed in [0, 1] or [-1, 1]
            y: Second tensor (B, C, H, W)

        Returns:
            SSIM value (scalar)
        """
        # Convert from [-1, 1] to [0, 1] if needed
        if x.min() < 0:
            x = (x + 1.0) / 2.0
            y = (y + 1.0) / 2.0

        # Clamp to valid range
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)

        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.window_size // 2)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.avg_pool2d(x**2, self.window_size, stride=1, padding=self.window_size // 2) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y**2, self.window_size, stride=1, padding=self.window_size // 2) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1, padding=self.window_size // 2) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / (
            (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)
        )

        return ssim_map.mean()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            SSIM loss scalar
        """
        return 1.0 - self.ssim(pred, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using feature extraction from a pretrained network.

    Compares features extracted from intermediate layers of a
    pretrained network (e.g., VGG) instead of just pixel values.
    """

    def __init__(
        self,
        feature_network: Optional[nn.Module] = None,
        layers: Optional[list] = None,
    ):
        """
        Initialize Perceptual Loss.

        Args:
            feature_network: Pretrained network for feature extraction
            layers: List of layer names to extract features from
        """
        super().__init__()

        if feature_network is None:
            # Use a simple VGG-like network by default
            # For OCR, we might use the OCR model's features
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.features = feature_network

        self.layers = layers or ["relu1_2", "relu2_2", "relu3_2"]

        # Freeze features
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image (B, 3, H, W) in range [-1, 1]
            target: Target image (B, 3, H, W) in range [-1, 1]

        Returns:
            Perceptual loss scalar
        """
        # Convert from [-1, 1] to [0, 1]
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        pred = (pred - mean) / std
        target = (target - mean) / std

        # Extract features
        pred_features = self.features(pred)
        target_features = self.features(target)

        # Compute L2 distance between features
        loss = F.mse_loss(pred_features, target_features)

        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 with epsilon for stability).

    L(x, y) = sqrt((x - y)^2 + eps^2)

    This is a differentiable variant of L1 that is more stable
    for very small differences.
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initialize Charbonnier Loss.

        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Charbonnier loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Charbonnier loss scalar
        """
        diff = pred - target
        return torch.sqrt(diff**2 + self.eps**2).mean()


if __name__ == "__main__":
    # Test losses
    print("Testing basic losses...")

    B, C, H, W = 4, 3, 32, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    # L1 Loss
    l1 = L1Loss()
    print(f"L1 Loss: {l1(pred, target).item():.4f}")

    # MSE Loss
    mse = MSELoss()
    print(f"MSE Loss: {mse(pred, target).item():.4f}")

    # SSIM Loss
    ssim_loss = SSIMLoss()
    print(f"SSIM Loss: {ssim_loss(pred, target).item():.4f}")

    # Charbonnier Loss
    charbonnier = CharbonnierLoss()
    print(f"Charbonnier Loss: {charbonnier(pred, target).item():.4f}")

    print("Basic losses test passed!")
