"""
Attention Modules for LP-ASRN

Implements:
1. Channel Attention - focuses on "what" features are important
2. Spatial Attention - focuses on "where" features are important
3. Enhanced Attention Module (EAM) - combines both with deformable convolutions

Based on the Pixel Level Three-Fold Attention Module (PLTFAM) from:
Nascimento et al. "Super-Resolution of License Plate Images Using
Attention Modules and Sub-Pixel Convolution Layers" (2023)
and "Enhancing License Plate Super-Resolution" (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .deform_conv import DeformableConv2d, DeformableConv2dV4, DCNV4_AVAILABLE


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Learns to weight different feature channels based on their importance
    for reconstructing license plate characters.

    Uses both average pooling and max pooling to capture different
    aspects of channel information.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Initialize Channel Attention.

        Args:
            in_channels: Number of input channels
            reduction_ratio: Ratio for channel reduction in FC layers
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(in_channels // reduction_ratio, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Channel attention map of shape (B, C, 1, 1)
        """
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x))

        # Max pooling branch
        max_out = self.fc(self.max_pool(x))

        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)

        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Learns to weight different spatial locations based on their importance.
    This helps focus on character regions within the license plate.
    """

    def __init__(self, kernel_size: int = 7):
        """
        Initialize Spatial Attention.

        Args:
            kernel_size: Size of the convolution kernel
        """
        super().__init__()

        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Spatial attention map of shape (B, 1, H, W)
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))

        return attention


class GeometricalPerceptionUnit(nn.Module):
    """
    Geometrical Perception Unit (GPU).

    Captures horizontal and vertical structural information
    important for license plate character recognition.

    This is the "three-fold" aspect of PLTFAM - combining
    channel, positional, and geometrical information.
    """

    def __init__(self, in_channels: int):
        """
        Initialize Geometrical Perception Unit.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        # Global average pooling in vertical and horizontal directions
        # This captures the structure of characters

        # Point-wise convolution to process the pooled features
        self.conv_h = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Geometrical attention map of shape (B, C, 1, 1)
        """
        # Global average pooling
        # Pool across width (get vertical profile)
        v_pooled = F.adaptive_avg_pool2d(x, (x.shape[2], 1))  # (B, C, H, 1)
        # Pool across height (get horizontal profile)
        h_pooled = F.adaptive_avg_pool2d(x, (1, x.shape[3]))  # (B, C, 1, W)

        # Process each profile
        v_out = self.conv_v(v_pooled)
        h_out = self.conv_h(h_pooled)

        # Combine: average over remaining spatial dims
        v_global = F.adaptive_avg_pool2d(v_out, 1)  # (B, C, 1, 1)
        h_global = F.adaptive_avg_pool2d(h_out, 1)  # (B, C, 1, 1)

        # Aggregate through element-wise multiplication
        attention = self.sigmoid(v_global * h_global)

        return attention


class PixelShuffleAttention(nn.Module):
    """
    PixelShuffle-based Attention for channel reorganization.

    Uses PixelShuffle and PixelUnshuffle operations to reorganize
    channel information for better feature extraction.
    """

    def __init__(self, in_channels: int, upscale_factor: int = 2):
        """
        Initialize PixelShuffle Attention.

        Args:
            in_channels: Number of input channels
            upscale_factor: Factor for PixelShuffle operation
        """
        super().__init__()

        self.upscale_factor = upscale_factor

        # PixelUnshuffle to compress spatial into channels
        self.pixel_unshuffle = nn.PixelUnshuffle(upscale_factor)

        # Process in compressed space
        compressed_channels = in_channels * (upscale_factor ** 2)
        self.conv1 = nn.Conv2d(compressed_channels, in_channels, 1)

        # PixelShuffle to expand back
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.conv2 = nn.Conv2d(in_channels // (upscale_factor ** 2), in_channels, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention map of shape (B, C, H, W)
        """
        # Compress spatial dims into channels
        compressed = self.pixel_unshuffle(x)  # (B, C*r^2, H/r, W/r)

        # Process
        processed = self.conv1(compressed)

        # Expand back
        expanded = self.pixel_shuffle(processed)  # (B, C/r^2, H, W)

        # Final projection
        attention = self.sigmoid(self.conv2(expanded))

        return attention


class EnhancedAttentionModule(nn.Module):
    """
    Enhanced Attention Module (EAM) from LP-ASRN.

    Combines:
    1. Channel Attention - for inter-channel relationships
    2. Spatial/Positional Attention - for spatial localization
    3. Geometrical Perception - for structural features
    4. Deformable Convolutions - for adaptive feature extraction

    This is the key innovation that improves character reconstruction.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        use_deformable: bool = True,
        use_dcnv4: bool = True,
        dcnv4_groups: int = 4,
        shared_weights: bool = False,
    ):
        """
        Initialize Enhanced Attention Module.

        Args:
            in_channels: Number of input channels
            reduction_ratio: Ratio for channel reduction
            kernel_size: Size of spatial attention kernel
            use_deformable: Whether to use deformable convolutions
            use_dcnv4: Whether to use DCNv4 instead of DCNv3 (if available)
            dcnv4_groups: Number of groups for DCNv4
            shared_weights: If True, weights can be shared across modules
        """
        super().__init__()

        self.use_deformable = use_deformable
        self.use_dcnv4 = use_dcnv4 and DCNV4_AVAILABLE
        self.in_channels = in_channels

        # Channel Unit (CA)
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)

        # Positional Unit (POS) - can use deformable conv
        if use_deformable:
            if self.use_dcnv4:
                self.pos_conv = DeformableConv2dV4(
                    in_channels, in_channels,
                    kernel_size=3, padding=1,
                    groups=dcnv4_groups
                )
            else:
                self.pos_conv = DeformableConv2d(
                    in_channels, in_channels,
                    kernel_size=3, padding=1
                )
        else:
            self.pos_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        # Geometrical Perception Unit (GP)
        self.geometrical_attention = GeometricalPerceptionUnit(in_channels)

        # Final projection
        if use_deformable:
            if self.use_dcnv4:
                self.final_conv = DeformableConv2dV4(
                    in_channels, in_channels,
                    kernel_size=3, padding=1,
                    groups=dcnv4_groups
                )
            else:
                self.final_conv = DeformableConv2d(
                    in_channels, in_channels,
                    kernel_size=3, padding=1
                )
        else:
            self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = x

        # Channel Attention
        ca_out = self.channel_attention(x)

        # Positional Unit with deformable conv
        pos_out = self.pos_conv(x)

        # Geometrical Perception
        gp_out = self.geometrical_attention(x)

        # Combine all three attention mechanisms
        # Element-wise sum and multiplication
        combined = ca_out * pos_out
        combined = combined + gp_out  # Broadcast GP from (B, C, 1, 1)

        # Final processing
        attention_mask = self.sigmoid(self.final_conv(combined))

        # Apply attention to input
        output = identity * attention_mask + identity

        return output


class ThreeFoldAttentionModule(nn.Module):
    """
    Pixel Level Three-Fold Attention Module (PLTFAM).

    This is the full implementation from Nascimento et al. (2023).

    Combines:
    1. Channel Unit (CA) - inter-channel relationships
    2. Positional Unit (POS) - spatial localization with PixelShuffle
    3. Geometrical Perception Unit (GP) - structural features

    The combination of these three units creates a powerful attention
    mechanism specifically designed for license plate character reconstruction.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        upscale_factor: int = 2,
    ):
        """
        Initialize PLTFAM.

        Args:
            in_channels: Number of input channels
            reduction_ratio: Ratio for channel reduction in FC layers
            upscale_factor: Factor for PixelShuffle operations
        """
        super().__init__()

        self.in_channels = in_channels

        # Channel Unit (CA)
        # Uses parallel convolutions followed by concatenation
        self.ca_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.ca_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.ca_concat_conv = nn.Conv2d(in_channels, in_channels, 1, bias=True)

        # PixelUnshuffle and PixelShuffle for channel reorganization
        self.upscale_factor = upscale_factor
        self.pixel_unshuffle = nn.PixelUnshuffle(upscale_factor)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # Depthwise separable convolution
        # After pixel_unshuffle, channels = in_channels * upscale_factor^2
        self.compressed_channels = in_channels * (upscale_factor ** 2)
        self.dw_conv = nn.Conv2d(
            self.compressed_channels, self.compressed_channels, 3, padding=1, groups=self.compressed_channels
        )

        # Positional Unit (POS)
        # First-order statistics via average and max pooling
        self.pos_pooled_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Geometrical Perception Unit (GP)
        # Global pooling in vertical and horizontal directions
        self.gp_conv_h = nn.Conv2d(in_channels, in_channels, 1)
        self.gp_conv_v = nn.Conv2d(in_channels, in_channels, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # === Channel Unit ===
        ca_branch1 = self.ca_conv1(x)
        ca_branch2 = self.ca_conv2(x)
        ca_concat = torch.cat([ca_branch1, ca_branch2], dim=1)
        ca_out = self.ca_concat_conv(ca_concat)

        # Apply PixelUnshuffle and PixelShuffle
        ca_compressed = self.pixel_unshuffle(ca_out)  # (B, C*r^2, H/r, W/r)
        ca_processed = self.dw_conv(ca_compressed)
        ca_restored = self.pixel_shuffle(ca_processed)  # (B, C/r^2, H, W)

        # Project back to original channels
        if ca_restored.shape[1] != C:
            ca_restored = F.adaptive_avg_pool2d(ca_restored, (H, W))
            ca_restored = ca_restored.mean(dim=1, keepdim=True).repeat(1, C, 1, 1)

        # === Positional Unit ===
        pos_avg = F.adaptive_avg_pool2d(x, 1)
        pos_max = F.adaptive_max_pool2d(x, 1)
        pos_pooled = torch.cat([pos_avg, pos_max], dim=1)
        pos_out = self.pos_pooled_conv(pos_pooled)
        pos_out = pos_out.repeat(1, 1, H, W)  # Broadcast to original size

        # Apply PixelShuffle
        pos_restored = self.pixel_shuffle(
            self.pixel_unshuffle(pos_out)
        )  # (B, C/r^2, H, W)

        if pos_restored.shape[1] != C:
            pos_restored = F.adaptive_avg_pool2d(pos_restored, (H, W))
            pos_restored = pos_restored.mean(dim=1, keepdim=True).repeat(1, C, 1, 1)

        # === Geometrical Perception Unit ===
        # Vertical pooling
        gp_v = F.adaptive_avg_pool2d(x, (H, 1))  # (B, C, H, 1)
        gp_v = self.gp_conv_v(gp_v)  # (B, C, H, 1)
        gp_v = F.adaptive_avg_pool2d(gp_v, 1)  # (B, C, 1, 1)

        # Horizontal pooling
        gp_h = F.adaptive_avg_pool2d(x, (1, W))  # (B, C, 1, W)
        gp_h = self.gp_conv_h(gp_h)  # (B, C, 1, W)
        gp_h = F.adaptive_avg_pool2d(gp_h, 1)  # (B, C, 1, 1)

        # Combine
        gp_out = gp_v * gp_h  # (B, C, 1, 1)

        # === Combine all three ===
        # Element-wise sum then multiplication
        combined = ca_restored + pos_restored
        combined = combined * gp_out  # GP broadcast

        # Final sigmoid
        attention_mask = self.sigmoid(combined)

        return attention_mask


class ResidualChannelAttentionBlock(nn.Module):
    """
    Residual Block with Channel Attention (RCAB).

    A simple residual block enhanced with channel attention.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
    ):
        """
        Initialize RCAB.

        Args:
            in_channels: Number of input/output channels
            reduction_ratio: Ratio for channel reduction
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply channel attention
        ca_mask = self.channel_attention(out)
        out = out * ca_mask

        # Residual connection
        out = out + identity

        return out


if __name__ == "__main__":
    # Test the attention modules
    x = torch.randn(2, 64, 32, 64)

    # Test Channel Attention
    ca = ChannelAttention(64)
    ca_out = ca(x)
    print(f"Channel Attention output shape: {ca_out.shape}")

    # Test Spatial Attention
    sa = SpatialAttention()
    sa_out = sa(x)
    print(f"Spatial Attention output shape: {sa_out.shape}")

    # Test Geometrical Perception Unit
    gpu = GeometricalPerceptionUnit(64)
    gpu_out = gpu(x)
    print(f"Geometrical Perception output shape: {gpu_out.shape}")

    # Test Enhanced Attention Module
    eam = EnhancedAttentionModule(64, use_deformable=False)
    eam_out = eam(x)
    print(f"Enhanced Attention Module output shape: {eam_out.shape}")

    # Test Three-Fold Attention Module
    pltfam = ThreeFoldAttentionModule(64)
    plfam_out = pltfam(x)
    print(f"PLTFAM output shape: {plfam_out.shape}")

    print("Attention modules test passed!")
