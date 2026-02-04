"""
Multi-Scale Character Attention Module for LP-SR

Implements a specialized attention mechanism that focuses on character-level
features at multiple scales for improved license plate reconstruction.

Key components:
1. CharacterRegionDetector - Lightweight character region detection
2. MultiScaleCharacterAttention - Combines multi-scale processing with character-aware attention

Based on the character-focused attention approach for license plate super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class CharacterRegionDetector(nn.Module):
    """
    Lightweight Character Region Detection using learned attention.

    Identifies regions likely to contain character strokes by learning
    spatial templates for character-like patterns.

    Uses a small set of learnable prototypes that represent typical
    character features (vertical/horizontal strokes, curves, etc.).
    """

    def __init__(
        self,
        in_channels: int,
        num_prototypes: int = 36,
        kernel_size: int = 3,
    ):
        """
        Initialize Character Region Detector.

        Args:
            in_channels: Number of input feature channels
            num_prototypes: Number of character prototype patterns
            kernel_size: Size of the prototype kernels
        """
        super().__init__()

        self.num_prototypes = num_prototypes

        # Learnable prototypes - each represents a character-like pattern
        # Initialize with random patterns
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, in_channels, kernel_size, kernel_size) * 0.1
        )

        # Attention mask generator
        self.detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # Prototype similarity head
        self.prototype_conv = nn.Conv2d(
            in_channels,
            num_prototypes,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect character regions in input features.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            region_mask: (B, 1, H, W) attention mask for character regions
            proto_scores: (B, num_prototypes, H, W) prototype similarity scores
        """
        # Generate region attention mask
        region_mask = self.detector(x)

        # Compute prototype similarity scores
        proto_scores = self.prototype_conv(x)  # (B, num_prototypes, H, W)

        return region_mask, proto_scores


class MultiScaleCharacterAttention(nn.Module):
    """
    Multi-Scale Character Attention Module (MSCA).

    Combines character region detection at multiple scales with
    guided attention to focus on character-relevant features.

    This helps the model focus on reconstructing individual characters
    at different scales, which is important for license plates where
    character size can vary significantly.
    """

    def __init__(
        self,
        in_channels: int,
        scales: Tuple[float, ...] = (1.0, 0.5, 0.25),
        num_prototypes: int = 36,
    ):
        """
        Initialize Multi-Scale Character Attention.

        Args:
            in_channels: Number of input feature channels
            scales: Tuple of scale factors for multi-scale processing
            num_prototypes: Number of character prototypes for detection
        """
        super().__init__()

        self.scales = scales
        self.in_channels = in_channels

        # Character detector at each scale
        self.detectors = nn.ModuleList([
            CharacterRegionDetector(in_channels, num_prototypes)
            for _ in scales
        ])

        # Guided attention at each scale
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.Sigmoid()
            )
            for _ in scales
        ])

        # Cross-scale fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels, 1),
            nn.ReLU(inplace=True),
        )

        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale character attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W) with enhanced character features
        """
        B, C, H, W = x.shape
        scale_features = []

        for scale, detector, attn_head in zip(
            self.scales, self.detectors, self.attention_heads
        ):
            # Scale the input if needed
            if scale < 1.0:
                scaled_x = F.interpolate(
                    x, scale_factor=scale, mode='bilinear', align_corners=False
                )
            else:
                scaled_x = x

            # Detect character regions at this scale
            region_mask, proto_scores = detector(scaled_x)

            # Apply guided attention
            # Concatenate features with region mask
            attn_input = torch.cat([scaled_x, region_mask], dim=1)
            attn_weights = attn_head(attn_input)

            # Apply attention to features
            attended = scaled_x * attn_weights

            # Upsample back to original size if needed
            if scale < 1.0:
                attended = F.interpolate(
                    attended, size=(H, W), mode='bilinear', align_corners=False
                )
                proto_scores = F.interpolate(
                    proto_scores, size=(H, W), mode='bilinear', align_corners=False
                )

            scale_features.append(attended)

        # Fuse multi-scale features
        fused = torch.cat(scale_features, dim=1)
        fused = self.fusion(fused)

        # Output projection
        output = self.output_conv(fused)

        # Residual connection
        return output + x


class CharacterAwareAttentionBlock(nn.Module):
    """
    Character-Aware Attention Block.

    A single attention block that combines:
    1. Character region detection
    2. Multi-scale feature processing
    3. Channel and spatial attention
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        num_prototypes: int = 36,
    ):
        """
        Initialize Character-Aware Attention Block.

        Args:
            in_channels: Number of input/output channels
            reduction_ratio: Reduction ratio for channel attention
            num_prototypes: Number of character prototypes
        """
        super().__init__()

        # Character region detection
        self.char_detector = CharacterRegionDetector(
            in_channels, num_prototypes
        )

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention with character guidance
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels + 1, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply character-aware attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = x

        # Detect character regions
        region_mask, _ = self.char_detector(x)

        # Channel attention
        channel_weights = self.channel_attn(x)
        channel_attended = x * channel_weights

        # Spatial attention with character region guidance
        spatial_input = torch.cat([channel_attended, region_mask], dim=1)
        spatial_weights = self.spatial_attn(spatial_input)
        spatial_attended = channel_attended * spatial_weights

        # Combine with original features
        combined = torch.cat([identity, spatial_attended], dim=1)
        output = self.fusion(combined)

        # Residual connection
        return output + identity


class AdaptiveCharacterAttention(nn.Module):
    """
    Adaptive Character Attention with dynamic scale selection.

    Unlike fixed multi-scale processing, this module learns to
    select the most appropriate scale for each region dynamically.
    """

    def __init__(
        self,
        in_channels: int,
        num_scales: int = 3,
        num_prototypes: int = 36,
    ):
        """
        Initialize Adaptive Character Attention.

        Args:
            in_channels: Number of input feature channels
            num_scales: Number of scales to use
            num_prototypes: Number of character prototypes
        """
        super().__init__()

        self.num_scales = num_scales

        # Scale-specific processors
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_scales)
        ])

        # Scale selection network
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_scales, 1),
            nn.Softmax(dim=1)
        )

        # Character-aware fusion
        self.char_detector = CharacterRegionDetector(in_channels, num_prototypes)
        self.fusion = nn.Conv2d(in_channels + 1, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive character attention.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        identity = x

        # Compute scale weights for the entire batch
        scale_weights = self.scale_selector(x)  # (B, num_scales, 1, 1)

        # Process at different scales
        scale_outputs = []
        scales = [1.0, 0.75, 0.5][:self.num_scales]

        for i, (scale, processor) in enumerate(zip(scales, self.scale_processors)):
            if scale < 1.0:
                scaled = F.interpolate(
                    x, scale_factor=scale, mode='bilinear', align_corners=False
                )
                processed = processor(scaled)
                processed = F.interpolate(
                    processed, size=(H, W), mode='bilinear', align_corners=False
                )
            else:
                processed = processor(x)

            # Apply scale weight
            weighted = processed * scale_weights[:, i:i+1, :, :]
            scale_outputs.append(weighted)

        # Combine scale outputs
        combined = torch.stack(scale_outputs, dim=0).sum(dim=0)

        # Character-aware fusion
        region_mask, _ = self.char_detector(combined)
        fused_input = torch.cat([combined, region_mask], dim=1)
        output = self.fusion(fused_input)

        # Residual connection
        return output + identity


if __name__ == "__main__":
    # Test the character attention modules
    print("Testing Character Attention Modules...")

    batch_size = 2
    in_channels = 64
    height, width = 32, 64

    x = torch.randn(batch_size, in_channels, height, width)

    # Test CharacterRegionDetector
    print("\n1. Testing CharacterRegionDetector...")
    detector = CharacterRegionDetector(in_channels, num_prototypes=36)
    region_mask, proto_scores = detector(x)
    print(f"   Region mask shape: {region_mask.shape}")
    print(f"   Proto scores shape: {proto_scores.shape}")

    # Test MultiScaleCharacterAttention
    print("\n2. Testing MultiScaleCharacterAttention...")
    msca = MultiScaleCharacterAttention(in_channels, scales=(1.0, 0.5, 0.25))
    msca_out = msca(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {msca_out.shape}")

    # Test CharacterAwareAttentionBlock
    print("\n3. Testing CharacterAwareAttentionBlock...")
    caab = CharacterAwareAttentionBlock(in_channels)
    caab_out = caab(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {caab_out.shape}")

    # Test AdaptiveCharacterAttention
    print("\n4. Testing AdaptiveCharacterAttention...")
    aca = AdaptiveCharacterAttention(in_channels, num_scales=3)
    aca_out = aca(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {aca_out.shape}")

    # Count parameters
    msca_params = sum(p.numel() for p in msca.parameters())
    print(f"\nMSCA parameters: {msca_params:,}")

    print("\nCharacter Attention Module tests passed!")
