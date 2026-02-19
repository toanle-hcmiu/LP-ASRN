"""
Generator Network for LP-ASRN (License Plate Super-Resolution Network)

Based on:
- Nascimento et al. "Super-Resolution of License Plate Images Using Attention
  Modules and Sub-Pixel Convolution Layers" (2023)
- Nascimento et al. "Enhancing License Plate Super-Resolution: A Layout-Aware
  and Character-Driven Approach" (2024)

The generator consists of:
1. Shallow Feature Extraction with PixelShuffle/PixelUnshuffle
2. Deep Feature Extraction with RRDB-EA blocks
3. Upscaling Module with PixelShuffle
4. Reconstruction Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .attention import EnhancedAttentionModule, ThreeFoldAttentionModule, ResidualChannelAttentionBlock
from .deform_conv import DeformableConv2d
from .character_attention import MultiScaleCharacterAttention


class ShallowFeatureExtractor(nn.Module):
    """
    Shallow Feature Extractor with auto-encoder structure.

    Uses PixelUnshuffle → Conv → PixelShuffle to extract and reorganize
    shallow features, eliminating less significant features early.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        kernel_size: int = 5,
        use_autoencoder: bool = True,
    ):
        """
        Initialize Shallow Feature Extractor.

        Args:
            in_channels: Number of input channels (3 for RGB)
            num_features: Number of feature channels
            kernel_size: Size of the initial convolution kernel
            use_autoencoder: Whether to use the auto-encoder structure
        """
        super().__init__()

        self.use_autoencoder = use_autoencoder

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size, padding=kernel_size // 2)

        if use_autoencoder:
            # PixelUnshuffle to compress spatial info into channels
            self.pixel_unshuffle = nn.PixelUnshuffle(2)  # 2x compression

            # Process compressed features
            compressed_channels = num_features * 4
            self.conv_compressed = nn.Conv2d(
                compressed_channels, num_features, 3, padding=1
            )

            # PixelShuffle to expand back
            self.pixel_shuffle = nn.PixelShuffle(2)

            # Final projection
            self.conv_final = nn.Conv2d(
                num_features // 4, num_features, 3, padding=1
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, num_features, H, W)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.relu(x)

        identity = x

        if self.use_autoencoder:
            # Get original size
            B, C, H, W = x.shape

            # Pad to make dimensions divisible by 2
            pad_h = (2 - H % 2) % 2
            pad_w = (2 - W % 2) % 2
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

            # Auto-encoder path
            compressed = self.pixel_unshuffle(x_padded)  # (B, 4*C, H'/2, W'/2)
            compressed = self.relu(compressed)
            processed = self.conv_compressed(compressed)
            processed = self.relu(processed)
            expanded = self.pixel_shuffle(processed)  # (B, C/4, H', W')

            # Remove padding
            expanded = expanded[:, :, :H, :W]

            # Project back to num_features
            x = self.conv_final(expanded)
            x = self.relu(x)

        # Residual connection
        x = x + identity

        return x


class ResidualInResidualDenseBlock(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB) with Enhanced Attention.

    Combines dense connections, residual learning, and the enhanced
    attention module for powerful feature extraction.
    """

    def __init__(
        self,
        num_features: int = 64,
        num_layers: int = 3,
        use_enhanced_attention: bool = True,
        use_deformable: bool = True,
        growth_rate: int = 32,
    ):
        """
        Initialize RRDB-EA block.

        Args:
            num_features: Number of feature channels
            num_layers: Number of dense layers in the block
            use_enhanced_attention: Whether to use Enhanced Attention Module
            use_deformable: Whether to use deformable convolutions in attention
            growth_rate: Growth rate for dense connections
        """
        super().__init__()

        self.num_layers = num_layers
        self.use_enhanced_attention = use_enhanced_attention

        # Dense layers
        self.dense_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_features + i * growth_rate
            self.dense_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, growth_rate, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        # Output convolution for dense block
        total_channels = num_features + num_layers * growth_rate
        self.dense_conv = nn.Conv2d(total_channels, num_features, 3, padding=1)

        # Enhanced Attention Module
        if use_enhanced_attention:
            self.attention = EnhancedAttentionModule(
                num_features,
                use_deformable=use_deformable,
            )
        else:
            self.attention = None

        # Local feature convolution
        self.local_conv = nn.Conv2d(num_features, num_features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = x

        # Dense connections
        features = [x]
        for i, layer in enumerate(self.dense_layers):
            concatenated = torch.cat(features, dim=1)
            out = layer(concatenated)
            features.append(out)

        # Concatenate all dense outputs and convolve
        dense_out = torch.cat(features, dim=1)
        dense_out = self.dense_conv(dense_out)

        # Apply attention if enabled
        if self.attention is not None:
            attention_out = self.attention(dense_out)
            out = attention_out
        else:
            out = dense_out

        # Local convolution
        out = self.local_conv(out)

        # Residual connection
        out = out + identity

        return out


class DeepFeatureExtractor(nn.Module):
    """
    Deep Feature Extractor with multiple RRDB-EA blocks.

    Uses residual connections at multiple levels for stable
    training of deep networks.
    """

    def __init__(
        self,
        num_features: int = 64,
        num_blocks: int = 16,
        num_layers_per_block: int = 3,
        use_enhanced_attention: bool = True,
        use_deformable: bool = True,
    ):
        """
        Initialize Deep Feature Extractor.

        Args:
            num_features: Number of feature channels
            num_blocks: Number of RRDB-EA blocks
            num_layers_per_block: Number of layers in each RRDB-EA block
            use_enhanced_attention: Whether to use Enhanced Attention
            use_deformable: Whether to use deformable convolutions
        """
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                ResidualInResidualDenseBlock(
                    num_features=num_features,
                    num_layers=num_layers_per_block,
                    use_enhanced_attention=use_enhanced_attention,
                    use_deformable=use_deformable,
                )
            )

        # Global residual convolution
        self.global_conv = nn.Conv2d(num_features, num_features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = x

        # Pass through all blocks
        for block in self.blocks:
            x = block(x)

        # Global residual connection
        x = self.global_conv(x)
        x = x + identity

        return x


class UpscalingModule(nn.Module):
    """
    Upscaling Module using PixelShuffle.

    Efficiently upscales features using sub-pixel convolution.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        upscale_factor: int = 2,
    ):
        """
        Initialize Upscaling Module.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output channels (3 for RGB)
            upscale_factor: Upscaling factor (2 for 2x, 4 for 4x)
        """
        super().__init__()

        self.upscale_factor = upscale_factor

        # For 2x upscaling
        if upscale_factor == 2:
            self.pre_conv = nn.Conv2d(
                in_channels, out_channels * 4, 3, padding=1
            )
            self.pixel_shuffle = nn.PixelShuffle(2)

        # For 4x upscaling (can be done as two 2x or one 4x)
        elif upscale_factor == 4:
            # Two-stage upscaling for better quality
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )
        else:
            # General case (single stage)
            self.pre_conv = nn.Conv2d(
                in_channels, out_channels * (upscale_factor ** 2), 3, padding=1
            )
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Upscaled tensor of shape (B, 3, H*scale, W*scale)
        """
        if self.upscale_factor == 2:
            x = self.pre_conv(x)
            x = self.pixel_shuffle(x)
        elif self.upscale_factor == 4:
            x = self.stage1(x)
            x = self.stage2(x)
        else:
            x = self.pre_conv(x)
            x = self.pixel_shuffle(x)

        return x


class Generator(nn.Module):
    """
    LP-ASRN Generator Network.

    Complete super-resolution generator for license plate images.

    Architecture:
    1. Shallow Feature Extractor - extracts initial features
    2. Deep Feature Extractor - processes with RRDB-EA blocks
    3. Multi-Scale Character Attention (optional) - focuses on character regions
    4. Upscaling Module - upscales using PixelShuffle
    5. Reconstruction Layer - final output
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        num_layers_per_block: int = 3,
        upscale_factor: int = 2,
        use_enhanced_attention: bool = True,
        use_deformable: bool = True,
        use_character_attention: bool = False,
        msca_scales: Tuple[float, ...] = (1.0, 0.5, 0.25),
        msca_num_prototypes: int = 36,
        use_autoencoder: bool = True,
    ):
        """
        Initialize Generator.

        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for RGB)
            num_features: Number of feature channels
            num_blocks: Number of RRDB-EA blocks in deep feature extractor
            num_layers_per_block: Number of layers in each RRDB-EA block
            upscale_factor: Upscaling factor (2 or 4)
            use_enhanced_attention: Whether to use Enhanced Attention Module
            use_deformable: Whether to use deformable convolutions
            use_character_attention: Whether to use Multi-Scale Character Attention
            msca_scales: Scales for multi-scale character attention
            msca_num_prototypes: Number of character prototypes
            use_autoencoder: Whether to use auto-encoder in shallow extractor
        """
        super().__init__()

        self.upscale_factor = upscale_factor
        self.use_character_attention = use_character_attention

        # Shallow Feature Extractor
        self.shallow_extractor = ShallowFeatureExtractor(
            in_channels=in_channels,
            num_features=num_features,
            use_autoencoder=use_autoencoder,
        )

        # Deep Feature Extractor
        self.deep_extractor = DeepFeatureExtractor(
            num_features=num_features,
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
            use_enhanced_attention=use_enhanced_attention,
            use_deformable=use_deformable,
        )

        # Multi-Scale Character Attention (optional)
        if use_character_attention:
            self.character_attention = MultiScaleCharacterAttention(
                in_channels=num_features,
                scales=msca_scales,
                num_prototypes=msca_num_prototypes,
            )
        else:
            self.character_attention = None

        # Upscaling Module
        self.upscaler = UpscalingModule(
            in_channels=num_features,
            out_channels=out_channels,
            upscale_factor=upscale_factor,
        )

        # Reconstruction Layer (final conv before output)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Tanh(),  # Output range [-1, 1]
        )

        # Skip connection from input to output (for color preservation)
        self.skip_upscale = nn.Upsample(
            scale_factor=upscale_factor,
            mode='bilinear',
            align_corners=False,
        )
        # Learnable skip weight (initialized to 0.5 for a strong baseline)
        # At 0.5, the model starts close to bilinear upscale and learns residuals.
        # This provides a better starting point than 0.1 which barely uses the skip.
        self.skip_weight = 0.2  # Fixed for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input LR tensor of shape (B, 3, H, W) in range [-1, 1]

        Returns:
            Super-resolved tensor of shape (B, 3, H*scale, W*scale) in range [-1, 1]
        """
        # Store input for skip connection
        input_lr = x

        # Extract shallow features
        shallow_features = self.shallow_extractor(x)

        # Extract deep features
        deep_features = self.deep_extractor(shallow_features)

        # Apply Multi-Scale Character Attention if enabled
        if self.character_attention is not None:
            deep_features = self.character_attention(deep_features)

        # Upscale
        upscaled = self.upscaler(deep_features)

        # Reconstruction
        output = self.reconstruction(upscaled)

        # Add skip connection from input (upscaled) with learnable weight
        skip = self.skip_upscale(input_lr)
        output = output + self.skip_weight * skip

        # Soft clamp to valid range using tanh (preserves gradients at boundaries)
        # tanh naturally maps to [-1, 1] without killing gradients like hard clamp
        # Only apply when values exceed range to avoid distorting well-behaved outputs
        output = torch.where(
            output.abs() > 1.0,
            torch.tanh(output),
            output,
        )

        return output

    def get_last_layer_weights(self):
        """Get weights of the last reconstruction layer for visualization."""
        return self.reconstruction[0].weight.data


class LightweightGenerator(nn.Module):
    """
    Lightweight Generator for faster training/inference.

    Uses fewer blocks and simpler attention mechanisms.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        upscale_factor: int = 2,
    ):
        """
        Initialize Lightweight Generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_features: Number of feature channels
            num_blocks: Number of residual blocks
            upscale_factor: Upscaling factor
        """
        super().__init__()

        self.upscale_factor = upscale_factor

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks with channel attention
        self.res_blocks = nn.ModuleList([
            ResidualChannelAttentionBlock(num_features)
            for _ in range(num_blocks)
        ])

        # Upscaling
        if upscale_factor == 2:
            self.upconv = nn.Conv2d(num_features, out_channels * 4, 3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale_factor == 4:
            # Two-stage upscaling
            self.up1 = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
            self.up2 = nn.Sequential(
                nn.Conv2d(num_features, out_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            )

        # Final activation
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input LR tensor of shape (B, 3, H, W)

        Returns:
            Super-resolved tensor
        """
        identity = x

        # Initial features
        x = self.conv1(x)
        x = self.relu(x)

        shallow = x

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Residual connection
        x = x + shallow

        # Upscale
        if self.upscale_factor == 2:
            x = self.pixel_shuffle(self.upconv(x))
        elif self.upscale_factor == 4:
            x = self.up1(x)
            x = self.up2(x)

        # Final activation
        x = self.tanh(x)

        # Skip connection
        skip = F.interpolate(
            identity,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False,
        )
        x = x + skip
        x = torch.clamp(x, -1.0, 1.0)

        return x


if __name__ == "__main__":
    # Test the generator
    print("Testing Generator...")

    # Test full generator
    generator = Generator(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=4,  # Use fewer blocks for testing
        upscale_factor=2,
        use_deformable=False,  # Disable for CPU testing
    )

    x = torch.randn(2, 3, 17, 31)
    y = generator(x)
    print(f"Generator output shape: {y.shape}")  # Should be (2, 3, 34, 62)

    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test lightweight generator
    print("\nTesting Lightweight Generator...")
    lite_gen = LightweightGenerator(
        num_features=64,
        num_blocks=4,
        upscale_factor=2,
    )

    y_lite = lite_gen(x)
    print(f"Lightweight Generator output shape: {y_lite.shape}")

    lite_params = sum(p.numel() for p in lite_gen.parameters())
    print(f"Lightweight Generator parameters: {lite_params:,}")

    print("\nGenerator test passed!")
