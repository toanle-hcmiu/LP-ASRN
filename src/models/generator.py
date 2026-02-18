"""
Generator Network for LP-ASRN (License Plate Super-Resolution Network)

Based on:
- Liang et al. "SwinIR: Image Restoration Using Swin Transformer" (CVPR 2022)
- Nascimento et al. "Enhancing License Plate Super-Resolution: A Layout-Aware
  and Character-Driven Approach" (2024)

The generator consists of:
1. Shallow Feature Extraction with PixelShuffle/PixelUnshuffle
2. Deep Feature Extraction with SwinIR Transformer blocks
3. Character Pyramid Attention for character-aware refinement
4. Upscaling Module with PixelShuffle
5. Reconstruction Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .attention import ThreeFoldAttentionModule, ResidualChannelAttentionBlock
from .deform_conv import DeformableConv2d
from .character_attention import MultiScaleCharacterAttention
from .swinir_blocks import ResidualSwinTransformerBlock


class ShallowFeatureExtractor(nn.Module):
    """
    Shallow Feature Extractor with auto-encoder structure.

    Uses PixelUnshuffle -> Conv -> PixelShuffle to extract and reorganize
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


class UpscalingModule(nn.Module):
    """
    Enhanced Upscaling Module with progressive refinement.

    Uses PixelShuffle with intermediate refinement convolutions
    and attention-based skip connections for better reconstruction.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        upscale_factor: int = 2,
        use_refinement: bool = True,
    ):
        """
        Initialize Upscaling Module.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output channels (3 for RGB)
            upscale_factor: Upscaling factor (2 for 2x, 4 for 4x)
            use_refinement: Whether to use intermediate refinement layers
        """
        super().__init__()

        self.upscale_factor = upscale_factor
        self.use_refinement = use_refinement

        # For 2x upscaling with progressive refinement
        if upscale_factor == 2:
            # First upscaling stage
            self.pre_conv = nn.Conv2d(
                in_channels, in_channels * 4, 3, padding=1
            )
            self.pixel_shuffle = nn.PixelShuffle(2)

            if use_refinement:
                # Intermediate refinement after upscaling
                self.refine1 = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                )
                # Attention for refinement
                self.refine_attn = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // 16, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // 16, in_channels, 1),
                    nn.Sigmoid()
                )

            # Final projection to output channels
            self.output_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # For 4x upscaling (two 2x stages with refinement)
        elif upscale_factor == 4:
            # First 2x stage
            self.stage1_pre = nn.Conv2d(in_channels, in_channels * 4, 3, padding=1)
            self.stage1_shuffle = nn.PixelShuffle(2)

            if use_refinement:
                # Refinement after first 2x
                self.stage1_refine = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.stage1_attn = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // 16, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // 16, in_channels, 1),
                    nn.Sigmoid()
                )

            # Second 2x stage
            self.stage2_pre = nn.Conv2d(in_channels, in_channels * 4, 3, padding=1)
            self.stage2_shuffle = nn.PixelShuffle(2)

            if use_refinement:
                # Refinement after second 2x
                self.stage2_refine = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.stage2_attn = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // 16, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // 16, in_channels, 1),
                    nn.Sigmoid()
                )

            # Final projection to output channels
            self.output_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

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
            # Pre-conv and pixel shuffle
            x = self.pre_conv(x)
            x = self.pixel_shuffle(x)

            if self.use_refinement:
                # Apply refinement with attention
                identity = x
                refined = self.refine1(x)
                attn_weights = self.refine_attn(refined)
                x = refined * attn_weights + identity

            # Final output projection
            x = self.output_conv(x)

        elif self.upscale_factor == 4:
            # First 2x stage
            x = self.stage1_pre(x)
            x = self.stage1_shuffle(x)

            if self.use_refinement:
                # Refinement after first 2x
                identity = x
                refined = self.stage1_refine(x)
                attn_weights = self.stage1_attn(refined)
                x = refined * attn_weights + identity

            # Second 2x stage
            x = self.stage2_pre(x)
            x = self.stage2_shuffle(x)

            if self.use_refinement:
                # Refinement after second 2x
                identity = x
                refined = self.stage2_refine(x)
                attn_weights = self.stage2_attn(refined)
                x = refined * attn_weights + identity

            # Final output projection
            x = self.output_conv(x)

        else:
            # General case
            x = self.pre_conv(x)
            x = self.pixel_shuffle(x)

        return x


class CharacterPyramidAttention(nn.Module):
    """
    Character Pyramid Attention Module for license plate super-resolution.

    Combines multi-scale feature pyramid with character-specific attention:
    1. Multi-scale feature pyramid (1/4, 1/2, 1/1 scales)
    2. Stroke detection kernels (horizontal, vertical, diagonal)
    3. Character gap detection
    4. Layout-aware positional encoding

    This module helps the generator focus on character regions and
    stroke-level details at multiple scales.
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_scales: int = 3,
        layout_type: str = "brazilian",  # "brazilian" or "mercocur"
    ):
        """
        Initialize Character Pyramid Attention.

        Args:
            in_channels: Number of input feature channels
            num_scales: Number of pyramid scales
            layout_type: Type of license plate layout for positional encoding
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_scales = num_scales
        self.layout_type = layout_type

        # Pyramid scales
        self.scales = [0.25, 0.5, 1.0][:num_scales]

        # Stroke detection kernels (detect horizontal, vertical, diagonal strokes)
        # These help identify character strokes at different orientations
        self.stroke_detectors = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
            for _ in range(4)  # H, V, D1, D2
        ])

        # Character gap detection (detect spaces between characters)
        self.gap_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Multi-scale feature processors
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
            )
            for _ in range(num_scales)
        ])

        # Scale-specific attention
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid()
            )
            for _ in range(num_scales)
        ])

        # Layout-aware positional encoding
        # Brazilian: LLLNNNN (3 letters + 4 digits)
        # Mercosur: LLLNLNN (3 letters + 1 digit + 1 letter + 2 digits)
        if layout_type == "brazilian":
            self.num_positions = 7
            self.layout_pattern = [0, 0, 0, 1, 1, 1, 1]  # 0=letter, 1=digit
        else:  # mercosur
            self.num_positions = 7
            self.layout_pattern = [0, 0, 0, 1, 0, 1, 1]  # LLLNLNN

        # Positional embeddings for each character position
        self.pos_embeddings = nn.Parameter(
            torch.randn(self.num_positions, in_channels) * 0.1
        )

        # Fusion of multi-scale features
        total_scale_channels = in_channels * num_scales + in_channels // 4 * 4  # scales + strokes
        self.fusion = nn.Sequential(
            nn.Conv2d(total_scale_channels, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

        # Output projection
        self.output_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W) with enhanced character features
        """
        B, C, H, W = x.shape
        identity = x

        # Detect strokes at different orientations
        stroke_features = []
        for stroke_detector in self.stroke_detectors:
            stroke = stroke_detector(x)
            stroke_features.append(stroke)
        all_strokes = torch.cat(stroke_features, dim=1)  # (B, C, H, W)

        # Detect character gaps
        gap_mask = self.gap_detector(x)  # (B, 1, H, W)

        # Multi-scale processing
        scale_outputs = []
        for i, (scale, processor, attn) in enumerate(
            zip(self.scales, self.scale_processors, self.scale_attentions)
        ):
            # Resize to scale
            if scale < 1.0:
                scaled_x = F.interpolate(
                    x, scale_factor=scale, mode='bilinear', align_corners=False
                )
            else:
                scaled_x = x

            # Process at this scale
            processed = processor(scaled_x)

            # Apply attention
            attn_weights = attn(processed)
            processed = processed * attn_weights

            # Resize back to original size
            if scale < 1.0:
                processed = F.interpolate(
                    processed, size=(H, W), mode='bilinear', align_corners=False
                )

            scale_outputs.append(processed)

        # Concatenate multi-scale features with stroke features
        multi_scale = torch.cat(scale_outputs, dim=1)  # (B, num_scales*C, H, W)
        combined = torch.cat([multi_scale, all_strokes], dim=1)

        # Fusion
        fused = self.fusion(combined)

        # Apply gap-aware attention
        fused = fused * (1 - gap_mask)  # Suppress gap regions

        # Add layout-aware positional encoding
        # Average pool to get (B, C, 1, 1) then add positional bias
        if W >= self.num_positions:
            # Divide width into character position regions
            region_width = W // self.num_positions
            pos_encoded = torch.zeros_like(fused)

            for pos_idx in range(self.num_positions):
                start_w = pos_idx * region_width
                end_w = min((pos_idx + 1) * region_width, W)

                # Add positional embedding to this region
                pos_emb = self.pos_embeddings[pos_idx].view(1, -1, 1, 1)
                pos_encoded[:, :, start_w:end_w] += pos_emb

            fused = fused + pos_encoded

        # Output projection
        output = self.output_conv(fused)

        # Residual connection
        return output + identity


class SwinIRDeepFeatureExtractor(nn.Module):
    """
    Deep Feature Extractor using SwinIR Architecture.

    Uses Residual Swin Transformer Blocks (RSTB) for efficient
    long-range modeling while maintaining spatial resolution.

    Automatically pads input to be divisible by window_size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 120,
        num_rstb: int = 6,
        num_heads: int = 6,
        window_size: int = 8,
        num_blocks_per_rstb: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Initialize SwinIR Deep Feature Extractor.

        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_rstb: Number of Residual Swin Transformer Blocks
            num_heads: Number of attention heads
            window_size: Window size for window attention
            num_blocks_per_rstb: Number of Swin blocks per RSTB
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Drop path rate
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_rstb = num_rstb
        self.window_size = window_size

        # Patch embedding (3x3 conv for feature extraction)
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)

        # Residual Swin Transformer Blocks
        self.layers = nn.ModuleList([
            ResidualSwinTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                num_blocks=num_blocks_per_rstb,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
            )
            for _ in range(num_rstb)
        ])

        # Final convolution
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Output tensor of shape (B, embed_dim, H, W)
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.conv_first(x)

        # Pad to be divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Pass through RSTB layers
        for layer in self.layers:
            x = layer(x)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H + pad_h, :W + pad_w]

        # Final conv + residual
        x = self.conv_after_body(x)

        return x


class Generator(nn.Module):
    """
    LP-ASRN Generator Network with SwinIR Architecture.

    Complete super-resolution generator for license plate images
    using Swin Transformer for efficient long-range modeling.

    Architecture:
    1. Shallow Feature Extractor - extracts initial features
    2. SwinIR Deep Features - processes with Transformer blocks
    3. Character Pyramid Attention (optional) - focuses on character regions
    4. Upscaling Module - upscales using PixelShuffle with progressive refinement
    5. Reconstruction Layer - final output

    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        embed_dim: Embedding dimension for SwinIR
        num_rstb: Number of Residual Swin Transformer Blocks
        num_heads: Number of attention heads
        window_size: Window size for window attention
        num_blocks_per_rstb: Number of Swin blocks per RSTB
        mlp_ratio: MLP expansion ratio
        qkv_bias: Whether to use bias in QKV projection
        upscale_factor: Upscaling factor (2 or 4)
        use_pyramid_attention: Whether to use Character Pyramid Attention
        use_upscale_refinement: Whether to use progressive refinement in upscaler
        pyramid_layout: Layout type for pyramid attention ("brazilian" or "mercocur")
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 144,
        num_rstb: int = 8,
        num_heads: int = 8,
        window_size: int = 6,
        num_blocks_per_rstb: int = 3,
        mlp_ratio: float = 6.0,
        qkv_bias: bool = True,
        upscale_factor: int = 2,
        use_pyramid_attention: bool = True,
        use_upscale_refinement: bool = True,
        pyramid_layout: str = "brazilian",
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Initialize SwinIR Generator.
        """
        super().__init__()

        self.upscale_factor = upscale_factor
        self.embed_dim = embed_dim

        # Shallow Feature Extractor
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, padding=1)

        # SwinIR Deep Feature Extractor
        self.deep_features = SwinIRDeepFeatureExtractor(
            in_channels=embed_dim,
            embed_dim=embed_dim,
            num_rstb=num_rstb,
            num_heads=num_heads,
            window_size=window_size,
            num_blocks_per_rstb=num_blocks_per_rstb,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        # Character Pyramid Attention (optional)
        if use_pyramid_attention:
            self.pyramid_attention = CharacterPyramidAttention(
                in_channels=embed_dim,
                num_scales=3,
                layout_type=pyramid_layout,
            )
        else:
            self.pyramid_attention = None

        # Upscaling Module with progressive refinement
        self.upscaler = UpscalingModule(
            in_channels=embed_dim,
            out_channels=out_channels,
            upscale_factor=upscale_factor,
            use_refinement=use_upscale_refinement,
        )

        # Reconstruction Layer
        self.reconstruction = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Tanh(),
        )

        # Skip connection
        self.skip_upscale = nn.Upsample(
            scale_factor=upscale_factor,
            mode='bilinear',
            align_corners=False,
        )
        self.skip_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input LR tensor of shape (B, 3, H, W) in range [-1, 1]

        Returns:
            Super-resolved tensor of shape (B, 3, H*scale, W*scale) in range [-1, 1]
        """
        input_lr = x
        B, C, H, W = x.shape

        # Shallow features
        shallow = self.conv_first(x)

        # Deep features with SwinIR
        deep_features = self.deep_features(shallow)

        # Character Pyramid Attention (optional)
        if self.pyramid_attention is not None:
            deep_features = self.pyramid_attention(deep_features)

        # Upscale
        upscaled = self.upscaler(deep_features)

        # Reconstruction
        output = self.reconstruction(upscaled)

        # Skip connection - match output size (may differ due to window padding)
        skip = self.skip_upscale(input_lr)
        _, _, out_H, out_W = output.shape
        if skip.shape[2:] != output.shape[2:]:
            skip = F.interpolate(skip, size=(out_H, out_W), mode='bilinear', align_corners=False)
        output = output + self.skip_weight * skip

        # Soft clamp to [-1, 1]
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

    Uses fewer blocks and simpler attention mechanisms for
    scenarios where speed is more important than accuracy.
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
    print("Testing SwinIR Generator...")

    # Test full generator with maximum configuration
    generator = Generator(
        in_channels=3,
        out_channels=3,
        embed_dim=144,
        num_rstb=8,
        num_heads=8,
        window_size=6,
        num_blocks_per_rstb=3,
        mlp_ratio=6.0,
        upscale_factor=2,
        use_pyramid_attention=True,
        pyramid_layout="brazilian",
    )

    x = torch.randn(2, 3, 34, 62)
    y = generator(x)
    print(f"Generator output shape: {y.shape}")  # Should be (2, 3, 68, 124)

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
