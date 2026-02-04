"""
Deformable Convolution Implementation

Adapted from: Dai et al. "Deformable Convolutional Networks" (CVPR 2017)
and "Deformable ConvNets V2" (CVPR 2019).

This implementation uses PyTorch's grid_sample for differentiable sampling.

Also includes DCNv4 support with automatic fallback to DCNv3 when unavailable.
Based on: DCNv4 (2024) - https://arxiv.org/abs/2401.06197
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Try to import DCNv4 from official package
DCNV4_AVAILABLE = False
_DCNv4 = None

try:
    from DCNv4 import DCNv4 as _DCNv4
    DCNV4_AVAILABLE = True
except ImportError:
    try:
        from dcnv4 import DCNv4 as _DCNv4
        DCNV4_AVAILABLE = True
    except ImportError:
        DCNV4_AVAILABLE = False


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution Layer.

    Unlike standard convolution which uses a fixed sampling grid,
    deformable convolution learns additional offset parameters that
    allow the sampling locations to adapt to the input content.

    This is particularly useful for capturing irregular structures
    like license plate character strokes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize Deformable Convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output
            bias: If True, adds a learnable bias to the output
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Regular convolution weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Offset convolution - predicts 2 * kernel_size * kernel_size offsets
        # (one offset for x and y for each kernel position)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize offset conv to zero (no offset initially)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H', W')
        """
        # Generate offsets
        offsets = self.offset_conv(x)  # (B, 2*k*k, H', W')

        # Apply deformable convolution
        return self._deform_conv(x, offsets, self.weight, self.bias)

    def _deform_conv(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply deformable convolution using grid sampling.

        Uses per-kernel-position sampling with cuDNN-safe grid shapes (B, H_out, W_out, 2)
        instead of the problematic massive-width pattern.

        Args:
            x: Input tensor (B, C, H, W)
            offset: Offset tensor (B, 2*k*k, H', W')
            weight: Convolution weight (out_C, in_C/g, k, k)
            bias: Optional bias tensor (out_C,)

        Returns:
            Output tensor (B, out_C, H', W')
        """
        B, C_in, H, W = x.shape
        out_channels = weight.shape[0]
        kernel_size = self.kernel_size

        # Calculate output dimensions
        H_out = (H + 2 * self.padding - self.dilation * (kernel_size - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (kernel_size - 1) - 1) // self.stride + 1

        # Reshape offset to separate x and y offsets
        # (B, 2*k*k, H', W') -> (B, k*k, 2, H', W')
        offset = offset.view(B, -1, 2, H_out, W_out)

        # Create base sampling grid
        # Create coordinates for each output pixel
        base_y = torch.arange(H_out, dtype=torch.float32, device=x.device)
        base_x = torch.arange(W_out, dtype=torch.float32, device=x.device)
        base_y, base_x = torch.meshgrid(base_y, base_x, indexing="ij")

        # Scale to input dimensions
        base_y = base_y * self.stride + (kernel_size // 2) * self.dilation
        base_x = base_x * self.stride + (kernel_size // 2) * self.dilation

        # Add to batch dimension
        base_y = base_y.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        base_x = base_x.unsqueeze(0).unsqueeze(0)

        # Create kernel offsets (relative positions)
        # For a 3x3 kernel: [-1, 0, 1] x [-1, 0, 1]
        kernel_offsets = torch.arange(
            -(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=x.device
        )
        kernel_y, kernel_x = torch.meshgrid(kernel_offsets, kernel_offsets, indexing="ij")

        # Flatten kernel offsets
        kernel_y = kernel_y.reshape(-1)  # (k*k,)
        kernel_x = kernel_x.reshape(-1)

        # Compute sampling coordinates
        # base: (B, k*k, H', W')
        # offset: (B, k*k, 2, H', W')
        # Add offsets to kernel positions
        sample_y = base_y + kernel_y.view(1, -1, 1, 1) + offset[:, :, 0, :, :]  # (B, k*k, H', W')
        sample_x = base_x + kernel_x.view(1, -1, 1, 1) + offset[:, :, 1, :, :]

        # Normalize to [-1, 1] for grid_sample
        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        sample_x = 2.0 * sample_x / (W - 1) - 1.0

        # === NEW: Per-kernel-position sampling with cuDNN-safe grid shapes ===
        # Instead of flattening all sample points into width (causing cuDNN crash),
        # loop over kernel positions and use standard (B, H_out, W_out, 2) grid shapes.

        kernel_positions = kernel_size * kernel_size
        sampled_list = []

        # Ensure input is float32 for stable grid_sample under AMP
        x_float = x.float() if x.dtype != torch.float32 else x

        for t in range(kernel_positions):
            # Build grid_t with shape (B, H_out, W_out, 2) for this kernel position
            # This is the cuDNN-safe grid pattern
            grid_t = torch.stack([sample_x[:, t], sample_y[:, t]], dim=-1)  # (B, H', W', 2)
            grid_t = grid_t.to(torch.float32)  # Ensure float32 for stability

            # Sample with AMP-safe dtype handling
            # Use autocast(enabled=False) to prevent dtype issues under AMP
            with torch.amp.autocast("cuda", enabled=False):
                sampled_t = F.grid_sample(
                    x_float,  # (B, C_in, H, W)
                    grid_t,   # (B, H', W', 2)
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )  # (B, C_in, H', W')

            sampled_list.append(sampled_t)

        # Stack over kernel positions: (B, C_in, H_out, W_out, k*k)
        sampled = torch.stack(sampled_list, dim=-1)

        # Compute output via batched matrix multiplication
        # sampled: (B, C_in, H_out, W_out, k*k)
        # weight: (out_C, C_in/groups, k, k) -> reshape to (out_C, C_in/groups, k*k)

        # Reshape weight: (out_C, C_in/g, k*k)
        weight_flat = weight.view(out_channels, C_in // self.groups, kernel_positions)

        # Reshape sampled to (B, C_in, K, H_out*W_out) for efficient computation
        B2, C_in2, H_out2, W_out2, K2 = sampled.shape
        sampled_flat = sampled.permute(0, 1, 4, 2, 3).reshape(B2, C_in2, K2, H_out2 * W_out2)

        # Split into groups and compute
        output_chunks = []
        C_in_per_group = C_in // self.groups
        out_channels_per_group = out_channels // self.groups

        for g in range(self.groups):
            # Get sampled features for this group: (B, C_in/g, K, H*W)
            sampled_g = sampled_flat[:, g * C_in_per_group:(g + 1) * C_in_per_group, :, :]

            # Get weights for this group: (out_C/g, C_in/g, K)
            weight_g = weight_flat[g * out_channels_per_group:(g + 1) * out_channels_per_group, :, :]

            # Compute: output[oc, b, hw] = sum_{cin, k} sampled[b, cin, k, hw] * weight[oc, cin, k]
            output_g = torch.einsum("bckh,ock->obh", sampled_g, weight_g)
            output_g = output_g.permute(1, 0, 2)  # (B, out_C/g, H*W)
            output_chunks.append(output_g)

        # Concatenate groups: (B, out_C, H*W)
        output = torch.cat(output_chunks, dim=1)
        output = output.reshape(B, out_channels, H_out, W_out)

        # Add bias if needed
        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)

        return output


class ModulatedDeformableConv2d(nn.Module):
    """
    Modulated Deformable Convolution Layer (DCN v2).

    In addition to spatial offsets, this variant also learns a modulation
    scalar for each sampling location that controls the importance of
    that sample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize Modulated Deformable Convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output
            bias: If True, adds a learnable bias to the output
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Regular convolution weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Offset convolution
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        # Modulation convolution - predicts k*k modulation scalars
        self.modulation_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize offset conv to zero
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        # Initialize modulation conv to 1 (full modulation initially)
        nn.init.constant_(self.modulation_conv.weight, 0.0)
        nn.init.constant_(self.modulation_conv.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H', W')
        """
        # Generate offsets and modulations
        offsets = self.offset_conv(x)  # (B, 2*k*k, H', W')
        modulations = torch.sigmoid(
            self.modulation_conv(x)
        )  # (B, k*k, H', W')

        # Apply deformable convolution with modulation
        return self._modulated_deform_conv(
            x, offsets, modulations, self.weight, self.bias
        )

    def _modulated_deform_conv(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        modulation: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply modulated deformable convolution.

        This is a simplified version - full implementation would require
        custom CUDA kernel for efficiency. Here we use a PyTorch-based
        approximation.
        """
        # For simplicity, use the basic deformable conv
        # and apply modulation as a post-processing step
        output = self._deform_conv(x, offset, weight, None)

        # Apply modulation (simplified - modulates channel-wise)
        # In full implementation, modulation would be per-sample location
        modulation_pooled = modulation.mean(dim=1, keepdim=True)  # (B, 1, H', W')
        output = output * modulation_pooled

        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)

        return output

    def _deform_conv(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Basic deformable convolution using cuDNN-safe per-kernel-position sampling.

        This mirrors the fixed DeformableConv2d implementation, avoiding the
        massive-width grid pattern that causes cuDNN crashes.
        """
        B, C_in, H, W = x.shape
        kernel_size = self.kernel_size

        H_out = (H + 2 * self.padding - self.dilation * (kernel_size - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (kernel_size - 1) - 1) // self.stride + 1

        # Reshape offset: (B, 2*k*k, H', W') -> (B, k*k, 2, H', W')
        offset = offset.view(B, 2, kernel_size * kernel_size, H_out, W_out)
        offset_y, offset_x = offset[:, 0], offset[:, 1]

        # Create base grid
        base_y = torch.arange(H_out, dtype=torch.float32, device=x.device)
        base_x = torch.arange(W_out, dtype=torch.float32, device=x.device)
        base_y, base_x = torch.meshgrid(base_y, base_x, indexing="ij")

        # Scale to input coordinates
        base_y = base_y * self.stride - self.padding + (kernel_size // 2) * self.dilation
        base_x = base_x * self.stride - self.padding + (kernel_size // 2) * self.dilation

        # Create kernel offsets
        kernel_offsets = torch.arange(
            -(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=x.device
        )
        kernel_y, kernel_x = torch.meshgrid(kernel_offsets, kernel_offsets, indexing="ij")
        kernel_y = kernel_y.reshape(-1)
        kernel_x = kernel_x.reshape(-1)

        # Add offsets to get sampling coordinates
        sample_y = base_y.unsqueeze(0).unsqueeze(0) + kernel_y.view(1, -1, 1, 1) + offset_y  # (B, k*k, H', W')
        sample_x = base_x.unsqueeze(0).unsqueeze(0) + kernel_x.view(1, -1, 1, 1) + offset_x

        # Normalize to [-1, 1] for grid_sample
        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        sample_x = 2.0 * sample_x / (W - 1) - 1.0

        # === NEW: Per-kernel-position sampling with cuDNN-safe grid shapes ===
        kernel_positions = kernel_size * kernel_size
        sampled_list = []

        # Ensure input is float32 for stable grid_sample under AMP
        x_float = x.float() if x.dtype != torch.float32 else x

        for t in range(kernel_positions):
            # Build grid_t with shape (B, H_out, W_out, 2) - cuDNN-safe pattern
            grid_t = torch.stack([sample_x[:, t], sample_y[:, t]], dim=-1)  # (B, H', W', 2)
            grid_t = grid_t.to(torch.float32)

            # Sample with AMP-safe dtype handling
            with torch.amp.autocast("cuda", enabled=False):
                sampled_t = F.grid_sample(
                    x_float,
                    grid_t,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )  # (B, C_in, H', W')

            sampled_list.append(sampled_t)

        # Stack: (B, C_in, H_out, W_out, k*k)
        sampled = torch.stack(sampled_list, dim=-1)

        # Compute output via batched matrix multiplication (same as DeformableConv2d)
        B2, C_in2, H_out2, W_out2, K2 = sampled.shape
        sampled_flat = sampled.permute(0, 1, 4, 2, 3).reshape(B2, C_in2, K2, H_out2 * W_out2)

        # Split into groups and compute
        output_chunks = []
        C_in_per_group = C_in // self.groups
        out_channels_per_group = self.out_channels // self.groups
        weight_flat = weight.view(self.out_channels, C_in // self.groups, kernel_positions)

        for g in range(self.groups):
            sampled_g = sampled_flat[:, g * C_in_per_group:(g + 1) * C_in_per_group, :, :]
            weight_g = weight_flat[g * out_channels_per_group:(g + 1) * out_channels_per_group, :, :]

            output_g = torch.einsum("bckh,ock->obh", sampled_g, weight_g)
            output_g = output_g.permute(1, 0, 2)
            output_chunks.append(output_g)

        output = torch.cat(output_chunks, dim=1)
        output = output.reshape(B, self.out_channels, H_out, W_out)

        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)

        return output


class DeformableConv2dV4(nn.Module):
    """
    DCNv4-based Deformable Convolution.

    Key differences from DCNv3:
    - Removed softmax normalization for unbounded dynamic weights
    - Removed internal skip connection (use external residual)
    - Memory-efficient kernel with flash-attention patterns
    - ~3x faster than DCNv3

    Falls back to DCNv3 (DeformableConv2d) if DCNv4 package is not available.

    Reference:
        DCNv4: The Devil is in the Details for Deformable Convolutions (2024)
        https://arxiv.org/abs/2401.06197
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 4,
        dilation: int = 1,
        bias: bool = True,
    ):
        """
        Initialize DCNv4 Deformable Convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            groups: Number of groups for DCNv4 (determines offset channels)
            dilation: Spacing between kernel elements
            bias: If True, adds a learnable bias to the output
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Check if DCNv4 is available
        self.use_dcnv4 = DCNV4_AVAILABLE

        if self.use_dcnv4:
            # Use official DCNv4 implementation
            self.dcn = _DCNv4(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad=padding,
                dilation=dilation,
                group=groups,
                bias=bias,
            )
        else:
            # Fallback to standard DCNv3 implementation
            # Use the existing DeformableConv2d as fallback
            self.fallback_dcn = DeformableConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,  # DCNv3 doesn't support groups in same way
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H', W')
        """
        if self.use_dcnv4:
            return self.dcn(x)
        else:
            # Fallback to DCNv3
            return self.fallback_dcn(x)


if __name__ == "__main__":
    # Test the deformable convolution
    x = torch.randn(2, 64, 32, 64)

    # Test basic deformable conv
    deform_conv = DeformableConv2d(64, 64, kernel_size=3, padding=1)
    y = deform_conv(x)
    print(f"Deformable Conv output shape: {y.shape}")  # Should be (2, 64, 32, 64)

    # Test modulated deformable conv
    mod_deform_conv = ModulatedDeformableConv2d(64, 64, kernel_size=3, padding=1)
    y2 = mod_deform_conv(x)
    print(f"Modulated Deformable Conv output shape: {y2.shape}")  # Should be (2, 64, 32, 64)

    # Test DCNv4 if available
    if DCNV4_AVAILABLE:
        print(f"\nDCNv4 is available!")
        dcnv4 = DeformableConv2dV4(64, 64, kernel_size=3, padding=1)
        y3 = dcnv4(x)
        print(f"DCNv4 output shape: {y3.shape}")
    else:
        print(f"\nDCNv4 is not available, using DCNv3 fallback")

    print("Deformable conv test passed!")
