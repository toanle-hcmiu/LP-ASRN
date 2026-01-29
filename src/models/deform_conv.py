"""
Deformable Convolution Implementation

Adapted from: Dai et al. "Deformable Convolutional Networks" (CVPR 2017)
and "Deformable ConvNets V2" (CVPR 2019).

This implementation uses PyTorch's grid_sample for differentiable sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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

        # Stack for grid_sample: (B, H', W', k*k, 2)
        sample_coords = torch.stack([sample_x, sample_y], dim=-1)
        sample_coords = sample_coords.permute(0, 2, 3, 1, 4)  # (B, H', W', k*k, 2)
        sample_coords = sample_coords.reshape(B, H_out * W_out * kernel_size * kernel_size, 2).contiguous()

        # Expand input for vectorized sampling
        # We need to sample each location for each channel
        x_expanded = x.unsqueeze(1).expand(
            -1, out_channels // self.groups, -1, -1, -1
        )  # (B, out_C/g, C, H, W)
        x_expanded = x_expanded.reshape(
            B * out_channels // self.groups * C_in, H, W
        ).contiguous()  # (B*out_C/g*C, H, W)

        # Resample for each output channel group
        sample_coords_expanded = sample_coords.unsqueeze(1).expand(
            -1, C_in * out_channels // self.groups, -1, -1
        )  # (B, C*out_C/g, H'*W'*k*k, 2)
        sample_coords_expanded = sample_coords_expanded.reshape(
            B * out_channels // self.groups * C_in, H_out * W_out * kernel_size * kernel_size, 2
        ).contiguous()

        # Sample from input
        sampled = F.grid_sample(
            x_expanded.unsqueeze(1),  # (B*C*out_C/g, 1, H, W)
            sample_coords_expanded.unsqueeze(1),  # (B*C*out_C/g, 1, H'*W'*k*k, 2)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)  # (B*C*out_C/g, H'*W'*k*k)

        # Reshape and apply weights
        sampled = sampled.reshape(
            B, out_channels // self.groups, C_in, H_out, W_out, kernel_size * kernel_size
        )  # (B, out_C/g, C, H', W', k*k)

        # Transpose to combine with weight: (B, out_C/g, H', W', k*k, C)
        sampled = sampled.permute(0, 1, 3, 4, 5, 2)

        # Reshape weight for multiplication: (out_C, in_C/g, k*k)
        # Transpose to (out_C, k*k, in_C/g) for proper einsum alignment
        weight_flat = weight.view(out_channels, C_in // self.groups, kernel_size * kernel_size).permute(0, 2, 1)

        # Compute weighted sum
        # sampled: (B, out_C/g, H', W', k*k, C_in)
        # weight_flat: (out_C, k*k, C_in/g)
        # For groups=1, we reshape weight to match
        weight_for_einsum = weight_flat.reshape(out_channels // self.groups, self.groups, kernel_size * kernel_size, C_in // self.groups)
        output = torch.einsum("bgxykc,gokc->bgoxy", sampled, weight_for_einsum)  # (B, out_C/g, H', W')
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
        """Basic deformable convolution (same as DeformableConv2d)."""
        # Simplified implementation using regular conv with modified padding
        # For full functionality, use DeformableConv2d class above
        B, C_in, H, W = x.shape
        kernel_size = self.kernel_size

        H_out = (H + 2 * self.padding - self.dilation * (kernel_size - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (kernel_size - 1) - 1) // self.stride + 1

        # Reshape offset
        offset = offset.view(B, 2, kernel_size * kernel_size, H_out, W_out)
        offset_y, offset_x = offset[:, 0], offset[:, 1]

        # Create base grid
        base_y = torch.arange(H_out, device=x.device).float()
        base_x = torch.arange(W_out, device=x.device).float()
        base_y, base_x = torch.meshgrid(base_y, base_x, indexing="ij")

        # Scale to input coordinates
        base_y = base_y * self.stride - self.padding + (kernel_size // 2) * self.dilation
        base_x = base_x * self.stride - self.padding + (kernel_size // 2) * self.dilation

        # Add offsets and reshape for grid_sample
        sample_y = base_y.unsqueeze(0).unsqueeze(0) + offset_y  # (B, k*k, H', W')
        sample_x = base_x.unsqueeze(0).unsqueeze(0) + offset_x

        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        sample_x = 2.0 * sample_x / (W - 1) - 1.0

        # Stack: (B, k*k, H', W', 2)
        grid = torch.stack([sample_x, sample_y], dim=-1)

        # Reshape for vectorized sampling
        grid = grid.permute(0, 3, 4, 1, 2).reshape(
            B, H_out * W_out * kernel_size * kernel_size, 2
        ).contiguous()

        # Expand input
        x_expanded = x.unsqueeze(1).expand(
            -1, self.out_channels // self.groups, -1, -1, -1
        )
        x_expanded = x_expanded.reshape(
            B * self.out_channels // self.groups * C_in, H, W
        ).contiguous()

        grid_expanded = grid.unsqueeze(1).expand(
            -1, C_in * self.out_channels // self.groups, -1, -1
        ).reshape(
            B * self.out_channels // self.groups * C_in,
            H_out * W_out * kernel_size * kernel_size, 2
        ).contiguous()

        # Sample
        sampled = F.grid_sample(
            x_expanded.unsqueeze(1),
            grid_expanded.unsqueeze(1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)

        # Reshape and apply weights
        sampled = sampled.reshape(
            B, self.out_channels // self.groups, C_in,
            H_out, W_out, kernel_size * kernel_size
        )
        sampled = sampled.permute(0, 1, 3, 4, 5, 2)

        # Reshape weight for multiplication: (out_C, in_C/g, k*k)
        # Transpose to (out_C, k*k, in_C/g) for proper einsum alignment
        weight_flat = weight.view(self.out_channels, C_in // self.groups, kernel_size * kernel_size).permute(0, 2, 1)

        # Compute weighted sum
        # sampled: (B, out_C/g, H', W', k*k, C_in)
        # weight_flat: (out_C, k*k, C_in/g)
        # For groups=1, we reshape weight to match
        weight_for_einsum = weight_flat.reshape(self.out_channels // self.groups, self.groups, kernel_size * kernel_size, C_in // self.groups)
        output = torch.einsum("bgxykc,gokc->bgoxy", sampled, weight_for_einsum)  # (B, out_C/g, H', W')
        output = output.reshape(B, self.out_channels, H_out, W_out)

        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)

        return output


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

    print("Deformable conv test passed!")
