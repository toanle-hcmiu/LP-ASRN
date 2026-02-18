"""
SwinIR Building Blocks for License Plate Super-Resolution

Based on:
- Liang et al. "SwinIR: Image Restoration Using Swin Transformer" (CVPR 2022)
- https://arxiv.org/abs/2109.15272

Key components:
1. WindowAttention - Window-based multi-head self-attention
2. SwinTransformerBlock - Basic Swin block with shifted window attention
3. ResidualSwinTransformerBlock (RSTB) - Residual wrapper for Swin block
4. PatchMerging - Downsample layer for hierarchical features

Adapted for license plate images with small spatial dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self Attention (W-MSA).

    Computes self-attention within local windows for efficiency.
    Uses relative position bias for spatial awareness.

    Args:
        dim: Number of input channels
        window_size: Size of the local window (tuple of ints)
        num_heads: Number of attention heads
        qkv_bias: If True, add bias to QKV projections
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias parameters
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get relative position index for each window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative position bias
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (num_windows*B, window_size*window_size, C)
            mask: Attention mask for shifted window (optional)

        Returns:
            Output tensor of shape (num_windows*B, window_size*window_size, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron with GELU activation.

    Used in Swin Transformer blocks.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """
    Partition feature map into windows.

    Args:
        x: Input tensor of shape (B, C, H, W)
        window_size: Window size (wh, ww)

    Returns:
        Windows of shape (B*num_windows, wh, ww, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int], H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.

    Args:
        windows: Windows of shape (B*num_windows, wh, ww, C)
        window_size: Window size (wh, ww)
        H: Height of original feature map
        W: Width of original feature map

    Returns:
        Feature map of shape (B, C, H, W)
    """
    wh, ww = window_size
    B = int(windows.shape[0] / (H * W / (wh * ww)))
    x = windows.view(B, H // wh, W // ww, wh, ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, H, W)
    return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with shifted window attention.

    Alternates between W-MSA (regular) and SW-MSA (shifted) for
    cross-window connections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Window attention
        self.attn = WindowAttention(
            dim=dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # MLP
        hidden_features = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_features,
            drop=drop,
        )

        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        shortcut = x

        # Reshape to (B, H*W, C) for attention
        x = x.view(B, C, H, W)

        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, (self.window_size, self.window_size))
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Attention
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, (self.window_size, self.window_size), H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        # Residual connection
        x = shortcut + self.drop_path(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).view(B, C, H, W))

        # MLP with residual
        shortcut = x
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x.view(B, C, H, W)
        x = self.mlp(x.permute(0, 2, 3, 1).view(B, H * W, C)).permute(0, 2, 1).view(B, C, H, W)
        x = shortcut + self.drop_path(x)

        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ResidualSwinTransformerBlock(nn.Module):
    """
    Residual Swin Transformer Block (RSTB).

    Contains multiple Swin Transformer Blocks with a residual connection.
    This is the main building block of SwinIR.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        num_blocks: int = 2,  # Number of SwinTransformerBlocks per RSTB
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Create alternating shifted and non-shifted blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
            )
            for i in range(num_blocks)
        ])

        # Conv after transformer blocks (for spatial information)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = x

        # Pass through Swin Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # Convolution for spatial refinement
        x = self.conv(x)

        # Residual connection
        return x + identity


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Downsamples feature maps by concatenating neighboring patches
    and applying a linear projection.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, 2*C, H/2, W/2)
        """
        B, C, H, W = x.shape

        # Reshape for merging
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = torch.permute(x, (0, 2, 4, 1, 3, 5)).contiguous()
        x = x.view(B, H // 2 * W // 2, 4 * C)

        # Normalize and reduce
        x = self.norm(x)
        x = self.reduction(x)

        # Reshape back
        x = x.view(B, H // 2, W // 2, -1).permute(0, 3, 1, 2).contiguous()

        return x


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """
    Truncated normal initialization.

    Args:
        tensor: Tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        a: Lower truncation bound
        b: Upper truncation bound

    Returns:
        Initialized tensor
    """
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


if __name__ == "__main__":
    # Test SwinIR blocks
    print("Testing SwinIR blocks...")

    batch_size = 2
    dim = 96
    H, W = 32, 64

    x = torch.randn(batch_size, dim, H, W)

    # Test WindowAttention
    print("\n1. Testing WindowAttention...")
    win_attn = WindowAttention(dim=dim, window_size=(8, 8), num_heads=6)
    x_windows = window_partition(x, (8, 8))
    x_windows = x_windows.view(-1, 8 * 8, dim)
    attn_out = win_attn(x_windows)
    print(f"   Input shape: {x_windows.shape}")
    print(f"   Output shape: {attn_out.shape}")

    # Test SwinTransformerBlock
    print("\n2. Testing SwinTransformerBlock...")
    swin_block = SwinTransformerBlock(dim=dim, num_heads=6, window_size=8)
    swin_out = swin_block(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {swin_out.shape}")

    # Test ResidualSwinTransformerBlock
    print("\n3. Testing ResidualSwinTransformerBlock...")
    rstb = ResidualSwinTransformerBlock(dim=dim, num_heads=6, window_size=8, num_blocks=2)
    rstb_out = rstb(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {rstb_out.shape}")

    # Count parameters
    rstb_params = sum(p.numel() for p in rstb.parameters())
    print(f"\n   RSTB parameters: {rstb_params:,}")

    print("\nSwinIR blocks test passed!")
