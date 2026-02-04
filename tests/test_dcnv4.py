"""
Unit tests for DCNv4 integration.

Tests the following components:
1. DeformableConv2dV4 - DCNv4-based deformable convolution
2. Fallback behavior when DCNv4 is unavailable
3. Integration with attention modules
"""

import pytest
import torch

from src.models.deform_conv import (
    DeformableConv2d,
    DeformableConv2dV4,
    DCNV4_AVAILABLE,
)
from src.models.attention import EnhancedAttentionModule


class TestDeformableConv2d:
    """Tests for the original DCNv3 implementation."""

    def test_output_shape(self):
        """DCNv3 should preserve spatial dimensions."""
        batch_size = 2
        in_channels = 64
        out_channels = 64

        dcn = DeformableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        x = torch.randn(batch_size, in_channels, 32, 64)

        output = dcn(x)

        assert output.shape == (batch_size, out_channels, 32, 64), \
            f"Output shape should be ({batch_size}, {out_channels}, 32, 64), got {output.shape}"

    def test_gradient_flow(self):
        """Gradients should flow through DCNv3."""
        dcn = DeformableConv2d(64, 64, kernel_size=3, padding=1)

        x = torch.randn(2, 64, 32, 64, requires_grad=True)
        output = dcn(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"


class TestDeformableConv2dV4:
    """Tests for DCNv4-based deformable convolution."""

    def test_fallback_to_dcnv3(self):
        """Should fallback to DCNv3 when DCNv4 unavailable."""
        batch_size = 2
        in_channels = 64
        out_channels = 64

        dcnv4 = DeformableConv2dV4(in_channels, out_channels, kernel_size=3, padding=1)
        x = torch.randn(batch_size, in_channels, 32, 64)

        output = dcnv4(x)

        assert output.shape == (batch_size, out_channels, 32, 64), \
            f"Output shape should be ({batch_size}, {out_channels}, 32, 64), got {output.shape}"

        # Check that the correct implementation is being used
        if DCNV4_AVAILABLE:
            assert dcnv4.use_dcnv4, "Should use DCNv4 when available"
        else:
            assert not dcnv4.use_dcnv4, "Should use fallback when DCNv4 unavailable"

    def test_gradient_flow(self):
        """Gradients should flow correctly through DCNv4."""
        dcnv4 = DeformableConv2dV4(64, 64, kernel_size=3, padding=1)

        x = torch.randn(2, 64, 32, 64, requires_grad=True)
        output = dcnv4(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"

    def test_different_groups(self):
        """DCNv4 should work with different group settings."""
        for groups in [1, 2, 4, 8]:
            dcnv4 = DeformableConv2dV4(
                64, 64,
                kernel_size=3,
                padding=1,
                groups=groups,
            )

            x = torch.randn(2, 64, 32, 64)
            output = dcnv4(x)

            assert output.shape == (2, 64, 32, 64), \
                f"Output shape should be (2, 64, 32, 64) for groups={groups}"

    def test_different_kernel_sizes(self):
        """DCNv4 should work with different kernel sizes."""
        for kernel_size in [3, 5, 7]:
            padding = kernel_size // 2

            dcnv4 = DeformableConv2dV4(
                64, 64,
                kernel_size=kernel_size,
                padding=padding,
            )

            x = torch.randn(2, 64, 32, 64)
            output = dcnv4(x)

            assert output.shape == (2, 64, 32, 64), \
                f"Output shape should be (2, 64, 32, 64) for kernel_size={kernel_size}"


class TestEnhancedAttentionWithDCNv4:
    """Tests for EnhancedAttentionModule with DCNv4."""

    def test_dcnv4_integration(self):
        """EnhancedAttentionModule should use DCNv4 when requested."""
        batch_size = 2
        in_channels = 64

        eam_dcnv4 = EnhancedAttentionModule(
            in_channels,
            use_deformable=True,
            use_dcnv4=True,
        )

        eam_dcnv3 = EnhancedAttentionModule(
            in_channels,
            use_deformable=True,
            use_dcnv4=False,
        )

        x = torch.randn(batch_size, in_channels, 32, 64)

        # Both should produce outputs of the same shape
        output_dcnv4 = eam_dcnv4(x)
        output_dcnv3 = eam_dcnv3(x)

        assert output_dcnv4.shape == (batch_size, in_channels, 32, 64), \
            "DCNv4 output should have correct shape"
        assert output_dcnv3.shape == (batch_size, in_channels, 32, 64), \
            "DCNv3 output should have correct shape"

        # Check that DCNv4 flag is set correctly
        if DCNV4_AVAILABLE:
            assert eam_dcnv4.use_dcnv4, "use_dcnv4 flag should be True when DCNv4 is available"
        else:
            assert not eam_dcnv4.use_dcnv4, "use_dcnv4 flag should be False when DCNv4 is unavailable"

    def test_gradient_flow_with_dcnv4(self):
        """Gradients should flow correctly with DCNv4 in attention."""
        eam = EnhancedAttentionModule(
            64,
            use_deformable=True,
            use_dcnv4=True,
        )

        x = torch.randn(2, 64, 32, 64, requires_grad=True)
        output = eam(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"

    def test_output_without_deformable(self):
        """Module should work without deformable convolutions."""
        eam = EnhancedAttentionModule(
            64,
            use_deformable=False,
        )

        x = torch.randn(2, 64, 32, 64)
        output = eam(x)

        assert output.shape == (2, 64, 32, 64), \
            "Output should have correct shape without deformable conv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
