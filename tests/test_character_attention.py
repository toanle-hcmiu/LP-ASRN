"""
Unit tests for Multi-Scale Character Attention module.

Tests the following components:
1. CharacterRegionDetector - Character region detection
2. MultiScaleCharacterAttention - Multi-scale character attention
3. CharacterAwareAttentionBlock - Character-aware attention
"""

import pytest
import torch

from src.models.character_attention import (
    CharacterRegionDetector,
    MultiScaleCharacterAttention,
    CharacterAwareAttentionBlock,
    AdaptiveCharacterAttention,
)


class TestCharacterRegionDetector:
    """Tests for CharacterRegionDetector."""

    def test_output_shape(self):
        """Detector should produce correct output shapes."""
        batch_size = 2
        in_channels = 64
        detector = CharacterRegionDetector(in_channels, num_prototypes=36)

        x = torch.randn(batch_size, in_channels, 32, 64)
        region_mask, proto_scores = detector(x)

        assert region_mask.shape == (batch_size, 1, 32, 64), \
            f"Region mask should be ({batch_size}, 1, 32, 64), got {region_mask.shape}"
        assert proto_scores.shape == (batch_size, 36, 32, 64), \
            f"Proto scores should be ({batch_size}, 36, 32, 64), got {proto_scores.shape}"

    def test_region_mask_range(self):
        """Region mask should be in [0, 1] (sigmoid output)."""
        detector = CharacterRegionDetector(64, num_prototypes=36)

        x = torch.randn(2, 64, 32, 64)
        region_mask, _ = detector(x)

        assert (region_mask >= 0).all() and (region_mask <= 1).all(), \
            "Region mask should be in [0, 1]"

    def test_gradient_flow(self):
        """Gradients should flow through detector."""
        detector = CharacterRegionDetector(64, num_prototypes=36)

        x = torch.randn(2, 64, 32, 64, requires_grad=True)
        region_mask, proto_scores = detector(x)

        loss = region_mask.sum() + proto_scores.sum()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"

    def test_prototypes_are_learned(self):
        """Prototypes should be learnable parameters."""
        detector = CharacterRegionDetector(64, num_prototypes=36)

        assert isinstance(detector.prototypes, torch.Parameter), \
            "Prototypes should be a Parameter"

        # Check that prototypes are updated during training
        optimizer = torch.optim.SGD([detector.prototypes], lr=0.01)

        x = torch.randn(2, 64, 32, 64)
        region_mask, _ = detector(x)
        loss = region_mask.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Prototypes should have been updated
        assert detector.prototypes.grad is not None, "Prototypes should have gradients"


class TestMultiScaleCharacterAttention:
    """Tests for MultiScaleCharacterAttention."""

    def test_output_shape(self):
        """MSCA should preserve input shape."""
        batch_size = 2
        in_channels = 64
        msca = MultiScaleCharacterAttention(
            in_channels,
            scales=(1.0, 0.5, 0.25),
        )

        x = torch.randn(batch_size, in_channels, 32, 64)
        output = msca(x)

        assert output.shape == x.shape, \
            f"Output shape {output.shape} should match input shape {x.shape}"

    def test_multi_scale_processing(self):
        """Features should be processed at multiple scales."""
        batch_size = 2
        in_channels = 64
        scales = (1.0, 0.5, 0.25)

        msca = MultiScaleCharacterAttention(
            in_channels,
            scales=scales,
        )

        x = torch.randn(batch_size, in_channels, 32, 64)
        output = msca(x)

        # Check that output is different from input (features were processed)
        assert not torch.allclose(output, x), "Output should differ from input"

    def test_residual_connection(self):
        """Output should include residual connection."""
        msca = MultiScaleCharacterAttention(64, scales=(1.0, 0.5))

        x = torch.randn(2, 64, 32, 64)

        # Zero out all parameters to test residual
        for param in msca.parameters():
            param.data.zero_()

        output = msca(x)

        # With zeroed parameters, output should equal input (pure residual)
        assert torch.allclose(output, x, atol=1e-5), \
            "With zeroed parameters, output should equal input (residual only)"

    def test_gradient_flow(self):
        """Gradients should flow correctly through MSCA."""
        msca = MultiScaleCharacterAttention(64, scales=(1.0, 0.5))

        x = torch.randn(2, 64, 32, 64, requires_grad=True)
        output = msca(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"


class TestCharacterAwareAttentionBlock:
    """Tests for CharacterAwareAttentionBlock."""

    def test_output_shape(self):
        """Character-aware attention block should preserve input shape."""
        batch_size = 2
        in_channels = 64

        block = CharacterAwareAttentionBlock(in_channels)
        x = torch.randn(batch_size, in_channels, 32, 64)
        output = block(x)

        assert output.shape == x.shape, \
            f"Output shape {output.shape} should match input shape {x.shape}"

    def test_residual_connection(self):
        """Output should include residual connection."""
        block = CharacterAwareAttentionBlock(64)

        x = torch.randn(2, 64, 32, 64)

        # Zero out all parameters
        for param in block.parameters():
            param.data.zero_()

        output = block(x)

        # With zeroed parameters, output should equal input
        assert torch.allclose(output, x, atol=1e-5), \
            "With zeroed parameters, output should equal input"


class TestAdaptiveCharacterAttention:
    """Tests for AdaptiveCharacterAttention."""

    def test_output_shape(self):
        """Adaptive character attention should preserve input shape."""
        batch_size = 2
        in_channels = 64

        aca = AdaptiveCharacterAttention(in_channels, num_scales=3)
        x = torch.randn(batch_size, in_channels, 32, 64)
        output = aca(x)

        assert output.shape == x.shape, \
            f"Output shape {output.shape} should match input shape {x.shape}"

    def test_scale_selection(self):
        """Scale selector should produce valid probability distribution."""
        batch_size = 4
        in_channels = 64
        num_scales = 3

        aca = AdaptiveCharacterAttention(in_channels, num_scales=num_scales)
        x = torch.randn(batch_size, in_channels, 32, 64)

        # Get scale weights
        scale_weights = aca.scale_selector(x)

        # Check shape
        assert scale_weights.shape == (batch_size, num_scales, 1, 1), \
            f"Scale weights shape should be ({batch_size}, {num_scales}, 1, 1)"

        # Check that weights sum to 1 (softmax output)
        sums = scale_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Scale weights should sum to 1"

    def test_gradient_flow(self):
        """Gradients should flow correctly."""
        aca = AdaptiveCharacterAttention(64, num_scales=3)

        x = torch.randn(2, 64, 32, 64, requires_grad=True)
        output = aca(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input should have gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
