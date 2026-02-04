"""
Unit tests for Embedding Consistency Loss and Siamese Embedder.

Tests the following components:
1. SiameseEmbedder - Embedding extraction from images
2. EmbeddingConsistencyLoss - Contrastive loss between embeddings
3. AdaptiveWeightScheduler - Weight warm-up scheduling
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.siamese_embedder import (
    SiameseEmbedder,
    LightweightSiameseEmbedder,
)
from src.losses.embedding_loss import (
    EmbeddingConsistencyLoss,
    TripletEmbeddingLoss,
    CosineEmbeddingLoss,
)
from src.utils.adaptive_scheduler import (
    AdaptiveWeightScheduler,
    MultiComponentScheduler,
)


class TestSiameseEmbedder:
    """Tests for SiameseEmbedder."""

    def test_embedding_dimensions(self):
        """Embeddings should have correct dimensions."""
        batch_size = 4
        embedder = SiameseEmbedder(embedding_dim=128)
        x = torch.randn(batch_size, 3, 64, 128) * 2 - 1

        embeddings = embedder(x)

        assert embeddings.shape == (batch_size, 128), \
            f"Expected shape ({batch_size}, 128), got {embeddings.shape}"

    def test_l2_normalization(self):
        """Embeddings should be L2 normalized."""
        embedder = SiameseEmbedder(embedding_dim=128)
        x = torch.randn(2, 3, 64, 128) * 2 - 1

        embeddings = embedder(x)

        norms = torch.norm(embeddings, p=2, dim=1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "Embeddings should be L2 normalized (norm ≈ 1.0)"

    def test_frozen_backbone(self):
        """Backbone parameters should not require gradients."""
        embedder = SiameseEmbedder(embedding_dim=128)

        for param in embedder.backbone.parameters():
            assert not param.requires_grad, "Backbone should be frozen"

    def test_identical_images_zero_distance(self):
        """Distance between identical images should be zero."""
        embedder = SiameseEmbedder(embedding_dim=128)
        x = torch.randn(2, 3, 64, 128) * 2 - 1

        distance = embedder.compute_distance(x, x, "manhattan")

        assert torch.allclose(distance, torch.zeros_like(distance), atol=1e-5), \
            "Distance between identical images should be ~0"

    def test_gradient_flow(self):
        """Gradients should flow correctly through loss."""
        embedder = SiameseEmbedder(embedding_dim=128)
        loss_fn = EmbeddingConsistencyLoss(margin=2.0)

        sr = torch.randn(2, 3, 64, 128) * 2 - 1
        hr = torch.randn(2, 3, 64, 128) * 2 - 1

        sr_emb = embedder(sr)
        hr_emb = embedder(hr)

        loss, _ = loss_fn(sr_emb, hr_emb)

        loss.backward()

        # Check that projection head has gradients
        for param in embedder.projection.parameters():
            assert param.grad is not None, "Projection head should have gradients"


class TestLightweightSiameseEmbedder:
    """Tests for LightweightSiameseEmbedder."""

    def test_embedding_dimensions(self):
        """Embeddings should have correct dimensions."""
        batch_size = 4
        embedder = LightweightSiameseEmbedder(embedding_dim=128)
        x = torch.randn(batch_size, 3, 64, 128) * 2 - 1

        embeddings = embedder(x)

        assert embeddings.shape == (batch_size, 128), \
            f"Expected shape ({batch_size}, 128), got {embeddings.shape}"

    def test_l2_normalization(self):
        """Embeddings should be L2 normalized."""
        embedder = LightweightSiameseEmbedder(embedding_dim=128)
        x = torch.randn(2, 3, 64, 128) * 2 - 1

        embeddings = embedder(x)

        norms = torch.norm(embeddings, p=2, dim=1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "Embeddings should be L2 normalized"


class TestEmbeddingConsistencyLoss:
    """Tests for EmbeddingConsistencyLoss."""

    def test_loss_zero_for_identical_embeddings(self):
        """Loss should be zero when embeddings are identical."""
        loss_fn = EmbeddingConsistencyLoss(margin=2.0)

        emb = F.normalize(torch.randn(4, 128), p=2, dim=1)
        loss, info = loss_fn(emb, emb)

        assert loss.item() < 1e-5, "Loss should be ~0 for identical embeddings"

    def test_loss_positive_for_different_embeddings(self):
        """Loss should be positive when embeddings differ."""
        loss_fn = EmbeddingConsistencyLoss(margin=2.0)

        sr_emb = F.normalize(torch.randn(4, 128), p=2, dim=1)
        hr_emb = F.normalize(torch.randn(4, 128), p=2, dim=1)

        loss, info = loss_fn(sr_emb, hr_emb)

        assert loss.item() > 0, "Loss should be positive for different embeddings"

    def test_gradient_flow(self):
        """Gradients should flow correctly through loss."""
        loss_fn = EmbeddingConsistencyLoss(margin=2.0)

        sr_emb = F.normalize(torch.randn(4, 128, requires_grad=True), p=2, dim=1)
        hr_emb = F.normalize(torch.randn(4, 128, requires_grad=True), p=2, dim=1)

        loss, _ = loss_fn(sr_emb, hr_emb)

        loss.backward()

        assert sr_emb.grad is not None, "sr_emb should have gradients"
        assert hr_emb.grad is not None, "hr_emb should have gradients"

    def test_margin_behavior(self):
        """Loss should respect margin parameter."""
        loss_fn_small = EmbeddingConsistencyLoss(margin=1.0)
        loss_fn_large = EmbeddingConsistencyLoss(margin=5.0)

        sr_emb = F.normalize(torch.randn(4, 128), p=2, dim=1)
        hr_emb = F.normalize(torch.randn(4, 128), p=2, dim=1)

        loss_small, _ = loss_fn_small(sr_emb, hr_emb)
        loss_large, _ = loss_fn_large(sr_emb, hr_emb)

        # Larger margin should generally result in larger loss
        # (except when distance is 0)
        assert loss_large.item() >= loss_small.item() - 1e-5, \
            "Larger margin should result in larger or equal loss"


class TestTripletEmbeddingLoss:
    """Tests for TripletEmbeddingLoss."""

    def test_loss_shape(self):
        """Triplet loss should return correct shape."""
        loss_fn = TripletEmbeddingLoss(margin=1.0)

        anchor = F.normalize(torch.randn(4, 128), p=2, dim=1)
        positive = F.normalize(torch.randn(4, 128), p=2, dim=1)
        negative = F.normalize(torch.randn(4, 128), p=2, dim=1)

        loss, info = loss_fn(anchor, positive, negative)

        assert loss.ndim == 0, "Loss should be a scalar"

    def test_batch_negative_mining(self):
        """Batch negative mining should work correctly."""
        loss_fn = TripletEmbeddingLoss(margin=1.0)

        anchor = F.normalize(torch.randn(4, 128), p=2, dim=1)
        positive = F.normalize(torch.randn(4, 128), p=2, dim=1)

        # Without providing negatives, should use batch negatives
        loss, info = loss_fn(anchor, positive, negative=None)

        assert loss.item() >= 0, "Loss should be non-negative"


class TestCosineEmbeddingLoss:
    """Tests for CosineEmbeddingLoss."""

    def test_loss_shape(self):
        """Cosine loss should return correct shape."""
        loss_fn = CosineEmbeddingLoss(margin=0.0)

        sr_emb = F.normalize(torch.randn(4, 128), p=2, dim=1)
        hr_emb = F.normalize(torch.randn(4, 128), p=2, dim=1)

        loss, info = loss_fn(sr_emb, hr_emb)

        assert loss.ndim == 0, "Loss should be a scalar"

    def test_cosine_similarity_range(self):
        """Cosine similarity should be in [-1, 1]."""
        loss_fn = CosineEmbeddingLoss(margin=0.0)

        sr_emb = F.normalize(torch.randn(4, 128), p=2, dim=1)
        hr_emb = F.normalize(torch.randn(4, 128), p=2, dim=1)

        loss, info = loss_fn(sr_emb, hr_emb)

        assert -1 <= info["cosine_similarity"] <= 1, \
            "Cosine similarity should be in [-1, 1]"


class TestAdaptiveWeightScheduler:
    """Tests for AdaptiveWeightScheduler."""

    def test_linear_schedule(self):
        """Linear schedule should increase weights linearly."""
        scheduler = AdaptiveWeightScheduler(
            start_weight=0.0,
            target_weight=0.3,
            warmup_epochs=50,
            schedule="linear",
        )

        # Check start and end
        assert scheduler.step(epoch=0) == 0.0, "Start weight should be 0.0"
        assert scheduler.step(epoch=50) == 0.3, "Target weight should be reached"

        # Check midpoint
        mid_weight = scheduler.step(epoch=25)
        assert 0.14 <= mid_weight <= 0.16, "Midpoint should be ~0.15"

    def test_cosine_schedule(self):
        """Cosine schedule should follow cosine curve."""
        scheduler = AdaptiveWeightScheduler(
            start_weight=0.0,
            target_weight=0.3,
            warmup_epochs=50,
            schedule="cosine",
        )

        # Cosine should start slower than linear
        linear_weight = 0.3 * (10 / 50)
        cosine_weight = scheduler.step(epoch=10)

        # Cosine should be slightly higher than linear at early epochs
        # due to the shape of the curve
        assert cosine_weight >= linear_weight - 0.01, \
            "Cosine should follow the expected curve"

    def test_ocr_guided_schedule(self):
        """OCR-guided schedule should adjust based on OCR accuracy."""
        scheduler = AdaptiveWeightScheduler(
            start_weight=0.0,
            target_weight=0.3,
            warmup_epochs=50,
            schedule="ocr_guided",
        )

        # Higher OCR accuracy should lead to faster warm-up
        weight_low_ocr = scheduler.step(epoch=10, ocr_accuracy=0.3)
        weight_high_ocr = scheduler.step(epoch=10, ocr_accuracy=0.7)

        assert weight_high_ocr > weight_low_ocr, \
            "Higher OCR accuracy should accelerate warm-up"

    def test_exponential_schedule(self):
        """Exponential schedule should have slower initial increase."""
        scheduler = AdaptiveWeightScheduler(
            start_weight=0.0,
            target_weight=0.3,
            warmup_epochs=50,
            schedule="exponential",
        )

        # Exponential (squared) should be slower than linear at early epochs
        linear_weight = 0.3 * (10 / 50)
        exp_weight = scheduler.step(epoch=10)

        assert exp_weight < linear_weight, \
            "Exponential schedule should be slower than linear at early epochs"


class TestMultiComponentScheduler:
    """Tests for MultiComponentScheduler."""

    def test_multiple_schedulers(self):
        """Multiple schedulers should work independently."""
        schedulers = MultiComponentScheduler({
            "embedding": {
                "start_weight": 0.0,
                "target_weight": 0.3,
                "warmup_epochs": 50,
                "schedule": "linear",
            },
            "ssim": {
                "start_weight": 0.0,
                "target_weight": 0.2,
                "warmup_epochs": 20,
                "schedule": "cosine",
            },
        })

        weights = schedulers.step(epoch=10)

        assert "embedding" in weights, "Should have embedding weight"
        assert "ssim" in weights, "Should have ssim weight"

        # SSIM should be further along (20 vs 50 epochs)
        assert weights["ssim"] > weights["embedding"], \
            "SSIM scheduler with shorter warmup should be further along"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
