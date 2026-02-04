"""
Embedding Consistency Loss for LP-SR (LCOFL-EC)

Based on: Sendjasni et al., "Embedding Consistency Loss for Enhanced
Super-Resolution" (2025)

Minimizes the distance between SR and HR embeddings in a learned
feature space, encouraging perceptually similar reconstructions.

Formula: L_EC = max(m - D(V_SR, V_HR), 0)²
where:
- D is Manhattan distance (ℓ₁-norm)
- m is the margin hyperparameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class EmbeddingConsistencyLoss(nn.Module):
    """
    Embedding Consistency Loss for LP-SR.

    Minimizes the distance between SR and HR embeddings in a learned
    feature space, encouraging perceptually similar reconstructions.

    Uses a contrastive loss formulation with a margin to prevent
    collapse to trivial solutions.
    """

    def __init__(
        self,
        margin: float = 2.0,
        distance: str = "manhattan",
        reduction: str = "mean",
    ):
        """
        Initialize Embedding Consistency Loss.

        Args:
            margin: Margin for contrastive loss. Higher values push
                   embeddings further apart for dissimilar images.
            distance: Distance metric ('manhattan' or 'euclidean')
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()

        self.margin = margin
        self.distance = distance
        self.reduction = reduction

    def forward(
        self,
        sr_embedding: torch.Tensor,
        hr_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute embedding consistency loss.

        Args:
            sr_embedding: (B, D) embeddings from SR images (L2-normalized)
            hr_embedding: (B, D) embeddings from HR images (L2-normalized)

        Returns:
            Loss tensor and info dict with distance, loss components
        """
        # Compute distance between embeddings
        if self.distance == "manhattan":
            dist = torch.sum(torch.abs(sr_embedding - hr_embedding), dim=1)
        elif self.distance == "euclidean":
            dist = torch.norm(sr_embedding - hr_embedding, p=2, dim=1)
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")

        # Contrastive loss: penalize when distance > margin
        # L_EC = max(margin - distance, 0)²
        # This encourages embeddings to be within margin distance
        loss = torch.pow(torch.clamp(self.margin - dist, min=0), 2)

        # Apply reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        # Info dict for logging
        info = {
            "embedding_distance": dist.mean().item(),
            "embedding_loss": loss.item() if self.reduction != "none" else loss.mean().item(),
            "margin": self.margin,
        }

        return loss, info


class TripletEmbeddingLoss(nn.Module):
    """
    Triplet Embedding Loss for LP-SR.

    Uses a triplet loss formulation with anchor (SR), positive (HR),
    and optionally negative (distractor) samples.

    Formula: L_triplet = max(D(anchor, positive) - D(anchor, negative) + margin, 0)
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance: str = "manhattan",
    ):
        """
        Initialize Triplet Embedding Loss.

        Args:
            margin: Margin for triplet loss
            distance: Distance metric ('manhattan' or 'euclidean')
        """
        super().__init__()

        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute triplet loss.

        Args:
            anchor: (B, D) anchor embeddings (SR images)
            positive: (B, D) positive embeddings (HR images)
            negative: (B, D) negative embeddings (optional, uses batch negatives if None)

        Returns:
            Loss tensor and info dict
        """
        # Compute distances
        if self.distance == "manhattan":
            pos_dist = torch.sum(torch.abs(anchor - positive), dim=1)
        elif self.distance == "euclidean":
            pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")

        if negative is not None:
            # Use provided negatives
            if self.distance == "manhattan":
                neg_dist = torch.sum(torch.abs(anchor - negative), dim=1)
            else:
                neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        else:
            # Use batch negatives (hard negative mining within batch)
            # For each anchor, find the hardest negative in the batch
            batch_size = anchor.shape[0]

            # Compute all pairwise distances
            if self.distance == "manhattan":
                # (B, B) matrix of distances
                all_dists = torch.cdist(anchor, anchor, p=1)
            else:
                all_dists = torch.cdist(anchor, anchor, p=2)

            # Mask out self-distances and positive pairs
            # Assuming batch is ordered (anchor[i] matches positive[i])
            mask = torch.eye(batch_size, device=anchor.device).bool()

            # Set negative distances (diagonal) to infinity
            all_dists = all_dists.masked_fill(mask, float('inf'))

            # Find minimum distance (hardest negative) for each anchor
            neg_dist, _ = all_dists.min(dim=1)

        # Triplet loss: max(pos_dist - neg_dist + margin, 0)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)

        info = {
            "triplet_pos_distance": pos_dist.mean().item(),
            "triplet_neg_distance": neg_dist.mean().item(),
            "triplet_loss": loss.mean().item(),
        }

        return loss.mean(), info


class CosineEmbeddingLoss(nn.Module):
    """
    Cosine Embedding Loss for LP-SR.

    Uses cosine similarity instead of distance.

    Formula: L_cosine = 1 - cosine_similarity(V_SR, V_HR)
    """

    def __init__(
        self,
        margin: float = 0.0,
    ):
        """
        Initialize Cosine Embedding Loss.

        Args:
            margin: Margin for loss (loss is max(margin - cos_sim, 0))
                   Set to 0 to maximize similarity directly
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        sr_embedding: torch.Tensor,
        hr_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cosine embedding loss.

        Args:
            sr_embedding: (B, D) embeddings from SR images
            hr_embedding: (B, D) embeddings from HR images

        Returns:
            Loss tensor and info dict
        """
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(sr_embedding, hr_embedding, dim=1)

        # Loss: 1 - similarity (or margin - similarity)
        loss = torch.clamp(self.margin - cos_sim, min=0).mean()

        info = {
            "cosine_similarity": cos_sim.mean().item(),
            "cosine_loss": loss.item(),
        }

        return loss, info


if __name__ == "__main__":
    # Test the embedding consistency loss
    print("Testing Embedding Consistency Loss...")

    batch_size = 4
    embedding_dim = 128

    # Create sample embeddings (L2-normalized)
    sr_embed = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    hr_embed = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)

    # Test EmbeddingConsistencyLoss
    loss_fn = EmbeddingConsistencyLoss(margin=2.0, distance="manhattan")
    loss, info = loss_fn(sr_embed, hr_embed)
    print(f"Embedding Loss: {loss.item():.4f}")
    print(f"Embedding Distance: {info['embedding_distance']:.4f}")

    # Test with identical embeddings (should be zero)
    identical_loss, identical_info = loss_fn(hr_embed, hr_embed)
    print(f"Identical embeddings loss: {identical_loss.item():.4f}")

    # Test TripletEmbeddingLoss
    print("\nTesting Triplet Embedding Loss...")
    triplet_fn = TripletEmbeddingLoss(margin=1.0)
    negative = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    triplet_loss, triplet_info = triplet_fn(sr_embed, hr_embed, negative)
    print(f"Triplet Loss: {triplet_loss.item():.4f}")

    # Test TripletEmbeddingLoss with batch negatives
    triplet_loss_bn, triplet_info_bn = triplet_fn(sr_embed, hr_embed, negative=None)
    print(f"Triplet Loss (batch negatives): {triplet_loss_bn.item():.4f}")

    # Test CosineEmbeddingLoss
    print("\nTesting Cosine Embedding Loss...")
    cosine_fn = CosineEmbeddingLoss(margin=0.0)
    cosine_loss, cosine_info = cosine_fn(sr_embed, hr_embed)
    print(f"Cosine Loss: {cosine_loss.item():.4f}")
    print(f"Cosine Similarity: {cosine_info['cosine_similarity']:.4f}")

    print("\nEmbedding Consistency Loss test passed!")
