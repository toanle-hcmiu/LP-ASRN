"""
Layout and Character Oriented Focal Loss (LCOFL)

From: Nascimento et al. "Enhancing License Plate Super-Resolution:
A Layout-Aware and Character-Driven Approach" (2024)

LCOFL consists of three components:
1. Classification Loss (L_C) - Weighted cross-entropy based on OCR predictions
2. LP Layout Penalty (L_P) - Penalizes digit/letter position mismatches
3. SSIM Loss (L_S) - Structural similarity for image quality

The weights for classification loss are dynamically updated based on
character confusion during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import defaultdict


class ClassificationLoss(nn.Module):
    """
    Classification Loss component of LCOFL.

    Weighted cross-entropy loss where weights are dynamically updated
    based on character confusion frequency.

    Formula: L_C = -(1/K) * Σ w_k * log(p(y_GT_k | x_SR))
    """

    def __init__(
        self,
        vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        alpha: float = 0.1,
        beta: float = 1.0,
    ):
        """
        Initialize Classification Loss.

        Args:
            vocab: Character vocabulary for OCR
            alpha: Weight increment for confused characters
            beta: Weight increment for layout violations
        """
        super().__init__()

        self.vocab = vocab
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(vocab)}
        self.num_classes = len(vocab)
        self.alpha = alpha
        self.beta = beta

        # Initialize weights (all 1 initially)
        self.register_buffer("weights", torch.ones(self.num_classes))

        # For tracking confusion
        self.confusion_matrix: Optional[torch.Tensor] = None

    def update_weights(self, confusion_matrix: torch.Tensor):
        """
        Update weights based on confusion matrix.

        Args:
            confusion_matrix: Matrix of shape (C, C) where confusion_matrix[i, j]
                            indicates how often character i was confused with j
        """
        # Get diagonal (correct predictions)
        correct = torch.diag(confusion_matrix)

        # For each character, add alpha for each confusion
        weight_increments = confusion_matrix.sum(dim=1) - correct
        self.weights = 1.0 + self.alpha * weight_increments

    def get_character_pairs(
        self, pred_text: str, gt_text: str
    ) -> List[Tuple[str, str]]:
        """
        Get pairs of predicted and ground truth characters.

        Args:
            pred_text: Predicted text
            gt_text: Ground truth text

        Returns:
            List of (pred_char, gt_char) tuples
        """
        pairs = []
        min_len = min(len(pred_text), len(gt_text))
        for i in range(min_len):
            pairs.append((pred_text[i], gt_text[i]))
        return pairs

    def compute_confusion_from_batch(
        self, pred_texts: List[str], gt_texts: List[str]
    ) -> torch.Tensor:
        """
        Compute confusion matrix from a batch of predictions.

        Args:
            pred_texts: List of predicted texts
            gt_texts: List of ground truth texts

        Returns:
            Confusion matrix of shape (C, C)
        """
        confusion = torch.zeros(self.num_classes, self.num_classes)

        for pred_text, gt_text in zip(pred_texts, gt_texts):
            for p_char, g_char in self.get_character_pairs(pred_text, gt_text):
                if p_char in self.char_to_idx and g_char in self.char_to_idx:
                    p_idx = self.char_to_idx[p_char]
                    g_idx = self.char_to_idx[g_char]
                    confusion[g_idx, p_idx] += 1

        return confusion

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_texts: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute classification loss.

        Args:
            pred_logits: OCR predictions of shape (B, K, C) where K is max length,
                        C is vocab size. Should be log probabilities.
            gt_texts: Ground truth texts

        Returns:
            Loss tensor and info dict
        """
        B, K, C = pred_logits.shape

        # Check if vocab size matches (e.g., external pretrained model outputs 94 chars, but our vocab is 36)
        if C != len(self.char_to_idx):
            # Vocab size mismatch - skip classification loss and rely on CTC loss only
            # Fixed: The previous implementation used predicted indices instead of ground truth,
            # creating a self-reinforcing loss that rewarded confidence in wrong predictions.
            # Proper cross-entropy with vocab mapping would require complex character mapping.
            # For now, we skip this loss component and rely on CTC loss for training.
            zero_loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=False)
            return zero_loss, {
                "classification_loss": zero_loss,
                "vocab_mismatch": True,
                "skipped": True
            }

        # Encode ground truth texts
        targets = []
        for text in gt_texts:
            encoded = torch.zeros(K, dtype=torch.long)
            for i, char in enumerate(text):
                if i >= K:
                    break
                if char in self.char_to_idx:
                    encoded[i] = self.char_to_idx[char]
                else:
                    # Use padding index for unknown characters
                    encoded[i] = C - 1  # Last index as padding
            targets.append(encoded)

        targets = torch.stack(targets).to(pred_logits.device)  # (B, K)

        # Reshape for cross-entropy
        pred_logits_flat = pred_logits.reshape(-1, C)  # (B*K, C)
        targets_flat = targets.reshape(-1)  # (B*K,)

        # Compute weighted cross-entropy
        # Filter out padding (index C-1 when char not in vocab)
        mask = targets_flat < C - 1
        if mask.sum() > 0:
            loss = F.cross_entropy(
                pred_logits_flat[mask],
                targets_flat[mask],
                weight=self.weights.to(pred_logits.device),
                reduction="mean",
            )
        else:
            loss = torch.tensor(0.0, device=pred_logits.device)

        return loss, {"classification_loss": loss}


class LayoutPenalty(nn.Module):
    """
    LP Layout Penalty component of LCOFL.

    Penalizes when digits are reconstructed as letters or vice versa,
    violating the expected license plate layout pattern.

    Formula: L_P = Σ [D(pred_i) * A(GT_i) + A(pred_i) * D(GT_i)]
    where D(c) = β if digit, A(c) = β if letter
    """

    def __init__(self, beta: float = 1.0):
        """
        Initialize Layout Penalty.

        Args:
            beta: Penalty value for each layout violation
        """
        super().__init__()
        self.beta = beta

    def is_digit(self, char: str) -> bool:
        """Check if character is a digit."""
        return char.isdigit()

    def is_letter(self, char: str) -> bool:
        """Check if character is a letter."""
        return char.isalpha()

    def forward(
        self, pred_texts: List[str], gt_texts: List[str], device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute layout penalty.

        Args:
            pred_texts: List of predicted texts
            gt_texts: List of ground truth texts
            device: Device to create the penalty tensor on

        Returns:
            Penalty tensor and info dict
        """
        total_penalty = 0.0
        total_violations = 0
        total_chars = 0

        for pred_text, gt_text in zip(pred_texts, gt_texts):
            min_len = min(len(pred_text), len(gt_text))
            for i in range(min_len):
                pred_char = pred_text[i]
                gt_char = gt_text[i]

                total_chars += 1

                # Check if pred is digit but GT is letter
                if self.is_digit(pred_char) and self.is_letter(gt_char):
                    total_penalty += self.beta
                    total_violations += 1

                # Check if pred is letter but GT is digit
                elif self.is_letter(pred_char) and self.is_digit(gt_char):
                    total_penalty += self.beta
                    total_violations += 1

        # Average over batch - create tensor on specified device
        penalty = torch.tensor(
            total_penalty / len(pred_texts) if pred_texts else 0.0,
            dtype=torch.float32,
            device=device,
        )

        return penalty, {
            "layout_penalty": penalty,
            "layout_violations": total_violations,
            "total_chars": total_chars,
        }


class LCOFL(nn.Module):
    """
    Layout and Character Oriented Focal Loss (LCOFL-EC).

    Combines:
    1. Classification Loss - for character recognition
    2. Layout Penalty - for layout consistency
    3. Optional SSIM Loss - for structural similarity
    4. Optional Embedding Consistency Loss - for perceptual similarity

    This loss function is designed specifically for license plate
    super-resolution, guiding the network to produce images that
    are recognizable by OCR systems.
    """

    def __init__(
        self,
        vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        lambda_layout: float = 0.5,
        lambda_ssim: float = 0.0,
        lambda_embed: float = 0.0,
        embedding_dim: int = 128,
        embed_margin: float = 2.0,
        use_lightweight_embedder: bool = False,
        alpha: float = 0.1,
        beta: float = 1.0,
    ):
        """
        Initialize LCOFL.

        Args:
            vocab: Character vocabulary for OCR
            lambda_layout: Weight for layout penalty
            lambda_ssim: Weight for SSIM loss
            lambda_embed: Weight for embedding consistency loss (will be warmed up)
            embedding_dim: Dimension of embedding vectors
            embed_margin: Margin for embedding contrastive loss
            use_lightweight_embedder: Use lightweight embedder instead of ResNet
            alpha: Weight increment for confused characters
            beta: Penalty value for layout violations
        """
        super().__init__()

        self.lambda_layout = lambda_layout
        self.lambda_ssim = lambda_ssim
        self.lambda_embed = lambda_embed
        self.embedding_dim = embedding_dim

        self.classification_loss = ClassificationLoss(vocab, alpha, beta)
        self.layout_penalty = LayoutPenalty(beta)

        # Initialize embedding loss components if enabled
        if lambda_embed > 0 or embedding_dim > 0:
            from src.models.siamese_embedder import (
                SiameseEmbedder,
                LightweightSiameseEmbedder
            )
            from src.losses.embedding_loss import EmbeddingConsistencyLoss

            # Choose embedder type based on flag
            if use_lightweight_embedder:
                self.embedder = LightweightSiameseEmbedder(
                    embedding_dim=embedding_dim
                )
            else:
                self.embedder = SiameseEmbedder(
                    embedding_dim=embedding_dim
                )

            self.embedding_loss_fn = EmbeddingConsistencyLoss(
                margin=embed_margin,
                distance="manhattan"
            )
        else:
            self.embedder = None
            self.embedding_loss_fn = None

    def update_weights(self, confusion_matrix: torch.Tensor):
        """Update classification weights based on confusion matrix."""
        self.classification_loss.update_weights(confusion_matrix)

    def compute_confusion(
        self, pred_texts: List[str], gt_texts: List[str]
    ) -> torch.Tensor:
        """Compute confusion matrix from predictions."""
        return self.classification_loss.compute_confusion_from_batch(
            pred_texts, gt_texts
        )

    def set_embed_weight(self, weight: float):
        """
        Set the embedding loss weight dynamically.

        This is used by the adaptive weight scheduler to gradually
        increase the embedding loss weight during training.

        Args:
            weight: New weight for embedding loss
        """
        self.lambda_embed = weight

    def get_embeddings(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from SR and HR images.

        Args:
            sr_images: Super-resolved images (B, 3, H, W)
            hr_images: Ground truth HR images (B, 3, H, W)

        Returns:
            Tuple of (sr_embeddings, hr_embeddings)
        """
        if self.embedder is None:
            return None, None

        sr_emb = self.embedder(sr_images)
        hr_emb = self.embedder(hr_images)

        return sr_emb, hr_emb

    def forward(
        self,
        sr_images: torch.Tensor,
        hr_images: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_texts: List[str],
        pred_texts: Optional[List[str]] = None,
        lambda_embed_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute LCOFL loss (with optional embedding consistency).

        Args:
            sr_images: Super-resolved images (B, 3, H, W) in range [-1, 1]
            hr_images: Ground truth HR images (B, 3, H, W) in range [-1, 1]
            pred_logits: OCR predictions (B, K, C) where K is max length
            gt_texts: Ground truth texts
            pred_texts: Optional decoded predicted texts for layout penalty
            lambda_embed_override: Override embedding weight for this forward pass

        Returns:
            Total loss and info dict
        """
        info = {}
        losses = []

        # Classification Loss
        cls_loss, cls_info = self.classification_loss(pred_logits, gt_texts)
        losses.append(cls_loss)
        info.update(cls_info)

        # Layout Penalty (requires decoded predictions)
        if pred_texts is not None:
            layout_loss, layout_info = self.layout_penalty(pred_texts, gt_texts, device=sr_images.device)
            weighted_layout = self.lambda_layout * layout_loss
            losses.append(weighted_layout)
            info.update(layout_info)

        # SSIM Loss (optional)
        if self.lambda_ssim > 0:
            ssim_loss = 1.0 - ssim(sr_images, hr_images)
            weighted_ssim = self.lambda_ssim * ssim_loss
            losses.append(weighted_ssim)
            info["ssim_loss"] = ssim_loss

        # Embedding Consistency Loss (optional)
        embed_weight = lambda_embed_override if lambda_embed_override is not None else self.lambda_embed
        if embed_weight > 0 and self.embedder is not None:
            sr_emb, hr_emb = self.get_embeddings(sr_images, hr_images)
            if sr_emb is not None and hr_emb is not None:
                embed_loss, embed_info = self.embedding_loss_fn(sr_emb, hr_emb)
                weighted_embed = embed_weight * embed_loss
                losses.append(weighted_embed)
                info["embedding_loss"] = embed_loss
                info["embedding_distance"] = torch.tensor(embed_info["embedding_distance"])

        # Total loss
        total_loss = torch.stack(losses).sum()

        info["total_loss"] = total_loss

        return total_loss, info


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two tensors.

    Args:
        x: First tensor (B, C, H, W)
        y: Second tensor (B, C, H, W)
        window_size: Size of the Gaussian window
        C1, C2: Stability constants

    Returns:
        SSIM value (scalar)
    """
    # Assume inputs are in [-1, 1], convert to [0, 1]
    x = (x + 1.0) / 2.0
    y = (y + 1.0) / 2.0

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x**2, window_size, stride=1, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y**2, window_size, stride=1, padding=window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    return ssim_map.mean()


if __name__ == "__main__":
    # Test LCOFL
    print("Testing LCOFL...")

    # Create sample predictions
    B, K, C = 4, 7, 36
    pred_logits = torch.randn(B, K, C).log_softmax(dim=-1)

    gt_texts = ["ABC1234", "XYZ5678", "DEF9012", "GHI3456"]
    pred_texts = ["ABC1234", "XZ5678", "DEF9O12", "GHI3456"]  # Some errors

    # Test Classification Loss
    cls_loss = ClassificationLoss()
    loss, info = cls_loss(pred_logits, gt_texts)
    print(f"Classification Loss: {loss.item():.4f}")

    # Test Layout Penalty
    layout_penalty = LayoutPenalty()
    penalty, info = layout_penalty(pred_texts, gt_texts)
    print(f"Layout Penalty: {penalty.item():.4f}")

    # Test LCOFL
    sr_images = torch.randn(B, 3, 32, 64) * 2 - 1
    hr_images = torch.randn(B, 3, 32, 64) * 2 - 1

    lcofl = LCOFL(lambda_layout=0.5, lambda_ssim=0.2)
    total_loss, info = lcofl(
        sr_images, hr_images, pred_logits, gt_texts, pred_texts
    )
    print(f"LCOFL Total Loss: {total_loss.item():.4f}")

    # Test weight update
    confusion = lcofl.compute_confusion(pred_texts, gt_texts)
    print(f"Confusion Matrix shape: {confusion.shape}")
    lcofl.update_weights(confusion)
    print("Weights updated")

    print("LCOFL test passed!")
