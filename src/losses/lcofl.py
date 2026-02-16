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

        Supports two modes:
        - CTC mode (C = vocab + 1): Uses proper CTC loss for gradient consistency
          with CTC-trained OCR models like SimpleCRNN.
        - Position-aligned mode (C = vocab): Uses weighted cross-entropy for
          attention-based OCR models like PARSeq where logits are position-aligned.

        Args:
            pred_logits: OCR predictions of shape (B, K, C) where K is max length,
                        C is vocab size (may include CTC blank token).
            gt_texts: Ground truth texts

        Returns:
            Loss tensor and info dict
        """
        B, K, C = pred_logits.shape
        num_vocab = len(self.char_to_idx)  # 36

        if C == num_vocab + 1:
            # CTC model: use proper CTC loss for gradient consistency
            return self._ctc_classification_loss(pred_logits, gt_texts, B, K, C)
        elif C == num_vocab:
            # Position-aligned model (e.g., PARSeq): use weighted cross-entropy
            return self._ce_classification_loss(pred_logits, gt_texts, B, K, C)
        else:
            # Large vocab mismatch — can't map reliably
            zero_loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
            return zero_loss, {
                "classification_loss": zero_loss,
                "vocab_mismatch": True,
                "skipped": True
            }

    def _ctc_classification_loss(
        self,
        pred_logits: torch.Tensor,
        gt_texts: List[str],
        B: int, K: int, C: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        CTC-based classification loss. Uses the same CTC loss function
        as OCR pretraining for consistent gradients.
        """
        blank_idx = C - 1  # Blank token is at last index

        # Encode targets as flat index list (CTC format)
        target_indices_list = []
        target_lengths = []
        for text in gt_texts:
            indices = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
            target_indices_list.extend(indices)
            target_lengths.append(len(indices))

        if len(target_indices_list) == 0:
            zero_loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
            return zero_loss, {"classification_loss": zero_loss}

        target_indices = torch.tensor(target_indices_list, dtype=torch.long,
                                       device=pred_logits.device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long,
                                       device=pred_logits.device)
        input_lengths = torch.full((B,), K, dtype=torch.long,
                                    device=pred_logits.device)

        # Log softmax + transpose for CTC: (T, B, C)
        log_probs = F.log_softmax(pred_logits, dim=-1).transpose(0, 1)

        # Apply per-class weights via weighted log probs
        # This preserves the confusion-based weighting from the original design
        weights = self.weights.to(pred_logits.device)
        # Extend weights to include blank token (weight=1.0)
        weights_full = torch.cat([weights, torch.ones(1, device=weights.device)])
        weighted_log_probs = log_probs * weights_full.unsqueeze(0).unsqueeze(0)

        loss = F.ctc_loss(
            weighted_log_probs,
            target_indices,
            input_lengths,
            target_lengths,
            blank=blank_idx,
            zero_infinity=True,
            reduction='mean',
        )

        return loss, {"classification_loss": loss, "loss_mode": "ctc"}

    def _ce_classification_loss(
        self,
        pred_logits: torch.Tensor,
        gt_texts: List[str],
        B: int, K: int, C: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Position-aligned cross-entropy loss for attention-based OCR models.
        """
        # Encode ground truth texts
        targets = torch.full((B, K), -100, dtype=torch.long)
        for b, text in enumerate(gt_texts):
            for i, char in enumerate(text):
                if i >= K:
                    break
                if char in self.char_to_idx:
                    targets[b, i] = self.char_to_idx[char]

        targets = targets.to(pred_logits.device)

        # Reshape for cross-entropy
        pred_logits_flat = pred_logits.reshape(-1, C)
        targets_flat = targets.reshape(-1)

        # Compute weighted cross-entropy
        mask = targets_flat != -100
        if mask.sum() > 0:
            loss = F.cross_entropy(
                pred_logits_flat[mask],
                targets_flat[mask],
                weight=self.weights.to(pred_logits.device),
                reduction="mean",
            )
        else:
            loss = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)

        return loss, {"classification_loss": loss, "loss_mode": "ce"}


class LayoutPenalty(nn.Module):
    """
    LP Layout Penalty component of LCOFL (differentiable version).

    Penalizes when digits are reconstructed as letters or vice versa,
    violating the expected license plate layout pattern.

    Uses OCR logits directly to compute a differentiable penalty:
    - At digit positions in GT: penalize probability mass on letter classes
    - At letter positions in GT: penalize probability mass on digit classes

    This allows gradients to flow back through the OCR to the generator.
    """

    # Indices of digit characters (0-9) and letter characters (A-Z) in vocab
    DIGITS = "0123456789"
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(
        self,
        beta: float = 1.0,
        vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ):
        """
        Initialize Layout Penalty.

        Args:
            beta: Penalty weight
            vocab: Character vocabulary (must match OCR vocab order)
        """
        super().__init__()
        self.beta = beta
        self.vocab = vocab

        # Pre-compute digit and letter index masks
        digit_indices = [i for i, c in enumerate(vocab) if c in self.DIGITS]
        letter_indices = [i for i, c in enumerate(vocab) if c in self.LETTERS]

        # Store as buffers (move to device with model)
        self.register_buffer("digit_mask", torch.zeros(len(vocab)))
        self.register_buffer("letter_mask", torch.zeros(len(vocab)))
        self.digit_mask[digit_indices] = 1.0
        self.letter_mask[letter_indices] = 1.0

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_texts: List[str],
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute differentiable layout penalty using OCR logits.

        Args:
            pred_logits: OCR logits of shape (B, K, C) — raw or softmaxed
            gt_texts: Ground truth texts
            device: Device for tensors

        Returns:
            Penalty tensor (differentiable) and info dict
        """
        B, K, C = pred_logits.shape

        # Handle CTC blank token (C = vocab + 1): slice it off
        vocab_size = len(self.vocab)
        if C > vocab_size:
            pred_logits = pred_logits[:, :, :vocab_size]
            C = vocab_size

        # Softmax to get probabilities
        probs = F.softmax(pred_logits, dim=-1)  # (B, K, C)

        # Move masks to correct device
        digit_mask = self.digit_mask.to(device)   # (C,)
        letter_mask = self.letter_mask.to(device)  # (C,)

        total_penalty = torch.tensor(0.0, device=device)
        total_violations = 0
        total_chars = 0

        for b in range(B):
            gt = gt_texts[b]
            for k in range(min(K, len(gt))):
                gt_char = gt[k]
                total_chars += 1

                # Probability the model assigns to WRONG character type
                if gt_char in self.DIGITS:
                    # GT is digit → penalize probability on letter classes
                    wrong_prob = (probs[b, k] * letter_mask).sum()
                    total_penalty = total_penalty + wrong_prob
                    if wrong_prob.item() > 0.5:
                        total_violations += 1
                elif gt_char in self.LETTERS:
                    # GT is letter → penalize probability on digit classes
                    wrong_prob = (probs[b, k] * digit_mask).sum()
                    total_penalty = total_penalty + wrong_prob
                    if wrong_prob.item() > 0.5:
                        total_violations += 1

        # Average over batch
        if B > 0:
            total_penalty = self.beta * total_penalty / B

        return total_penalty, {
            "layout_penalty": total_penalty.detach(),
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
        self.layout_penalty = LayoutPenalty(beta, vocab=vocab)

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

        # Layout Penalty (differentiable — uses logits, not decoded texts)
        if self.lambda_layout > 0:
            layout_loss, layout_info = self.layout_penalty(pred_logits, gt_texts, device=sr_images.device)
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


def _gaussian_window(window_size: int, sigma: float, channels: int = 1, device: torch.device = None) -> torch.Tensor:
    """Create a Gaussian window for SSIM computation (Wang et al. 2004)."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # Create 2D kernel
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # Expand to all channels
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two tensors.

    Uses a proper Gaussian window (Wang et al. 2004) instead of simple
    average pooling for more accurate local statistics estimation.

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

    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma=1.5, channels=channels, device=x.device)
    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x ** 2, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

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

    # Test Layout Penalty (now uses logits, not decoded texts)
    layout_penalty = LayoutPenalty()
    penalty, info = layout_penalty(pred_logits, gt_texts)
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
