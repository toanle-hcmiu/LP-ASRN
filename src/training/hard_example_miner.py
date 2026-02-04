"""
Hard Example Mining for OCR-Driven Training (Stage 4)

Implements hard example mining strategies that focus training on
samples that OCR struggles with.

Key components:
1. HardExampleMiner - Tracks per-sample OCR accuracy for weighted sampling
2. CharacterConfusionTracker - Analyzes character-level confusion patterns
3. CurriculumSampler - Provides curriculum-based sampling strategies

This module enables Stage 4 training where the model focuses on
hard examples identified by OCR errors.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from typing import Dict, List, Tuple, Optional
import numpy as np


class HardExampleMiner:
    """
    Mines hard examples based on OCR performance.

    Tracks per-sample OCR accuracy and provides weighted sampling
    to focus training on difficult cases.

    Uses an exponential moving average to maintain stable estimates
    of per-sample difficulty.
    """

    def __init__(
        self,
        dataset_size: int,
        difficulty_bins: int = 5,
        alpha: float = 2.0,
        ema_decay: float = 0.9,
        min_samples_seen: int = 10,
    ):
        """
        Initialize Hard Example Miner.

        Args:
            dataset_size: Total number of samples in the dataset
            difficulty_bins: Number of bins for difficulty categorization
            alpha: Exponent for difficulty weighting (higher = more focus on hard samples)
            ema_decay: Decay factor for exponential moving average
            min_samples_seen: Minimum samples seen before applying difficulty weighting
        """
        self.dataset_size = dataset_size
        self.difficulty_bins = difficulty_bins
        self.alpha = alpha
        self.ema_decay = ema_decay
        self.min_samples_seen = min_samples_seen

        # Track per-sample metrics
        # Using CPU tensors for memory efficiency
        self.register_buffer("sample_accuracies", torch.zeros(dataset_size))
        self.register_buffer("sample_counts", torch.zeros(dataset_size))

        # Difficulty bins for curriculum learning
        self.bin_boundaries = torch.linspace(0, 1, difficulty_bins + 1)[:-1]

        # Statistics
        self.total_updates = 0

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a buffer (tensor that persists but doesn't require gradients)."""
        setattr(self, name, tensor)

    def update(
        self,
        indices: torch.Tensor,
        char_accuracies: torch.Tensor,
        word_accuracies: Optional[torch.Tensor] = None,
    ):
        """
        Update accuracy estimates for samples.

        Args:
            indices: Tensor of sample indices (N,)
            char_accuracies: Character-level accuracies (N,)
            word_accuracies: Optional word-level accuracies (N,)
        """
        with torch.no_grad():
            for idx, char_acc in zip(indices, char_accuracies):
                idx_int = int(idx.item())
                acc_val = char_acc.item()

                # Update using exponential moving average
                if self.sample_counts[idx_int] > 0:
                    self.sample_accuracies[idx_int] = (
                        self.ema_decay * self.sample_accuracies[idx_int] +
                        (1 - self.ema_decay) * acc_val
                    )
                else:
                    self.sample_accuracies[idx_int] = acc_val

                self.sample_counts[idx_int] += 1

            self.total_updates += len(indices)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Get sampling weights (inversely proportional to accuracy).

        Returns:
            Weight tensor of shape (dataset_size,) that sums to 1
        """
        # Get valid samples (seen at least min_samples_seen times)
        valid_mask = self.sample_counts >= self.min_samples_seen

        # Initialize weights (uniform for unseen samples)
        weights = torch.ones(self.dataset_size)

        # Inverse of accuracy, raised to power alpha
        # Add small epsilon to prevent division by zero
        valid_accs = self.sample_accuracies[valid_mask]
        valid_weights = torch.pow(1.0 - valid_accs + 0.1, self.alpha)

        weights[valid_mask] = valid_weights

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return weights

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """
        Get distribution of samples across difficulty bins.

        Returns:
            Dictionary mapping bin names to counts
        """
        valid_mask = self.sample_counts >= self.min_samples_seen
        valid_accs = self.sample_accuracies[valid_mask]

        distribution = {}
        for i in range(len(self.bin_boundaries)):
            if i < len(self.bin_boundaries) - 1:
                bin_mask = (valid_accs >= self.bin_boundaries[i]) & (
                    valid_accs < self.bin_boundaries[i + 1]
                )
            else:
                bin_mask = valid_accs >= self.bin_boundaries[i]

            distribution[f"bin_{i}"] = bin_mask.sum().item()

        return distribution

    def get_confusion_weights(
        self,
        confusion_matrix: torch.Tensor,
        vocab_size: int = 36,
    ) -> torch.Tensor:
        """
        Get per-character weights based on confusion matrix.

        Characters that are frequently confused get higher weights.

        Args:
            confusion_matrix: Confusion matrix (vocab_size, vocab_size)
            vocab_size: Size of the vocabulary

        Returns:
            Weight tensor of shape (vocab_size,)
        """
        # Sum confusions per character (excluding diagonal)
        if confusion_matrix.shape[0] != vocab_size:
            # Handle size mismatch
            vocab_size = confusion_matrix.shape[0]

        diagonal = torch.diag(confusion_matrix)
        confusion_sum = confusion_matrix.sum(dim=1) - diagonal

        # Normalize to range [1, 3] for weighting
        max_confusion = confusion_sum.max() + 1e-6
        weights = 1.0 + 2.0 * confusion_sum / max_confusion

        return weights

    def get_hard_samples(
        self,
        n_samples: int,
        min_acc: float = 0.0,
        max_acc: float = 0.5,
    ) -> torch.Tensor:
        """
        Get indices of hard samples within accuracy range.

        Args:
            n_samples: Maximum number of samples to return
            min_acc: Minimum accuracy threshold
            max_acc: Maximum accuracy threshold

        Returns:
            Tensor of hard sample indices
        """
        valid_mask = (self.sample_counts >= self.min_samples_seen) & (
            self.sample_accuracies >= min_acc
        ) & (self.sample_accuracies <= max_acc)

        hard_indices = torch.where(valid_mask)[0]

        if len(hard_indices) > n_samples:
            # Sample from hard indices
            perm = torch.randperm(len(hard_indices))[:n_samples]
            hard_indices = hard_indices[perm]

        return hard_indices


class CharacterConfusionTracker:
    """
    Tracks character-level confusion patterns for adaptive loss weighting.

    Maintains a confusion matrix and provides confusion-based
    loss weights for individual characters.
    """

    def __init__(
        self,
        vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        smoothing: float = 0.1,
    ):
        """
        Initialize Character Confusion Tracker.

        Args:
            vocab: Character vocabulary
            smoothing: Smoothing factor for confusion matrix updates
        """
        self.vocab = vocab
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.smoothing = smoothing

        # Confusion matrix (rows = true, cols = predicted)
        self.register_buffer("confusion_matrix", torch.zeros(self.vocab_size, self.vocab_size))

        # Character frequencies
        self.register_buffer("char_counts", torch.zeros(self.vocab_size))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a buffer."""
        setattr(self, name, tensor)

    def update(
        self,
        pred_texts: List[str],
        gt_texts: List[str],
    ):
        """
        Update confusion matrix from predictions.

        Args:
            pred_texts: List of predicted texts
            gt_texts: List of ground truth texts
        """
        with torch.no_grad():
            for pred_text, gt_text in zip(pred_texts, gt_texts):
                min_len = min(len(pred_text), len(gt_text))

                for i in range(min_len):
                    pred_char = pred_text[i]
                    gt_char = gt_text[i]

                    if pred_char in self.char_to_idx and gt_char in self.char_to_idx:
                        p_idx = self.char_to_idx[pred_char]
                        g_idx = self.char_to_idx[gt_char]

                        # Smooth update
                        self.confusion_matrix[g_idx, p_idx] += 1.0
                        self.char_counts[g_idx] += 1.0

    def get_loss_weights(
        self,
        base_weight: float = 1.0,
        max_weight: float = 3.0,
    ) -> torch.Tensor:
        """
        Get per-character loss weights based on confusion patterns.

        Args:
            base_weight: Base weight for all characters
            max_weight: Maximum weight for highly confused characters

        Returns:
            Weight tensor of shape (vocab_size,)
        """
        # Get confusion sum (excluding diagonal)
        diagonal = torch.diag(self.confusion_matrix)
        confusion_sum = self.confusion_matrix.sum(dim=1) - diagonal

        # Normalize by character counts to get confusion rate
        confusion_rate = confusion_sum / (self.char_counts + 1e-6)

        # Scale to [base_weight, max_weight]
        max_rate = confusion_rate.max() + 1e-6
        weights = base_weight + (max_weight - base_weight) * confusion_rate / max_rate

        return weights

    def get_confused_pairs(
        self,
        top_k: int = 10,
        min_count: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Get top confused character pairs.

        Args:
            top_k: Number of top pairs to return
            min_count: Minimum count for a pair to be considered

        Returns:
            List of (gt_char, pred_char, count) tuples
        """
        # Get off-diagonal entries
        mask = ~torch.eye(self.vocab_size, dtype=torch.bool)
        off_diagonal = self.confusion_matrix[mask]

        # Get top confused pairs
        counts, indices = torch.topk(off_diagonal, min(top_k, off_diagonal.numel()))

        pairs = []
        for count, idx in zip(counts, indices):
            if count.item() < min_count:
                break

            # Convert flat index back to (row, col)
            row = idx // self.vocab_size
            col = idx % self.vocab_size

            gt_char = self.vocab[int(row)]
            pred_char = self.vocab[int(col)]
            pairs.append((gt_char, pred_char, count.item()))

        return pairs


class CurriculumSampler:
    """
    Curriculum-based sampling strategy.

    Gradually transitions from easy to hard examples during training,
    following a curriculum learning approach.
    """

    def __init__(
        self,
        hard_example_miner: HardExampleMiner,
        total_epochs: int = 50,
        initial_p_hard: float = 0.1,
        final_p_hard: float = 0.8,
        curriculum_type: str = "linear",
    ):
        """
        Initialize Curriculum Sampler.

        Args:
            hard_example_miner: HardExampleMiner instance
            total_epochs: Total epochs for curriculum
            initial_p_hard: Initial probability of sampling hard examples
            final_p_hard: Final probability of sampling hard examples
            curriculum_type: Type of curriculum ('linear', 'exponential', 'step')
        """
        self.miner = hard_example_miner
        self.total_epochs = total_epochs
        self.initial_p_hard = initial_p_hard
        self.final_p_hard = final_p_hard
        self.curriculum_type = curriculum_type

    def get_p_hard(self, epoch: int) -> float:
        """
        Get probability of sampling hard examples at given epoch.

        Args:
            epoch: Current epoch

        Returns:
            Probability of sampling hard examples
        """
        progress = min(epoch / self.total_epochs, 1.0)

        if self.curriculum_type == "linear":
            p_hard = self.initial_p_hard + progress * (
                self.final_p_hard - self.initial_p_hard
            )
        elif self.curriculum_type == "exponential":
            # Slower initial increase, faster later
            p_hard = self.initial_p_hard + (
                self.final_p_hard - self.initial_p_hard
            ) * (progress ** 2)
        elif self.curriculum_type == "step":
            # Step-wise increases at 25%, 50%, 75%
            if progress < 0.25:
                p_hard = self.initial_p_hard
            elif progress < 0.5:
                p_hard = self.initial_p_hard + 0.25 * (
                    self.final_p_hard - self.initial_p_hard
                )
            elif progress < 0.75:
                p_hard = self.initial_p_hard + 0.5 * (
                    self.final_p_hard - self.initial_p_hard
                )
            else:
                p_hard = self.final_p_hard
        else:
            p_hard = self.initial_p_hard

        return p_hard

    def create_sampler(
        self,
        epoch: int,
        batch_size: int = 32,
    ) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler for the given epoch.

        Args:
            epoch: Current epoch
            batch_size: Batch size (for num_samples)

        Returns:
            WeightedRandomSampler instance
        """
        # Get base difficulty weights
        difficulty_weights = self.miner.get_sample_weights()

        # Blend with uniform sampling based on curriculum
        p_hard = self.get_p_hard(epoch)
        uniform_weights = torch.ones_like(difficulty_weights) / len(difficulty_weights)

        # Blend weights
        blended_weights = (
            p_hard * difficulty_weights +
            (1 - p_hard) * uniform_weights
        )

        # Normalize
        blended_weights = blended_weights / blended_weights.sum()

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=blended_weights,
            num_samples=batch_size * 1000,  # Approximate for infinite sampling
            replacement=True,
        )

        return sampler


if __name__ == "__main__":
    # Test the hard example miner
    print("Testing Hard Example Mining...")

    dataset_size = 100
    miner = HardExampleMiner(
        dataset_size=dataset_size,
        alpha=2.0,
    )

    # Simulate updates
    for epoch in range(10):
        # Random indices and accuracies
        indices = torch.randint(0, dataset_size, (32,))
        char_accs = torch.rand(32) * 0.8 + 0.2  # 0.2 to 1.0

        miner.update(indices, char_accs)

    # Get weights
    weights = miner.get_sample_weights()
    print(f"Sample weights shape: {weights.shape}")
    print(f"Sum of weights: {weights.sum().item():.4f}")

    # Test confusion tracker
    print("\nTesting Character Confusion Tracker...")
    tracker = CharacterConfusionTracker()

    pred_texts = ["ABC123", "ABD123", "ABE123", "ABF123"]
    gt_texts = ["ABC123", "ABC123", "ABC123", "ABC123"]

    tracker.update(pred_texts, gt_texts)
    loss_weights = tracker.get_loss_weights()
    print(f"Loss weights shape: {loss_weights.shape}")

    confused_pairs = tracker.get_confused_pairs(top_k=5)
    print(f"Confused pairs: {confused_pairs}")

    # Test curriculum sampler
    print("\nTesting Curriculum Sampler...")
    curriculum = CurriculumSampler(
        hard_example_miner=miner,
        total_epochs=50,
    )

    for epoch in [0, 12, 25, 37, 50]:
        p_hard = curriculum.get_p_hard(epoch)
        print(f"  Epoch {epoch:2d}: p_hard = {p_hard:.3f}")

    print("\nHard Example Mining test passed!")
