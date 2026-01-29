"""
Confusion Matrix Tracker for LCOFL

Tracks character confusions during training to update the
weights for the Classification Loss component of LCOFL.
"""

import torch
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict


class ConfusionTracker:
    """
    Tracks character confusions during validation.

    Maintains a confusion matrix and provides methods to:
    - Update with batch predictions
    - Get weight updates for LCOFL
    - Save/load confusion state
    """

    def __init__(
        self,
        vocab: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ):
        """
        Initialize Confusion Tracker.

        Args:
            vocab: Character vocabulary
        """
        self.vocab = vocab
        self.char_to_idx = {c: i for i, c in enumerate(vocab)}
        self.num_classes = len(vocab)

        # Initialize confusion matrix
        self.reset()

    def reset(self):
        """Reset confusion matrix to zeros."""
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

    def update(
        self,
        pred_texts: List[str],
        gt_texts: List[str],
    ):
        """
        Update confusion matrix with a batch of predictions.

        Args:
            pred_texts: List of predicted texts
            gt_texts: List of ground truth texts
        """
        for pred_text, gt_text in zip(pred_texts, gt_texts):
            min_len = min(len(pred_text), len(gt_text))
            for i in range(min_len):
                pred_char = pred_text[i]
                gt_char = gt_text[i]

                if pred_char in self.char_to_idx and gt_char in self.char_to_idx:
                    pred_idx = self.char_to_idx[pred_char]
                    gt_idx = self.char_to_idx[gt_char]
                    self.confusion_matrix[gt_idx, pred_idx] += 1

    def get_confusion_pairs(
        self,
        threshold: int = 1,
    ) -> List[Tuple[str, str, int]]:
        """
        Get character pairs that are frequently confused.

        Args:
            threshold: Minimum number of confusions to include

        Returns:
            List of (gt_char, pred_char, count) tuples
        """
        pairs = []

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i == j:
                    continue  # Skip correct predictions

                count = int(self.confusion_matrix[i, j].item())
                if count >= threshold:
                    gt_char = self.vocab[i]
                    pred_char = self.vocab[j]
                    pairs.append((gt_char, pred_char, count))

        # Sort by count descending
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs

    def get_weights(
        self,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        """
        Get weight updates for LCOFL Classification Loss.

        Args:
            alpha: Weight increment for each confusion

        Returns:
            Weight tensor of shape (vocab_size,)
        """
        # Get diagonal (correct predictions)
        correct = torch.diag(self.confusion_matrix)

        # For each character, sum confusions
        weight_increments = self.confusion_matrix.sum(dim=1) - correct

        # Compute weights
        weights = 1.0 + alpha * weight_increments

        return weights

    def get_accuracy(
        self,
        include_partial: bool = True,
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics from confusion matrix.

        Args:
            include_partial: Whether to compute partial match accuracy

        Returns:
            Dict with accuracy metrics
        """
        total = self.confusion_matrix.sum().item()
        correct = torch.diag(self.confusion_matrix).sum().item()

        char_accuracy = correct / total if total > 0 else 0.0

        metrics = {
            "char_accuracy": char_accuracy,
            "total_predictions": int(total),
            "correct_predictions": int(correct),
        }

        return metrics

    def get_report(self) -> str:
        """
        Generate a human-readable report.

        Returns:
            String report with confusion statistics
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Confusion Matrix Report")
        lines.append("=" * 60)

        # Overall statistics
        metrics = self.get_accuracy()
        lines.append(f"Character Accuracy: {metrics['char_accuracy']:.4f}")
        lines.append(f"Total Predictions: {metrics['total_predictions']}")
        lines.append(f"Correct Predictions: {metrics['correct_predictions']}")
        lines.append("")

        # Top confusions
        lines.append("Top Character Confusions:")
        pairs = self.get_confusion_pairs(threshold=1)
        for gt_char, pred_char, count in pairs[:20]:
            lines.append(f"  '{gt_char}' -> '{pred_char}': {count}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save(self, path: str):
        """
        Save confusion matrix to file.

        Args:
            path: Path to save file (json or pt)
        """
        path = Path(path)

        if path.suffix == ".json":
            # Save as JSON
            data = {
                "vocab": self.vocab,
                "confusion_matrix": self.confusion_matrix.tolist(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            # Save as PyTorch file
            torch.save({
                "vocab": self.vocab,
                "confusion_matrix": self.confusion_matrix,
            }, path)

    def load(self, path: str):
        """
        Load confusion matrix from file.

        Args:
            path: Path to load file (json or pt)
        """
        path = Path(path)

        if path.suffix == ".json":
            # Load from JSON
            with open(path, "r") as f:
                data = json.load(f)
            self.vocab = data["vocab"]
            self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
            self.confusion_matrix = torch.tensor(data["confusion_matrix"])
        else:
            # Load from PyTorch file
            checkpoint = torch.load(path, map_location="cpu")
            self.vocab = checkpoint["vocab"]
            self.confusion_matrix = checkpoint["confusion_matrix"]
            self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}


class MetricsTracker:
    """
    Tracks multiple metrics during training.

    Useful for monitoring recognition rate, loss, and other metrics.
    """

    def __init__(self):
        """Initialize Metrics Tracker."""
        self.metrics = defaultdict(list)

    def update(self, name: str, value: float):
        """
        Update a metric.

        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name].append(value)

    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value of a metric."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None

    def get_average(self, name: str) -> float:
        """Get the average value of a metric."""
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0.0

    def is_improving(
        self,
        name: str,
        patience: int = 5,
        maximize: bool = True,
    ) -> bool:
        """
        Check if a metric is improving.

        Args:
            name: Metric name
            patience: Number of steps to check
            maximize: If True, check for increase; otherwise decrease

        Returns:
            True if metric is improving
        """
        if name not in self.metrics or len(self.metrics[name]) < patience:
            return True

        recent = self.metrics[name][-patience:]

        if maximize:
            # Check if latest is the best
            return recent[-1] >= max(recent[:-1])
        else:
            # Check if latest is the best (lowest)
            return recent[-1] <= min(recent[:-1])

    def save(self, path: str):
        """Save metrics to file."""
        with open(path, "w") as f:
            json.dump(dict(self.metrics), f, indent=2)

    def load(self, path: str):
        """Load metrics from file."""
        with open(path, "r") as f:
            data = json.load(f)
            self.metrics = defaultdict(list, data)


if __name__ == "__main__":
    # Test Confusion Tracker
    print("Testing ConfusionTracker...")

    tracker = ConfusionTracker()

    # Simulate some predictions
    pred_texts = [
        "ABC1234",
        "ABC1234",  # Correct
        "XBC1234",  # A -> X confusion
        "ABC1Z34",  # 2 -> Z confusion
        "ABO1234",  # C -> O confusion
    ]
    gt_texts = [
        "ABC1234",
        "ABC1234",
        "ABC1234",
        "ABC1234",
        "ABC1234",
    ]

    tracker.update(pred_texts, gt_texts)

    # Get report
    print(tracker.get_report())

    # Get weights
    weights = tracker.get_weights(alpha=0.1)
    print(f"Weights shape: {weights.shape}")
    print(f"Weights: {weights}")

    # Get accuracy
    metrics = tracker.get_accuracy()
    print(f"Character accuracy: {metrics['char_accuracy']:.4f}")

    # Test save/load
    tracker.save("test_confusion.json")
    tracker2 = ConfusionTracker()
    tracker2.load("test_confusion.json")
    print(f"Loaded confusion matrix matches: {torch.allclose(tracker.confusion_matrix, tracker2.confusion_matrix)}")

    import os
    os.remove("test_confusion.json")

    print("ConfusionTracker test passed!")
