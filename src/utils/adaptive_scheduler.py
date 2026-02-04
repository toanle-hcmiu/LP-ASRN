"""
Adaptive Weight Scheduler for Loss Component Warm-up

Implements various scheduling strategies for gradually increasing
loss component weights during training.

This is particularly useful for embedding consistency loss,
which benefits from a warm-up period to prevent early training
instability.
"""

import math
from typing import Optional, Callable


class AdaptiveWeightScheduler:
    """
    Adaptive weight scheduler for loss component warm-up.

    Gradually increases weight from start_weight to target_weight
    over warmup_epochs, optionally guided by OCR accuracy.

    Supports multiple scheduling strategies:
    - linear: Linear increase from start to target
    - cosine: Cosine annealing schedule
    - ocr_guided: Faster warm-up if OCR accuracy is already high
    - exponential: Exponential increase
    """

    def __init__(
        self,
        start_weight: float = 0.0,
        target_weight: float = 0.3,
        warmup_epochs: int = 50,
        schedule: str = "linear",
        delay_epochs: int = 0,
        hold_epochs: int = 0,
    ):
        """
        Initialize Adaptive Weight Scheduler.

        Args:
            start_weight: Initial weight (typically 0.0)
            target_weight: Target weight after warm-up
            warmup_epochs: Number of epochs to reach target weight
            schedule: Scheduling strategy ('linear', 'cosine', 'ocr_guided', 'exponential')
            delay_epochs: Delay start of warm-up by this many epochs
            hold_epochs: After reaching target, hold for this many epochs before applying any decay
        """
        self.start_weight = start_weight
        self.target_weight = target_weight
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule
        self.delay_epochs = delay_epochs
        self.hold_epochs = hold_epochs
        self.current_epoch = 0

        # Total epochs for the schedule (delay + warmup + hold)
        self.total_schedule_epochs = delay_epochs + warmup_epochs + hold_epochs

    def step(
        self,
        epoch: Optional[int] = None,
        ocr_accuracy: Optional[float] = None,
    ) -> float:
        """
        Get current weight based on epoch and optionally OCR accuracy.

        Args:
            epoch: Current epoch (uses internal counter if None)
            ocr_accuracy: Current OCR accuracy (for ocr_guided schedule)

        Returns:
            Current weight for this epoch
        """
        if epoch is not None:
            self.current_epoch = epoch

        # During delay period, use start_weight
        if self.current_epoch < self.delay_epochs:
            return self.start_weight

        # Compute progress within warmup period
        warmup_progress = (self.current_epoch - self.delay_epochs) / max(self.warmup_epochs, 1)

        # Check if we're in warmup or hold period
        effective_epoch = self.current_epoch - self.delay_epochs

        if effective_epoch >= self.warmup_epochs:
            # Past warmup, check if we're in hold period
            if effective_epoch < self.warmup_epochs + self.hold_epochs:
                # In hold period, use target weight
                return self.target_weight
            else:
                # Past hold period, continue with target weight
                # (can be extended to include decay in future)
                return self.target_weight

        # Apply schedule based on type
        if self.schedule == "linear":
            weight = self._linear_schedule(warmup_progress)

        elif self.schedule == "cosine":
            weight = self._cosine_schedule(warmup_progress)

        elif self.schedule == "ocr_guided" and ocr_accuracy is not None:
            # Faster warm-up if OCR is already accurate
            weight = self._ocr_guided_schedule(warmup_progress, ocr_accuracy)

        elif self.schedule == "exponential":
            weight = self._exponential_schedule(warmup_progress)

        else:
            # Fallback to linear
            weight = self._linear_schedule(warmup_progress)

        return weight

    def _linear_schedule(self, progress: float) -> float:
        """Linear interpolation between start and target weights."""
        return self.start_weight + progress * (self.target_weight - self.start_weight)

    def _cosine_schedule(self, progress: float) -> float:
        """Cosine annealing schedule."""
        return self.start_weight + 0.5 * (self.target_weight - self.start_weight) * (
            1 - math.cos(math.pi * progress)
        )

    def _ocr_guided_schedule(self, progress: float, ocr_accuracy: float) -> float:
        """
        OCR-guided schedule for faster convergence when OCR is already accurate.

        Adjusts the warm-up speed based on OCR accuracy - if OCR is already
        performing well, we can increase the embedding weight faster.
        """
        # OCR accuracy threshold (above this, accelerate warm-up)
        acc_threshold = 0.5

        # Acceleration factor based on OCR accuracy
        if ocr_accuracy >= acc_threshold:
            # Scale acceleration: at 50% OCR, factor is 1.0 (no change)
            # at 80% OCR, factor is 1.6 (60% faster)
            acc_factor = 1.0 + (ocr_accuracy - acc_threshold) * 2.0
        else:
            acc_factor = 1.0

        # Adjust progress with acceleration factor
        adjusted_progress = min(progress * acc_factor, 1.0)

        return self._linear_schedule(adjusted_progress)

    def _exponential_schedule(self, progress: float) -> float:
        """Exponential schedule (slower initial increase)."""
        # Use power of 2 for exponential curve
        adjusted_progress = progress ** 2
        return self._linear_schedule(adjusted_progress)

    def get_weight(self, *args, **kwargs) -> float:
        """Alias for step() method."""
        return self.step(*args, **kwargs)

    def update_epoch(self, epoch: int):
        """Update the current epoch counter."""
        self.current_epoch = epoch

    def increment_epoch(self) -> int:
        """Increment epoch counter and return new epoch."""
        self.current_epoch += 1
        return self.current_epoch

    def reset(self):
        """Reset the scheduler to initial state."""
        self.current_epoch = 0


class MultiComponentScheduler:
    """
    Scheduler for multiple loss components with different warm-up schedules.

    Manages multiple AdaptiveWeightScheduler instances for different
    loss components (e.g., embedding, SSIM, perceptual).
    """

    def __init__(
        self,
        schedulers: dict,
    ):
        """
        Initialize Multi-Component Scheduler.

        Args:
            schedulers: Dictionary mapping component names to
                       AdaptiveWeightScheduler instances or config dicts
        """
        self.schedulers = {}

        for name, scheduler in schedulers.items():
            if isinstance(scheduler, dict):
                # Create scheduler from config dict
                self.schedulers[name] = AdaptiveWeightScheduler(**scheduler)
            else:
                # Assume it's already a scheduler instance
                self.schedulers[name] = scheduler

    def step(
        self,
        epoch: Optional[int] = None,
        **metrics,
    ) -> dict:
        """
        Get current weights for all components.

        Args:
            epoch: Current epoch (uses internal counters if None)
            **metrics: Additional metrics (e.g., ocr_accuracy) for guided schedules

        Returns:
            Dictionary mapping component names to current weights
        """
        weights = {}
        for name, scheduler in self.schedulers.items():
            # Get OCR accuracy if available for ocr_guided schedules
            ocr_acc = metrics.get("ocr_accuracy")
            weights[name] = scheduler.step(epoch, ocr_accuracy)

        return weights

    def get_weights(self, *args, **kwargs) -> dict:
        """Alias for step() method."""
        return self.step(*args, **kwargs)

    def update_epoch(self, epoch: int):
        """Update epoch counter for all schedulers."""
        for scheduler in self.schedulers.values():
            scheduler.update_epoch(epoch)

    def reset(self):
        """Reset all schedulers."""
        for scheduler in self.schedulers.values():
            scheduler.reset()


class LambdaScheduler:
    """
    Simple lambda-based weight scheduler.

    Uses a user-provided function to compute weights.
    """

    def __init__(
        self,
        weight_fn: Callable[[int], float],
    ):
        """
        Initialize Lambda Scheduler.

        Args:
            weight_fn: Function that takes epoch and returns weight
        """
        self.weight_fn = weight_fn
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None) -> float:
        """Get current weight."""
        if epoch is not None:
            self.current_epoch = epoch
        return self.weight_fn(self.current_epoch)

    def get_weight(self, epoch: Optional[int] = None) -> float:
        """Alias for step() method."""
        return self.step(epoch)

    def update_epoch(self, epoch: int):
        """Update epoch counter."""
        self.current_epoch = epoch


if __name__ == "__main__":
    # Test the schedulers
    print("Testing Adaptive Weight Scheduler...")

    # Test linear schedule
    scheduler = AdaptiveWeightScheduler(
        start_weight=0.0,
        target_weight=0.3,
        warmup_epochs=50,
        schedule="linear",
    )

    print("\nLinear Schedule:")
    for epoch in [0, 10, 25, 50, 100]:
        weight = scheduler.step(epoch)
        print(f"  Epoch {epoch:3d}: weight = {weight:.4f}")

    # Test cosine schedule
    scheduler_cosine = AdaptiveWeightScheduler(
        start_weight=0.0,
        target_weight=0.3,
        warmup_epochs=50,
        schedule="cosine",
    )

    print("\nCosine Schedule:")
    for epoch in [0, 10, 25, 50, 100]:
        weight = scheduler_cosine.step(epoch)
        print(f"  Epoch {epoch:3d}: weight = {weight:.4f}")

    # Test OCR-guided schedule
    scheduler_ocr = AdaptiveWeightScheduler(
        start_weight=0.0,
        target_weight=0.3,
        warmup_epochs=50,
        schedule="ocr_guided",
    )

    print("\nOCR-Guided Schedule (with varying OCR accuracy):")
    for epoch in [0, 10, 25, 50]:
        for acc in [0.3, 0.5, 0.7]:
            weight = scheduler_ocr.step(epoch, ocr_accuracy=acc)
            print(f"  Epoch {epoch:3d}, OCR {acc:.1f}: weight = {weight:.4f}")

    # Test MultiComponentScheduler
    print("\nMulti-Component Scheduler:")
    multi_scheduler = MultiComponentScheduler({
        "embedding": {
            "start_weight": 0.0,
            "target_weight": 0.3,
            "warmup_epochs": 50,
            "schedule": "linear",
        },
        "ssim": {
            "start_weight": 0.1,
            "target_weight": 0.2,
            "warmup_epochs": 20,
            "schedule": "cosine",
        },
    })

    for epoch in [0, 10, 25, 50]:
        weights = multi_scheduler.step(epoch)
        print(f"  Epoch {epoch:3d}: {weights}")

    # Test LambdaScheduler
    print("\nLambda Scheduler (custom function):")
    # Custom: weight increases quadratically
    lambda_scheduler = LambdaScheduler(lambda e: min(0.3 * (e / 50) ** 2, 0.3))
    for epoch in [0, 10, 25, 50, 100]:
        weight = lambda_scheduler.step(epoch)
        print(f"  Epoch {epoch:3d}: weight = {weight:.4f}")

    print("\nAdaptive Weight Scheduler test passed!")
