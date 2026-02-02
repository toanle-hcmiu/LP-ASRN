"""
License Plate Dataset for Super-Resolution Training
Supports both Brazilian and Mercosur layouts with paired LR/HR images.
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class AddGaussianNoise:
    """Add Gaussian noise to tensor for low-light simulation."""

    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        noise = torch.randn_like(img) * self.std + self.mean
        return torch.clamp(img + noise, 0, 1)


class LicensePlateDataset(Dataset):
    """
    Dataset for license plate super-resolution.

    Loads paired low-resolution and high-resolution images from track folders.
    Each track contains:
    - lr-001.png to lr-005.png (low-resolution images)
    - hr-001.png to hr-005.png (high-resolution images)
    - annotations.json (plate text, layout, and corner coordinates)
    """

    # Layout specifications for Brazilian and Mercosur plates
    LAYOUT_PATTERNS = {
        "Brazilian": "LLLNNNN",  # 3 letters, 4 digits
        "Mercosur": "LLLNLNN",   # 3 letters, 1 digit, 2 letters, 2 digits
    }

    def __init__(
        self,
        root_dir: str,
        scenarios: Optional[List[str]] = None,
        layouts: Optional[List[str]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        crop_to_corners: bool = True,
        augment: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            root_dir: Root directory containing the data (e.g., "data/train")
            scenarios: List of scenarios to include ["Scenario-A", "Scenario-B"]
            layouts: List of layouts to include ["Brazilian", "Mercosur"]
            image_size: Target size for (lr_h, lr_w). If None, use original size.
            crop_to_corners: Whether to crop images using corner coordinates
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images to [-1, 1]
        """
        self.root_dir = Path(root_dir)
        self.scenarios = scenarios or ["Scenario-A", "Scenario-B"]
        self.layouts = layouts or ["Brazilian", "Mercosur"]
        self.image_size = image_size
        self.crop_to_corners = crop_to_corners
        self.augment = augment
        self.normalize = normalize

        # Load all samples
        self.samples = self._load_samples()

        # Setup augmentation
        self.augment_transform = self._get_augment_transform() if augment else None

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all valid samples from the dataset directory."""
        samples = []

        for scenario in self.scenarios:
            scenario_path = self.root_dir / scenario
            if not scenario_path.exists():
                continue

            for layout in self.layouts:
                layout_path = scenario_path / layout
                if not layout_path.exists():
                    continue

                # Iterate through track folders
                for track_dir in sorted(layout_path.iterdir()):
                    if not track_dir.is_dir():
                        continue

                    # Check for annotations file
                    anno_path = track_dir / "annotations.json"
                    if not anno_path.exists():
                        continue

                    try:
                        with open(anno_path, "r") as f:
                            annotations = json.load(f)

                        plate_layout = annotations.get("plate_layout")
                        plate_text = annotations.get("plate_text")
                        corners = annotations.get("corners", {})

                        # Skip if layout doesn't match
                        if plate_layout not in self.layouts:
                            continue

                        # Load all 5 image pairs
                        for i in range(1, 6):
                            lr_name = f"lr-{i:03d}.png"
                            hr_name = f"hr-{i:03d}.png"

                            lr_path = track_dir / lr_name
                            hr_path = track_dir / hr_name

                            if not (lr_path.exists() and hr_path.exists()):
                                continue

                            samples.append({
                                "lr_path": str(lr_path),
                                "hr_path": str(hr_path),
                                "plate_text": plate_text,
                                "plate_layout": plate_layout,
                                "lr_corners": corners.get(lr_name),
                                "hr_corners": corners.get(hr_name),
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue

        return samples

    def _get_augment_transform(self) -> transforms.Compose:
        """Get license plate specific augmentation pipeline - NO horizontal flip!"""
        return transforms.Compose([
            # Geometric transforms (preserve text order - NO horizontal flip!)
            transforms.RandomAffine(
                degrees=5,              # Slight rotation
                translate=(0.05, 0.1),  # Small translation
                scale=(0.9, 1.1),       # Scale variation
                shear=5,                # Perspective simulation
            ),
            # Photometric transforms
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            ),
            # Blur for motion/defocus simulation
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0)),
            ], p=0.3),
            # Noise for low-light simulation
            transforms.RandomApply([
                AddGaussianNoise(mean=0, std=0.05),
            ], p=0.3),
        ])

    def _load_image(self, path: str) -> Image.Image:
        """Load an image file."""
        return Image.open(path).convert("RGB")

    def _crop_with_corners(
        self,
        image: Image.Image,
        corners: Optional[Dict[str, List[int]]],
    ) -> Image.Image:
        """
        Crop image using corner coordinates.

        Args:
            image: PIL Image to crop
            corners: Dictionary with 'top-left', 'top-right', 'bottom-right', 'bottom-left'

        Returns:
            Cropped PIL Image
        """
        if not corners or not self.crop_to_corners:
            return image

        try:
            tl = corners["top-left"]
            br = corners["bottom-right"]

            left = min(tl[0], br[0])
            top = min(tl[1], br[1])
            right = max(tl[0], br[0])
            bottom = max(tl[1], br[1])

            # Add small padding
            padding = 2
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.width, right + padding)
            bottom = min(image.height, bottom + padding)

            return image.crop((left, top, right, bottom))
        except (KeyError, IndexError):
            return image

    def _resize_image(
        self,
        image: Image.Image,
        target_size: Optional[Tuple[int, int]],
    ) -> Image.Image:
        """Resize image to exact target size."""
        if target_size is None:
            return image

        target_h, target_w = target_size

        # Resize directly to target size (no aspect ratio preservation)
        # This ensures all images have the same size for batching
        try:
            return image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        except AttributeError:
            # For older Pillow versions
            return image.resize((target_w, target_h), Image.BICUBIC)

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        tensor = transforms.ToTensor()(image)

        if self.normalize:
            # Normalize to [-1, 1]
            tensor = tensor * 2.0 - 1.0

        return tensor

    def _apply_augmentation(self, lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply same augmentation to both LR and HR images."""
        if self.augment_transform is None:
            return lr, hr

        # Convert to PIL for augmentation
        to_pil = transforms.ToPILImage()

        # Denormalize if needed
        if self.normalize:
            lr_display = (lr + 1.0) / 2.0
            hr_display = (hr + 1.0) / 2.0
        else:
            lr_display = lr
            hr_display = hr

        lr_pil = to_pil(lr_display)
        hr_pil = to_pil(hr_display)

        # Apply same random seed for consistent augmentation
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)

        lr_aug = self.augment_transform(lr_pil)
        torch.manual_seed(seed)
        hr_aug = self.augment_transform(hr_pil)

        # Convert back to tensor
        lr_tensor = self._to_tensor(lr_aug)
        hr_tensor = self._to_tensor(hr_aug)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - lr: Low-resolution image tensor [3, H, W]
                - hr: High-resolution image tensor [3, 2*H, 2*W]
                - plate_text: License plate text string
                - plate_layout: Layout type (Brazilian/Mercosur)
                - lr_path: Path to LR image
                - hr_path: Path to HR image
        """
        sample = self.samples[idx]

        # Load images
        lr_image = self._load_image(sample["lr_path"])
        hr_image = self._load_image(sample["hr_path"])

        # Crop using corner coordinates
        lr_image = self._crop_with_corners(lr_image, sample["lr_corners"])
        hr_image = self._crop_with_corners(hr_image, sample["hr_corners"])

        # Resize to target size
        lr_image = self._resize_image(lr_image, self.image_size)
        if self.image_size is not None:
            hr_size = (self.image_size[0] * 2, self.image_size[1] * 2)
            hr_image = self._resize_image(hr_image, hr_size)

        # Convert to tensor
        lr_tensor = self._to_tensor(lr_image)
        hr_tensor = self._to_tensor(hr_image)

        # Apply augmentation
        lr_tensor, hr_tensor = self._apply_augmentation(lr_tensor, hr_tensor)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "plate_text": sample["plate_text"],
            "plate_layout": sample["plate_layout"],
            "lr_path": sample["lr_path"],
            "hr_path": sample["hr_path"],
        }

    def get_layout_pattern(self, layout: str) -> str:
        """Get the character pattern for a given layout."""
        return self.LAYOUT_PATTERNS.get(layout, "")

    def is_digit_position(self, layout: str, position: int) -> bool:
        """
        Check if a position in the layout should be a digit.

        Args:
            layout: "Brazilian" or "Mercosur"
            position: 0-indexed position in the plate

        Returns:
            True if position should be a digit, False if letter
        """
        pattern = self.LAYOUT_PATTERNS.get(layout, "")
        if position >= len(pattern):
            return False
        return pattern[position] == "N"


def create_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.1,
    scenarios: Optional[List[str]] = None,
    layouts: Optional[List[str]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    augment_train: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        root_dir: Root directory containing the data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        val_split: Fraction of data to use for validation
        scenarios: List of scenarios to include
        layouts: List of layouts to include
        image_size: Target size for LR images (h, w)
        augment_train: Whether to augment training data
        distributed: Whether to use DistributedSampler for DDP
        rank: Rank of current process (for DDP)
        world_size: Total number of processes (for DDP)

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create full dataset
    full_dataset = LicensePlateDataset(
        root_dir=root_dir,
        scenarios=scenarios,
        layouts=layouts,
        image_size=image_size,
        augment=False,  # We'll augment in the collate function or separately
    )

    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Enable augmentation for training set
    if augment_train:
        # We need to wrap the train subset to enable augmentation
        train_dataset.dataset.augment = True

    # Create dataloaders
    if distributed:
        # Use DistributedSampler for DDP training
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            timeout=300,  # 5 minute timeout for worker response
            persistent_workers=False,  # Recreate workers each epoch for DDP stability
        )

        # Validation: only rank 0 validates on full dataset
        # Other ranks skip validation (early return in validate() method)
        if rank == 0:
            # Rank 0 validates on FULL validation set (no DistributedSampler)
            # Use fewer workers to avoid resource exhaustion
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(num_workers, 4),  # Limit workers for validation
                pin_memory=True,
                drop_last=False,
                timeout=300,  # 5 minute timeout for worker response
                persistent_workers=False,  # Recreate workers each epoch for DDP stability
            )
        else:
            # Other ranks: None (completely skip validation dataloader)
            # This avoids multiprocessing issues in DDP
            val_loader = None
    else:
        # Single GPU training
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            timeout=300,  # 5 minute timeout for worker response
            persistent_workers=False,  # Recreate workers each epoch for stability
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=300,  # 5 minute timeout for worker response
            persistent_workers=False,  # Recreate workers each epoch for stability
        )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = LicensePlateDataset(
        root_dir="G:/LP-ASRN/data/train",
        image_size=(17, 31),  # (h, w) for LR images
    )

    print(f"Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"LR shape: {sample['lr'].shape}")
    print(f"HR shape: {sample['hr'].shape}")
    print(f"Plate text: {sample['plate_text']}")
    print(f"Plate layout: {sample['plate_layout']}")
