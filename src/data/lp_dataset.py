"""
License Plate Dataset for Super-Resolution Training
Supports both Brazilian and Mercosur layouts with paired LR/HR images.
"""

import os
import json
import random
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
        ocr_pretrain_mode: bool = False,
        aspect_ratio_augment: bool = False,
        test_aspect_range: Tuple[float, float] = (0.29, 0.40),
        test_resolution_augment: bool = False,
        test_resolution_prob: float = 0.5,
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
            ocr_pretrain_mode: If True, use OCR-specific augmentation for pretraining
            aspect_ratio_augment: If True, randomly vary aspect ratio during training
                to match test-time distribution (test images have different ratios
                than training crops). This pads images before resize to simulate
                different crop sizes.
            test_aspect_range: (min_ratio, max_ratio) of test image aspect ratios (H/W).
                Used to sample random target ratios during augmentation.
                Default (0.29, 0.40) matches observed test-public distribution.
            test_resolution_augment: If True, randomly downsample LR images to match
                test-time resolution (test images are ~17x49 vs training ~30x87).
            test_resolution_prob: Probability (0-1) of applying test-resolution augmentation.
        """
        self.root_dir = Path(root_dir)
        self.scenarios = scenarios or ["Scenario-A", "Scenario-B"]
        self.layouts = layouts or ["Brazilian", "Mercosur"]
        self.image_size = image_size
        self.crop_to_corners = crop_to_corners
        self.augment = augment
        self.normalize = normalize
        self.ocr_pretrain_mode = ocr_pretrain_mode
        self.aspect_ratio_augment = aspect_ratio_augment
        self.test_aspect_range = test_aspect_range
        self.test_resolution_augment = test_resolution_augment
        self.test_resolution_prob = test_resolution_prob

        # Load all samples
        self.samples = self._load_samples()

        # Setup augmentation (OCR-specific or standard)
        # Fixed: Split into geometric (both LR/HR) and photometric degradation (LR only)
        if augment:
            self.geometric_transform = self._get_geometric_transform(ocr_pretrain_mode)
            self.photometric_degradation = self._get_photometric_degradation(ocr_pretrain_mode)
        else:
            self.geometric_transform = None
            self.photometric_degradation = None

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

    def _get_ocr_augment_transform(self) -> transforms.Compose:
        """Enhanced augmentation pipeline for OCR pretraining."""
        return transforms.Compose([
            # Stronger geometric transforms
            transforms.RandomAffine(
                degrees=10,              # More rotation
                translate=(0.1, 0.15),   # More translation
                scale=(0.85, 1.15),      # Wider scale
                shear=8,                 # More shear
            ),
            # Perspective warp for camera angle simulation
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            # Photometric: more aggressive
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.3,
                hue=0.15,
            ),
            # More blur/noise
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0)),
            ], p=0.4),
            transforms.RandomApply([
                AddGaussianNoise(mean=0, std=0.08),
            ], p=0.4),
        ])

    def _get_geometric_transform(self, ocr_pretrain_mode: bool = False) -> transforms.Compose:
        """
        Get geometric transforms (applied to both LR and HR for consistency).
        These preserve spatial correspondence between LR and HR.

        ENHANCED: Increased affine range and added perspective transform.
        """
        if ocr_pretrain_mode:
            return transforms.Compose([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.15),
                    scale=(0.85, 1.15),
                    shear=8,
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.2,
                    p=0.5
                ),
            ])
        else:
            # STRONG augmentation for better generalization
            return transforms.Compose([
                transforms.RandomAffine(
                    degrees=15,    # More rotation
                    translate=(0.15, 0.2),  # More translation
                    scale=(0.8, 1.2),  # Wider scale
                    shear=10,   # More shear
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.25,  # More perspective distortion
                    p=0.6   # 60% chance
                ),
            ])

    def _get_photometric_degradation(self, ocr_pretrain_mode: bool = False) -> transforms.Compose:
        """
        Get photometric degradation transforms (applied ONLY to LR).
        This simulates real-world degradation while keeping HR clean.

        Enhanced with extreme degradation to better match test distribution.
        """
        if ocr_pretrain_mode:
            # ENHANCED: Added heavier degradation for OCR pretraining robustness
            return transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.6,  # Increased from 0.5
                    contrast=0.6,   # Increased from 0.5
                    saturation=0.4, # Increased from 0.3
                    hue=0.2,       # Increased from 0.15
                ),
                # Standard blur (40% chance)
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0)),
                ], p=0.4),
                # HEAVY blur for extreme cases (25% chance)
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 5.0)),
                ], p=0.25),
                # Standard noise (40% chance)
                transforms.RandomApply([
                    AddGaussianNoise(mean=0, std=0.08),
                ], p=0.4),
                # STRONGER noise for extreme cases (25% chance)
                transforms.RandomApply([
                    AddGaussianNoise(mean=0, std=0.015),
                ], p=0.25),
            ])
        else:
            # STRONG photometric degradation for better robustness
            return transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.6,  # More brightness variation
                    contrast=0.6,   # More contrast variation
                    saturation=0.4, # More saturation variation
                    hue=0.2,       # More hue shift
                ),
                # Standard blur (40% chance)
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0)),
                ], p=0.4),
                # HEAVY blur for extreme cases (30% chance)
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 6.0)),
                ], p=0.3),
                # Standard noise (40% chance)
                transforms.RandomApply([
                    AddGaussianNoise(mean=0, std=0.08),
                ], p=0.4),
                # STRONGER noise for extreme cases (30% chance)
                transforms.RandomApply([
                    AddGaussianNoise(mean=0, std=0.02),
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

    def _apply_aspect_ratio_augment(self, image: Image.Image) -> Image.Image:
        """
        Randomly change the aspect ratio of a cropped plate image to simulate
        the test-time distribution.

        Test images have aspect ratios (H/W) of 0.29-0.40, while training crops
        resized to 34×62 have ratio 0.55. This augmentation pads the image to a
        random target ratio BEFORE the final resize, so the generator learns to
        handle varied aspect ratios.

        Applied with 50% probability to keep some training at the native ratio.

        Args:
            image: Cropped PIL image (before resize)

        Returns:
            Padded PIL image with randomized aspect ratio
        """
        if not self.aspect_ratio_augment or not self.augment:
            return image

        # NOTE: Removed 50% probability skip - always apply for test distribution consistency
        # This ensures the model learns to handle test-time aspect ratios (0.29-0.40)

        orig_w, orig_h = image.size
        orig_ratio = orig_h / orig_w

        # Sample a random target aspect ratio
        # Blend between test range and training range for smooth distribution
        min_ratio, max_ratio = self.test_aspect_range
        if self.image_size is not None:
            train_ratio = self.image_size[0] / self.image_size[1]
        else:
            train_ratio = orig_ratio

        # 70% chance: sample from test range, 30% chance: between test and train
        if random.random() < 0.7:
            target_ratio = random.uniform(min_ratio, max_ratio)
        else:
            target_ratio = random.uniform(min(min_ratio, train_ratio),
                                          max(max_ratio, train_ratio))

        if abs(target_ratio - orig_ratio) < 0.02:
            return image  # Already close enough

        img_arr = np.array(image)

        if target_ratio < orig_ratio:
            # Target is wider — pad width (add columns on sides)
            new_w = int(orig_h / target_ratio)
            pad_total = new_w - orig_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            img_arr = np.pad(img_arr,
                             ((0, 0), (pad_left, pad_right), (0, 0)),
                             mode='edge')
        else:
            # Target is taller — pad height (add rows top/bottom)
            new_h = int(orig_w * target_ratio)
            pad_total = new_h - orig_h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            img_arr = np.pad(img_arr,
                             ((pad_top, pad_bottom), (0, 0), (0, 0)),
                             mode='edge')

        return Image.fromarray(img_arr)

    def _apply_test_resolution(self, lr_image: Image.Image) -> Image.Image:
        """
        Randomly downsample LR image to simulate test-time resolution.

        Test images are significantly smaller (avg 17x49) than training images
        (avg 30x87). This augmentation downsamples training LR images to test
        resolution so the model learns to handle low-resolution inputs.

        Args:
            lr_image: PIL LR image (after cropping, before aspect ratio augment)

        Returns:
            Downsampled PIL image or original (based on probability)
        """
        if not self.test_resolution_augment or not self.augment:
            return lr_image

        if random.random() > self.test_resolution_prob:
            return lr_image  # Skip this augmentation

        # Sample from test distribution (from analyze_data_mismatch.py results)
        # Test: height=17.9±1.6, width=49.1±4.8
        target_h = int(random.gauss(17.9, 1.6))
        target_w = int(random.gauss(49.1, 4.8))
        target_h = max(12, min(25, target_h))  # Clamp to reasonable range
        target_w = max(35, min(65, target_w))

        # Downsample LR image (simulates low-res source like test images)
        try:
            return lr_image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        except AttributeError:
            # For older Pillow versions
            return lr_image.resize((target_w, target_h), Image.BICUBIC)

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
        """
        Apply augmentation - geometric to both LR/HR, degradation only to LR.
        Fixed: Previously applied same augmentation (including blur/noise) to both LR and HR,
        which defeated the purpose of degraded→clean training.
        """
        if self.geometric_transform is None:
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

        # Apply geometric transforms to both LR and HR (same seed for consistency)
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        lr_aug = self.geometric_transform(lr_pil)
        torch.manual_seed(seed)
        hr_aug = self.geometric_transform(hr_pil)

        # Apply photometric degradation ONLY to LR (blur, noise, color jitter)
        # HR stays clean - this is the target we want the model to learn
        if self.photometric_degradation is not None:
            lr_aug = self.photometric_degradation(lr_aug)

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

        # Test-resolution augmentation: downsample LR to match test distribution
        # Applied BEFORE aspect ratio augmentation so both work correctly
        # Only applied to LR (HR stays high-res as the target)
        if self.augment and self.test_resolution_augment:
            lr_image = self._apply_test_resolution(lr_image)
            # HR is not downsampled - it's the high-quality target

        # Aspect ratio augmentation: randomly pad to simulate test-time aspect ratios
        # Applied BEFORE resize so the generator learns to handle varied ratios
        # Must use same padding for both LR and HR to keep them aligned
        # NOTE: Now applied during validation too to match test distribution
        if self.aspect_ratio_augment:
            # Save random state to apply identical padding to both LR and HR
            rng_state = random.getstate()
            np_rng_state = np.random.get_state()

            lr_image = self._apply_aspect_ratio_augment(lr_image)

            # Restore and re-apply for HR (same random decisions)
            random.setstate(rng_state)
            np.random.set_state(np_rng_state)
            hr_image = self._apply_aspect_ratio_augment(hr_image)

        # Multi-scale training: randomly resize to smaller scales (30% chance)
        # This helps the model handle test images which are often smaller than training
        if self.augment and self.image_size is not None and random.random() < 0.3:
            scale = random.choice([0.7, 0.8, 0.9])
            scaled_h = int(self.image_size[0] * scale)
            scaled_w = int(self.image_size[1] * scale)
            # Apply temporary resize before final resize
            lr_image = lr_image.resize((scaled_w, scaled_h), Image.BICUBIC)

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

    def set_aspect_ratio_range(self, aspect_ratio_range: Tuple[float, float]):
        """
        Update the aspect ratio range for augmentation.

        This allows progressive training stages to use different aspect ratio ranges.

        Args:
            aspect_ratio_range: (min_ratio, max_ratio) for aspect ratio augmentation
        """
        self.test_aspect_range = aspect_ratio_range

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
    ocr_pretrain_mode: bool = False,
    aspect_ratio_augment: bool = False,
    test_aspect_range: Tuple[float, float] = (0.29, 0.40),
    test_resolution_augment: bool = False,
    test_resolution_prob: float = 0.5,
) -> Tuple[DataLoader, DataLoader, Optional["DistributedSampler"]]:
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
        ocr_pretrain_mode: If True, use OCR-specific augmentation for pretraining
        aspect_ratio_augment: If True, randomly vary aspect ratio to match test distribution
        test_aspect_range: (min_ratio, max_ratio) of test image H/W ratios
        test_resolution_augment: If True, randomly downsample LR to match test resolution
        test_resolution_prob: Probability (0-1) of applying test-resolution augmentation

    Returns:
        Tuple of (train_dataloader, val_dataloader, train_sampler)
        train_sampler is None when distributed=False
    """
    # Create full dataset
    full_dataset = LicensePlateDataset(
        root_dir=root_dir,
        scenarios=scenarios,
        layouts=layouts,
        image_size=image_size,
        augment=False,  # We'll augment in the collate function or separately
        ocr_pretrain_mode=ocr_pretrain_mode,
        aspect_ratio_augment=aspect_ratio_augment,
        test_aspect_range=test_aspect_range,
        test_resolution_augment=test_resolution_augment,
        test_resolution_prob=test_resolution_prob,
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

        # Reduce workers for DDP to avoid BrokenPipeError with mp.spawn
        ddp_workers = min(num_workers, 2)  # Max 2 workers per GPU in DDP

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=ddp_workers,
            pin_memory=True,
            drop_last=True,
            timeout=300,  # 5 minute timeout for worker response
            persistent_workers=ddp_workers > 0,  # Keep workers alive between batches
            multiprocessing_context='spawn' if ddp_workers > 0 else None,  # Match mp.spawn
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
                num_workers=ddp_workers,  # Same workers as train for consistency
                pin_memory=True,
                drop_last=False,
                timeout=300,  # 5 minute timeout for worker response
                persistent_workers=ddp_workers > 0,
                multiprocessing_context='spawn' if ddp_workers > 0 else None,
            )
        else:
            # Other ranks: None (completely skip validation dataloader)
            # This avoids multiprocessing issues in DDP
            val_loader = None
    else:
        # Single GPU training
        # When num_workers=0, timeout must be 0 (no workers to timeout)
        loader_timeout = 0 if num_workers == 0 else 300
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            timeout=loader_timeout,
            persistent_workers=num_workers > 0,  # Only use persistent workers when num_workers > 0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=loader_timeout,
            persistent_workers=num_workers > 0,
        )

    if distributed:
        return train_loader, val_loader, train_sampler
    else:
        return train_loader, val_loader, None


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
