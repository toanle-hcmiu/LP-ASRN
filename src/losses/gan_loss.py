"""
GAN Loss for Super-Resolution with Discriminator.

Implements:
- PatchGAN discriminator (classifies local image patches)
- Hinge loss for GAN training
- Feature matching loss for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for SR.

    Classifies each N×N patch as real or fake rather than the entire image.
    This preserves more high-frequency structure.

    Architecture: Conv layers with leaky ReLU, no pooling.
    Output: (B, 1, H/16, W/16) - each position is a patch prediction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 64,
        num_layers: int = 3,
    ):
        """
        Initialize PatchGAN Discriminator.

        Args:
            in_channels: Number of input channels (3 for RGB)
            num_filters: Base number of filters
            num_layers: Number of conv layers (default 3 for 64x64 -> 4x4 patches)
        """
        super().__init__()

        layers = []

        # Layer 0: (B, 3, H, W) -> (B, 64, H/2, W/2)
        layers += [
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Layer 1+: Each halves spatial dimensions, doubles filters
        for i in range(1, num_layers):
            mult = 2 ** i
            layers += [
                nn.Conv2d(
                    num_filters * mult // 2,
                    num_filters * mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(num_filters * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Final layer: (B, 512, H/16, W/16) -> (B, 1, H/16, W/16)
        mult = 2 ** num_layers
        layers += [
            nn.Conv2d(num_filters * mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W) in range [-1, 1]

        Returns:
            Patch predictions (B, 1, H/16, W/16), each value is logit (before sigmoid)
        """
        return self.model(x)

    @staticmethod
    def get_num_params(in_channels: int = 3, num_filters: int = 64, num_layers: int = 3) -> int:
        """Calculate total parameters."""
        # Simplified - just return estimate
        return num_filters * 8 * 10000  # ~400K params default


class GANLoss(nn.Module):
    """
    GAN Loss with Hinge loss formulation.

    Uses Hinge loss which is more stable than BCE for GANs:
    - D loss: max(0, 1 - D(real)) + max(0, 1 + D(fake))
    - G loss: -D(fake)  (generator wants D to think fake is real)
    """

    def __init__(self):
        """Initialize GAN Loss."""
        super().__init__()

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss (Hinge loss).

        Args:
            real_logits: D(real) output, shape (B, 1, H, W)
            fake_logits: D(fake) output, shape (B, 1, H, W)

        Returns:
            Discriminator loss scalar
        """
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss (Hinge loss).

        Args:
            fake_logits: D(fake) output, shape (B, 1, H, W)

        Returns:
            Generator loss scalar
        """
        return -fake_logits.mean()


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss for stable GAN training.

    Matches intermediate features of the discriminator between
    real and fake images. This provides additional gradients
    and prevents mode collapse.

    L = Σ |D^i(real) - D^i(fake)|
    """

    def __init__(self, discriminator: nn.Module):
        """
        Initialize Feature Matching Loss.

        Args:
            discriminator: The discriminator to extract features from
        """
        super().__init__()
        self.discriminator = discriminator

    def forward(
        self,
        real_imgs: torch.Tensor,
        fake_imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            real_imgs: Real HR images (B, 3, H, W)
            fake_imgs: Generated SR images (B, 3, H, W)

        Returns:
            Feature matching loss scalar
        """
        loss = 0.0
        num_features = 0

        # Extract intermediate features from discriminator
        x_real = real_imgs
        x_fake = fake_imgs

        for i, layer in enumerate(self.discriminator.model):
            # Skip the final layer (no batchnorm, different structure)
            if isinstance(layer, nn.Conv2d) and i == len(self.discriminator.model) - 1:
                break

            x_real = layer(x_real)
            x_fake = layer(x_fake)

            # Only match at ReLU activations (after batchnorm if present)
            if isinstance(layer, nn.LeakyReLU):
                loss = loss + F.l1_loss(x_real, x_fake)
                num_features += 1

        if num_features > 0:
            loss = loss / num_features

        return loss


class RelativisticGANLoss(nn.Module):
    """
    Relativistic GAN Loss (RaGAN).

    The discriminator predicts whether real is more realistic than fake,
    rather than predicting real/fake independently. This produces
    sharper gradients and better results.

    Paper: RaGAN: Relativistic GAN
    """

    def __init__(self, average: bool = True):
        """
        Initialize Relativistic GAN Loss.

        Args:
            average: Use average over all real/fake pairs
        """
        super().__init__()
        self.average = average

    def _discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RaGAN discriminator loss.

        Args:
            real_logits: D(real) predictions
            fake_logits: D(fake) predictions

        Returns:
            Discriminator loss
        """
        if self.average:
            # Average discriminator prediction
            real_loss = F.binary_cross_entropy_with_logits(
                real_logits - fake_logits.mean(dim=0, keepdim=True),
                torch.ones_like(real_logits),
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logits - real_logits.mean(dim=0, keepdim=True),
                torch.zeros_like(fake_logits),
            )
        else:
            # Standard RaGAN (sigmoid on difference)
            real_loss = F.binary_cross_entropy_with_logits(
                real_logits - fake_logits,
                torch.ones_like(real_logits),
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logits - real_logits,
                torch.zeros_like(fake_logits),
            )

        return (real_loss + fake_loss) / 2

    def _generator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RaGAN generator loss.

        Args:
            real_logits: D(real) predictions
            fake_logits: D(fake) predictions

        Returns:
            Generator loss
        """
        if self.average:
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logits - real_logits.mean(dim=0, keepdim=True),
                torch.ones_like(fake_logits),
            )
        else:
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_logits - real_logits,
                torch.ones_like(fake_logits),
            )

        return fake_loss

    def forward(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        for_generator: bool = False,
    ) -> torch.Tensor:
        """
        Compute RaGAN loss.

        Args:
            real_logits: D(real) predictions
            fake_logits: D(fake) predictions
            for_generator: If True, return generator loss

        Returns:
            GAN loss scalar
        """
        if for_generator:
            return self._generator_loss(real_logits, fake_logits)
        else:
            return self._discriminator_loss(real_logits, fake_logits)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better perceptual quality.

    Uses multiple discriminators at different scales to capture
    both fine details and global structure.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 64,
        num_scales: int = 3,
    ):
        """
        Initialize Multi-scale Discriminator.

        Args:
            in_channels: Number of input channels
            num_filters: Base number of filters
            num_scales: Number of scales (default 3)
        """
        super().__init__()

        self.discriminators = nn.ModuleList()
        self.num_scales = num_scales

        for _ in range(num_scales):
            self.discriminators.append(
                Discriminator(in_channels, num_filters, num_layers=3)
            )

        # Downsampling for multi-scale input
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> Tuple[list, list]:
        """
        Forward pass on multiple scales.

        Args:
            real: Real HR images (B, 3, H, W)
            fake: Generated SR images (B, 3, H, W)

        Returns:
            (real_logits_list, fake_logits_list) for each scale
        """
        real_logits = []
        fake_logits = []

        real_input = real
        fake_input = fake

        for i, disc in enumerate(self.discriminators):
            real_logits.append(disc(real_input))
            fake_logits.append(disc(fake_input))

            # Downsample for next scale
            if i < self.num_scales - 1:
                real_input = self.downsample(real_input)
                fake_input = self.downsample(fake_input)

        return real_logits, fake_logits


if __name__ == "__main__":
    # Test discriminator
    print("Testing GAN components...")

    B, C, H, W = 4, 3, 128, 256

    # Test discriminator
    disc = Discriminator(in_channels=3, num_filters=64, num_layers=3)
    real = torch.randn(B, C, H, W)
    fake = torch.randn(B, C, H, W)

    real_out = disc(real)
    fake_out = disc(fake)

    print(f"Real output shape: {real_out.shape}")
    print(f"Fake output shape: {fake_out.shape}")

    # Test GAN loss
    gan_loss = GANLoss()
    d_loss = gan_loss.discriminator_loss(real_out, fake_out)
    g_loss = gan_loss.generator_loss(fake_out)

    print(f"Discriminator loss: {d_loss.item():.4f}")
    print(f"Generator loss: {g_loss.item():.4f}")

    # Test feature matching
    fm_loss = FeatureMatchingLoss(disc)
    fm = fm_loss(real, fake)
    print(f"Feature matching loss: {fm.item():.4f}")

    # Test multi-scale discriminator
    ms_disc = MultiScaleDiscriminator(in_channels=3, num_scales=2)
    real_logits, fake_logits = ms_disc(real, fake)

    print(f"Multi-scale: {len(real_logits)} scales")
    for i, (rl, fl) in enumerate(zip(real_logits, fake_logits)):
        print(f"  Scale {i}: real={rl.shape}, fake={fl.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator parameters: {num_params:,}")

    print("GAN components test passed!")
