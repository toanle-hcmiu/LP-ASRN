"""
Siamese Embedding Network for LP-SR (LCOFL-EC)

Based on: Sendjasni et al., "Embedding Consistency Loss for Enhanced
Super-Resolution" (2025)

Uses a frozen pre-trained ResNet-18 backbone to extract embeddings
from SR and HR images, then computes L2-normalized embeddings for
contrastive loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SiameseEmbedder(nn.Module):
    """
    Siamese Embedding Network for LP-SR.

    Uses a frozen pre-trained ResNet-18 backbone to extract embeddings
    from SR and HR images, then computes L2-normalized embeddings.

    The backbone is frozen to provide stable semantic features during training.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ):
        """
        Initialize Siamese Embedder.

        Args:
            embedding_dim: Dimension of the output embedding vector
            backbone: Backbone architecture ('resnet18' or 'resnet34')
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.backbone_name = backbone

        # Load pretrained ResNet
        if backbone == "resnet18":
            resnet = torch.hub.load(
                'pytorch/vision:v0.13.0', 'resnet18', pretrained=pretrained
            ) if pretrained else torch.hub.load(
                'pytorch/vision:v0.13.0', 'resnet18', pretrained=False
            )
            backbone_out_dim = 512
        elif backbone == "resnet34":
            resnet = torch.hub.load(
                'pytorch/vision:v0.13.0', 'resnet34', pretrained=pretrained
            ) if pretrained else torch.hub.load(
                'pytorch/vision:v0.13.0', 'resnet34', pretrained=False
            )
            backbone_out_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the classification head (fc layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Projection head to embedding dimension
        # Uses an MLP-style architecture
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Light dropout for regularization
            nn.Linear(256, embedding_dim),
        )

        # Initialize projection head
        self._init_projection_weights()

    def _init_projection_weights(self):
        """Initialize projection head weights using Xavier initialization."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features from input images.

        Args:
            x: Input tensor of shape (B, 3, H, W) in range [-1, 1] or [0, 1]

        Returns:
            Feature tensor of shape (B, backbone_out_dim)
        """
        # Normalize to [0, 1] if in [-1, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0

        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std

        # Extract features
        with torch.no_grad():  # Backbone is frozen, no need for gradients
            features = self.backbone(x)  # (B, 512, 1, 1)

        # Global average pooling
        features = features.view(features.size(0), -1)  # (B, 512)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized embedding from image.

        Args:
            x: Input tensor of shape (B, 3, H, W) in range [-1, 1] or [0, 1]

        Returns:
            L2-normalized embedding tensor of shape (B, embedding_dim)
        """
        # Extract backbone features
        features = self.extract_features(x)

        # Project to embedding dimension
        embedding = self.projection(features)  # (B, embedding_dim)

        # L2 normalize (project onto unit hypersphere)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def compute_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        distance_type: str = "manhattan"
    ) -> torch.Tensor:
        """
        Compute distance between embeddings of two image batches.

        Args:
            x1: First image batch (B, 3, H, W)
            x2: Second image batch (B, 3, H, W)
            distance_type: Type of distance ('manhattan' or 'euclidean')

        Returns:
            Distance tensor of shape (B,)
        """
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)

        if distance_type == "manhattan":
            dist = torch.sum(torch.abs(emb1 - emb2), dim=1)
        elif distance_type == "euclidean":
            dist = torch.norm(emb1 - emb2, p=2, dim=1)
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")

        return dist

    def get_trainable_params(self) -> list:
        """Get list of trainable parameters (only projection head)."""
        return list(self.projection.parameters())


class LightweightSiameseEmbedder(nn.Module):
    """
    Lightweight Siamese Embedder with custom backbone.

    Uses a custom lightweight CNN backbone instead of ResNet for
    faster training when GPU memory is limited.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        base_channels: int = 32,
    ):
        """
        Initialize Lightweight Siamese Embedder.

        Args:
            embedding_dim: Dimension of the output embedding vector
            base_channels: Base number of channels in the CNN
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Lightweight backbone (custom CNN)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),

            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(base_channels * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized embedding from image.

        Args:
            x: Input tensor of shape (B, 3, H, W) in range [-1, 1] or [0, 1]

        Returns:
            L2-normalized embedding tensor of shape (B, embedding_dim)
        """
        # Normalize to [0, 1] if in [-1, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0

        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)

        # Project to embedding dimension
        embedding = self.projection(features)

        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


if __name__ == "__main__":
    # Test the Siamese Embedder
    print("Testing Siamese Embedder...")

    # Test with dummy images
    batch_size = 4
    sr_images = torch.randn(batch_size, 3, 64, 128) * 2 - 1
    hr_images = torch.randn(batch_size, 3, 64, 128) * 2 - 1

    # Test standard SiameseEmbedder
    embedder = SiameseEmbedder(embedding_dim=128)

    sr_embeddings = embedder(sr_images)
    hr_embeddings = embedder(hr_images)

    print(f"SR embeddings shape: {sr_embeddings.shape}")
    print(f"HR embeddings shape: {hr_embeddings.shape}")

    # Check L2 normalization
    sr_norms = torch.norm(sr_embeddings, p=2, dim=1)
    print(f"SR embedding norms (should be ~1.0): {sr_norms}")

    # Test distance computation
    manhattan_dist = embedder.compute_distance(sr_images, hr_images, "manhattan")
    euclidean_dist = embedder.compute_distance(sr_images, hr_images, "euclidean")

    print(f"Manhattan distances: {manhattan_dist}")
    print(f"Euclidean distances: {euclidean_dist}")

    # Test identical images (should have zero distance)
    identical_dist = embedder.compute_distance(hr_images, hr_images, "manhattan")
    print(f"Distance between identical images: {identical_dist}")

    # Test lightweight version
    print("\nTesting Lightweight Siamese Embedder...")
    lite_embedder = LightweightSiameseEmbedder(embedding_dim=128)

    lite_embeddings = lite_embedder(sr_images)
    print(f"Lightweight embeddings shape: {lite_embeddings.shape}")

    lite_params = sum(p.numel() for p in lite_embedder.parameters())
    print(f"Lightweight embedder parameters: {lite_params:,}")

    print("\nSiamese Embedder test passed!")
