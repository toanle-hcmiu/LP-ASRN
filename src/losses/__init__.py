# Losses module
from .lcofl import LCOFL, ClassificationLoss, LayoutPenalty
from .basic import (
    L1Loss,
    SSIMLoss,
    PerceptualLoss,
    GradientLoss,
    FrequencyLoss,
    EdgeLoss,
    CharbonnierLoss,
)
from .gan_loss import (
    Discriminator,
    GANLoss,
    FeatureMatchingLoss,
    RelativisticGANLoss,
    MultiScaleDiscriminator,
)
from .embedding_loss import (
    EmbeddingConsistencyLoss,
    TripletEmbeddingLoss,
    CosineEmbeddingLoss,
)

__all__ = [
    "LCOFL",
    "ClassificationLoss",
    "LayoutPenalty",
    "L1Loss",
    "SSIMLoss",
    "PerceptualLoss",
    "GradientLoss",
    "FrequencyLoss",
    "EdgeLoss",
    "CharbonnierLoss",
    "Discriminator",
    "GANLoss",
    "FeatureMatchingLoss",
    "RelativisticGANLoss",
    "MultiScaleDiscriminator",
    "EmbeddingConsistencyLoss",
    "TripletEmbeddingLoss",
    "CosineEmbeddingLoss",
]
