# Losses module
from .lcofl import LCOFL, ClassificationLoss, LayoutPenalty
from .basic import L1Loss, SSIMLoss, PerceptualLoss
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
    "EmbeddingConsistencyLoss",
    "TripletEmbeddingLoss",
    "CosineEmbeddingLoss",
]
