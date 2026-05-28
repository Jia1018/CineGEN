"""CineGen — text-conditioned camera trajectory generation."""

from cinegen.model import (
    CineGen,
    CineGenSequencer,
    CineGenDiffuser,
    IdentityAE,
    IdentityEncoder,
    AspectEncoder,
    CosineScheduler,
)

__version__ = "0.1.0"

__all__ = [
    "CineGen",
    "CineGenSequencer",
    "CineGenDiffuser",
    "IdentityAE",
    "IdentityEncoder",
    "AspectEncoder",
    "CosineScheduler",
]
