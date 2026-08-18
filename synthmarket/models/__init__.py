"""Neural architectures used by SynthMarket."""

from .base import ModelMetadata, SyntheticModel
from .registry import get_model, list_models, register_model
from .wgan import RecurrentCritic, RecurrentGenerator, WGANSequential

__all__ = [
    "ModelMetadata",
    "SyntheticModel",
    "RecurrentCritic",
    "RecurrentGenerator",
    "WGANSequential",
    "get_model",
    "list_models",
    "register_model",
]
