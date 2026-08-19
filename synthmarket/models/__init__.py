"""Model architectures and statistical baselines used by SynthMarket."""

from .base import ModelMetadata, SyntheticModel
from .baselines import BlockBootstrapModel, GaussianGarchModel, GarchParameters
from .registry import get_model, list_models, register_model

try:
    from .wgan import RecurrentCritic, RecurrentGenerator, WGANSequential
except ImportError:  # pragma: no cover - torch is an optional import for baseline-only use
    RecurrentCritic = None  # type: ignore[assignment,misc]
    RecurrentGenerator = None  # type: ignore[assignment,misc]
    WGANSequential = None  # type: ignore[assignment,misc]

__all__ = [
    "BlockBootstrapModel",
    "GaussianGarchModel",
    "GarchParameters",
    "ModelMetadata",
    "SyntheticModel",
    "RecurrentCritic",
    "RecurrentGenerator",
    "WGANSequential",
    "get_model",
    "list_models",
    "register_model",
]
