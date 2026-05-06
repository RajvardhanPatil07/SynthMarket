"""Neural architectures used by SynthMarket."""

from .wgan import RecurrentCritic, RecurrentGenerator, WGANSequential

__all__ = ["RecurrentCritic", "RecurrentGenerator", "WGANSequential"]

