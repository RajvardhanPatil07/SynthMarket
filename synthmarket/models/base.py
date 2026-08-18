"""Stable interfaces shared by SynthMarket generative model backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    """Describes a model backend without depending on its implementation."""

    name: str
    version: str = "0.1"
    family: str = "unknown"
    capabilities: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class SyntheticModel(Protocol):
    """Protocol implemented by every synthetic-market model backend."""

    metadata: ModelMetadata

    def fit(self, windows: np.ndarray, **kwargs: Any) -> Any:
        """Fit the model on an array shaped ``(N, T, F)``."""
        ...

    def generate(self, n_paths: int, length: int, **kwargs: Any) -> np.ndarray:
        """Generate synthetic sequences shaped ``(N, T, F)``."""
        ...

    def save(self, path: str | Path) -> Path:
        """Persist the model to a checkpoint path."""
        ...

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "SyntheticModel":
        """Restore a persisted model."""
        ...
