"""Model registry used by the API and research tooling."""

from __future__ import annotations

from typing import Any, Callable

from .base import ModelMetadata

_REGISTRY: dict[str, tuple[ModelMetadata, Callable[..., Any]]] = {}


def register_model(name: str, metadata: ModelMetadata, factory: Callable[..., Any]) -> None:
    """Register a named model factory."""

    key = name.strip().lower()
    if not key:
        raise ValueError("Model name must not be empty.")
    if key in _REGISTRY:
        raise ValueError(f"Model '{key}' is already registered.")
    _REGISTRY[key] = (metadata, factory)


def get_model(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered model."""

    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available models: {', '.join(list(_REGISTRY)) or 'none'}")
    return _REGISTRY[key][1](**kwargs)


def list_models() -> list[dict[str, Any]]:
    """Return serializable metadata for all registered models."""

    return [
        {
            "name": metadata.name,
            "version": metadata.version,
            "family": metadata.family,
            "capabilities": list(metadata.capabilities),
        }
        for metadata, _ in _REGISTRY.values()
    ]


def _register_baseline_wgan() -> None:
    try:
        from .wgan import WGANConfig, WGANSequential
    except ImportError:
        return

    metadata = ModelMetadata(
        name="wgan-gp",
        version="1",
        family="wasserstein-gan",
        capabilities=("single_asset", "multi_asset", "sequence_generation"),
    )

    def factory(config: WGANConfig | None = None) -> WGANSequential:
        return WGANSequential(config or WGANConfig())

    register_model("wgan-gp", metadata, factory)


_register_baseline_wgan()
