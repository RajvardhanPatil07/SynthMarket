"""Optional FastAPI service layer for hosted SynthMarket deployments."""

from __future__ import annotations

from typing import Any

from .models.registry import list_models

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[assignment,misc]


def create_app() -> Any:
    """Create the HTTP API application.

    FastAPI remains an optional dependency so the core research library keeps a
    small install footprint. Install ``synthmarket[api]`` for the service layer.
    """

    if FastAPI is None:
        raise ImportError("Install the API extra with: pip install 'synthmarket[api]'")

    app = FastAPI(title="SynthMarket API", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "synthmarket"}

    @app.get("/models")
    def models() -> dict[str, list[dict[str, Any]]]:
        return {"models": list_models()}

    return app


app = create_app() if FastAPI is not None else None
