"""Optional FastAPI service layer for hosted SynthMarket deployments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models.registry import get_model, list_models

API_VERSION = "0.3.0"
MAX_PATHS = 100
MAX_HORIZON = 2520

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    BaseModel = object  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment,misc]


def create_app() -> Any:
    """Create the HTTP API application."""
    if FastAPI is None:
        raise ImportError("Install the API extra with: pip install 'synthmarket[api]'")

    class GenerateRequest(BaseModel):
        close: list[float] = Field(min_length=31)
        model: str = "block-bootstrap"
        n_paths: int = Field(default=10, ge=1, le=MAX_PATHS)
        horizon: int = Field(default=252, ge=2, le=MAX_HORIZON)
        seed: int | None = None

    application = FastAPI(title="SynthMarket API", version=API_VERSION)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "synthmarket", "version": API_VERSION}

    @application.get("/models")
    def models() -> dict[str, list[dict[str, Any]]]:
        return {"models": list_models()}

    @application.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        if request.model not in {"block-bootstrap", "garch"}:
            raise HTTPException(status_code=404, detail=f"Unknown or unsupported model '{request.model}'.")
        close = np.asarray(request.close, dtype=float)
        if not np.isfinite(close).all() or np.any(close <= 0.0):
            raise HTTPException(status_code=422, detail="close must contain finite positive prices.")
        returns = np.diff(np.log(close))
        model = get_model(request.model)
        model.fit(returns.reshape(1, -1, 1))
        generated = model.generate(request.n_paths, request.horizon, seed=request.seed)[:, :, 0]
        prices = close[-1] * np.exp(np.cumsum(generated, axis=1))
        return {
            "model": request.model,
            "n_paths": request.n_paths,
            "horizon": request.horizon,
            "seed": request.seed,
            "paths": prices.tolist(),
        }

    return application


app = create_app() if FastAPI is not None else None
