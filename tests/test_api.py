import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from synthmarket.api import create_app


def test_health_models_and_generation() -> None:
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["version"] == "0.3.0"

    models = client.get("/models")
    assert models.status_code == 200
    assert {entry["name"] for entry in models.json()["models"]} >= {"block-bootstrap", "garch"}

    response = client.post(
        "/generate",
        json={"close": [100 + index * 0.1 for index in range(80)], "model": "garch", "n_paths": 2, "horizon": 30, "seed": 7},
    )
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert len(paths) == 2
    assert all(len(path) == 30 for path in paths)
    assert all(price > 0 for path in paths for price in path)


def test_unknown_model_returns_404() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/generate",
        json={"close": [100 + index * 0.1 for index in range(40)], "model": "missing", "n_paths": 1, "horizon": 2},
    )
    assert response.status_code == 404
