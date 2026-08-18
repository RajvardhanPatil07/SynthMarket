"""Tests for the optional FastAPI service layer."""

from synthmarket.api import create_app


def test_api_factory() -> None:
    try:
        app = create_app()
    except ImportError:
        return
    assert app.title == "SynthMarket API"
