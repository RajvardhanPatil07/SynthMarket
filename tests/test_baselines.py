import numpy as np

from synthmarket.models.baselines import BlockBootstrapModel, GaussianGarchModel
from synthmarket.models.registry import get_model, list_models


def _windows(seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0004, 0.01, size=(4, 200, 2))


def _garch_series(seed: int = 5, length: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 5e-6, 0.10, 0.85
    variance = omega / (1 - alpha - beta)
    output = np.empty(length)
    for step in range(length):
        shock = rng.standard_normal() * np.sqrt(variance)
        output[step] = shock
        variance = omega + alpha * shock**2 + beta * variance
    return output


def test_registry_contains_statistical_baselines() -> None:
    names = {entry["name"] for entry in list_models()}
    assert {"block-bootstrap", "garch"} <= names
    assert isinstance(get_model("block-bootstrap"), BlockBootstrapModel)


def test_block_bootstrap_shapes_determinism_and_observed_rows() -> None:
    windows = _windows()
    model = BlockBootstrapModel().fit(windows)
    first = model.generate(5, 50, seed=3)
    second = model.generate(5, 50, seed=3)
    assert first.shape == (5, 50, 2)
    assert np.array_equal(first, second)
    observed = {tuple(row) for row in windows.reshape(-1, 2).round(12).tolist()}
    assert all(tuple(row) in observed for row in first.reshape(-1, 2).round(12).tolist())


def test_block_bootstrap_save_load_roundtrip(tmp_path) -> None:
    model = BlockBootstrapModel(block_length=7).fit(_windows())
    restored = BlockBootstrapModel.load(model.save(tmp_path / "bootstrap.npz"))
    assert restored.block_length == 7
    assert np.array_equal(restored.generate(3, 20, seed=2), model.generate(3, 20, seed=2))


def test_garch_recovers_persistence_and_scale() -> None:
    series = _garch_series()
    model = GaussianGarchModel().fit(series.reshape(1, -1, 1))
    parameters = model.parameters
    assert parameters is not None
    assert 0.5 < parameters[0].alpha + parameters[0].beta < 0.999
    generated = model.generate(20, 500, seed=4)
    assert generated.shape == (20, 500, 1)
    assert np.isfinite(generated).all()
    assert 0.6 < generated.std() / series.std() < 1.6


def test_garch_determinism_and_save_load(tmp_path) -> None:
    model = GaussianGarchModel(seed=9).fit(_garch_series().reshape(1, -1, 1))
    expected = model.generate(4, 60, seed=9)
    assert np.array_equal(expected, model.generate(4, 60, seed=9))
    restored = GaussianGarchModel.load(model.save(tmp_path / "garch.json"))
    assert np.array_equal(restored.generate(4, 60, seed=9), expected)
