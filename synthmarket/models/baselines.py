"""Dependency-light statistical baseline models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import ModelMetadata


def _as_windows(windows: np.ndarray) -> np.ndarray:
    array = np.asarray(windows, dtype=float)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError("windows must have shape (n_windows, sequence_length, n_features).")
    if not np.isfinite(array).all():
        raise ValueError("windows must contain only finite values.")
    return array


class BlockBootstrapModel:
    """Resample contiguous blocks while preserving cross-feature dependence."""

    metadata = ModelMetadata(
        name="block-bootstrap",
        version="1",
        family="statistical-baseline",
        capabilities=("single_asset", "multi_asset", "sequence_generation", "cpu_only"),
    )

    def __init__(self, block_length: Optional[int] = None, seed: Optional[int] = None) -> None:
        if block_length is not None and block_length <= 0:
            raise ValueError("block_length must be positive.")
        self.block_length = block_length
        self.seed = seed
        self._rows: Optional[np.ndarray] = None

    def fit(self, windows: np.ndarray, **kwargs: Any) -> "BlockBootstrapModel":
        array = _as_windows(windows)
        rows = array.reshape(-1, array.shape[-1])
        if len(rows) < 2:
            raise ValueError("At least two observations are required to fit the bootstrap.")
        self._rows = rows
        return self

    def generate(self, n_paths: int, length: int, **kwargs: Any) -> np.ndarray:
        if self._rows is None:
            raise RuntimeError("Call fit() before generate().")
        if n_paths <= 0 or length <= 0:
            raise ValueError("n_paths and length must be positive.")
        rng = np.random.default_rng(kwargs.get("seed", self.seed))
        rows = self._rows
        n_rows = len(rows)
        block = self.block_length or max(1, int(round(float(np.sqrt(length)))))
        block = min(block, n_rows)
        paths = np.empty((n_paths, length, rows.shape[-1]), dtype=float)
        for path_index in range(n_paths):
            chunks: list[np.ndarray] = []
            remaining = length
            while remaining > 0:
                start = int(rng.integers(0, n_rows - block + 1))
                take = min(block, remaining)
                chunks.append(rows[start : start + take])
                remaining -= take
            paths[path_index] = np.concatenate(chunks, axis=0)
        return paths

    def save(self, path: str | Path) -> Path:
        if self._rows is None:
            raise RuntimeError("Call fit() before save().")
        target = Path(path)
        if target.suffix != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        block_length = -1 if self.block_length is None else int(self.block_length)
        np.savez_compressed(target, rows=self._rows, block_length=np.asarray(block_length))
        return target

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "BlockBootstrapModel":
        payload = np.load(Path(path), allow_pickle=False)
        block_length = int(payload["block_length"])
        model = cls(block_length=None if block_length < 0 else block_length)
        model._rows = np.asarray(payload["rows"], dtype=float)
        return model


@dataclass(frozen=True)
class GarchParameters:
    mu: float
    omega: float
    alpha: float
    beta: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _garch_negative_log_likelihood(
    residuals: np.ndarray,
    unconditional_variance: float,
    alpha: float,
    beta: float,
) -> float:
    omega = unconditional_variance * (1.0 - alpha - beta)
    if omega <= 0.0:
        return float("inf")
    variance = unconditional_variance
    nll = 0.0
    log_two_pi = float(np.log(2.0 * np.pi))
    for residual in residuals:
        variance = max(variance, 1e-18)
        nll += 0.5 * (log_two_pi + float(np.log(variance)) + (residual * residual) / variance)
        variance = omega + alpha * residual * residual + beta * variance
    return float(nll)


class GaussianGarchModel:
    """Per-feature Gaussian GARCH(1,1) with variance-targeted grid MLE."""

    metadata = ModelMetadata(
        name="garch",
        version="1",
        family="statistical-baseline",
        capabilities=("single_asset", "sequence_generation", "volatility_clustering", "cpu_only"),
    )

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._params: Optional[list[GarchParameters]] = None

    @property
    def parameters(self) -> Optional[list[GarchParameters]]:
        return None if self._params is None else list(self._params)

    def fit(self, windows: np.ndarray, **kwargs: Any) -> "GaussianGarchModel":
        array = _as_windows(windows)
        series = array.reshape(-1, array.shape[-1])
        if len(series) < 30:
            raise ValueError("At least 30 observations are required to fit GARCH(1,1).")
        self._params = [self._fit_series(series[:, index]) for index in range(series.shape[-1])]
        return self

    @staticmethod
    def _fit_series(values: np.ndarray) -> GarchParameters:
        mu = float(np.mean(values))
        residuals = np.asarray(values, dtype=float) - mu
        variance = float(np.var(residuals))
        if variance <= 0.0:
            return GarchParameters(mu=mu, omega=1e-12, alpha=0.0, beta=0.0)

        best_nll = float("inf")
        best_alpha = 0.05
        best_beta = 0.90
        alphas = np.linspace(0.01, 0.30, 12)
        betas = np.linspace(0.50, 0.98, 13)
        for _ in range(2):
            for alpha in alphas:
                for beta in betas:
                    if alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
                        continue
                    nll = _garch_negative_log_likelihood(residuals, variance, float(alpha), float(beta))
                    if nll < best_nll:
                        best_nll = nll
                        best_alpha = float(alpha)
                        best_beta = float(beta)
            alphas = np.linspace(max(1e-4, best_alpha - 0.03), min(0.5, best_alpha + 0.03), 9)
            betas = np.linspace(max(0.0, best_beta - 0.05), min(0.998, best_beta + 0.05), 9)

        omega = variance * (1.0 - best_alpha - best_beta)
        return GarchParameters(mu=mu, omega=max(omega, 1e-12), alpha=best_alpha, beta=best_beta)

    def generate(self, n_paths: int, length: int, **kwargs: Any) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("Call fit() before generate().")
        if n_paths <= 0 or length <= 0:
            raise ValueError("n_paths and length must be positive.")
        rng = np.random.default_rng(kwargs.get("seed", self.seed))
        paths = np.empty((n_paths, length, len(self._params)), dtype=float)
        for feature_index, params in enumerate(self._params):
            persistence = params.alpha + params.beta
            unconditional = params.omega / max(1e-12, 1.0 - persistence)
            variance = np.full(n_paths, max(unconditional, 1e-18), dtype=float)
            shocks = rng.standard_normal((n_paths, length))
            for step in range(length):
                draws = shocks[:, step] * np.sqrt(variance)
                paths[:, step, feature_index] = params.mu + draws
                variance = params.omega + params.alpha * draws**2 + params.beta * variance
        return paths

    def save(self, path: str | Path) -> Path:
        if self._params is None:
            raise RuntimeError("Call fit() before save().")
        target = Path(path)
        if target.suffix != ".json":
            target = target.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seed": self.seed, "features": [params.to_dict() for params in self._params]}
        target.write_text(json.dumps(payload, indent=2))
        return target

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "GaussianGarchModel":
        payload = json.loads(Path(path).read_text())
        model = cls(seed=payload.get("seed"))
        model._params = [GarchParameters(**entry) for entry in payload["features"]]
        return model
