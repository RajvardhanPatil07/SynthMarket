"""Training utilities for SynthMarket WGAN-GP models."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data_utils import FEATURE_COLUMNS, FinancialFeatureScaler, PreparedData, TimeSeriesWindowDataset
from .models.wgan import WGANConfig, WGANSequential, critic_loss_wgan_gp, generator_loss_wgan


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for recurrent WGAN-GP training."""

    epochs: int = 100
    batch_size: int = 64
    n_critic: int = 5
    gradient_penalty_weight: float = 10.0
    lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    drop_last: bool = True
    checkpoint_path: Optional[Union[str, Path]] = None
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.n_critic <= 0:
            raise ValueError("n_critic must be positive.")
        if self.gradient_penalty_weight <= 0:
            raise ValueError("gradient_penalty_weight must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")


@dataclass
class TrainingHistory:
    """Epoch-level WGAN training metrics."""

    epochs: list[int] = field(default_factory=list)
    critic_loss: list[float] = field(default_factory=list)
    generator_loss: list[float] = field(default_factory=list)
    wasserstein: list[float] = field(default_factory=list)
    gradient_penalty: list[float] = field(default_factory=list)
    real_score: list[float] = field(default_factory=list)
    fake_score: list[float] = field(default_factory=list)

    def append(self, epoch: int, metrics: dict[str, float]) -> None:
        self.epochs.append(epoch)
        self.critic_loss.append(float(metrics.get("critic_loss", np.nan)))
        self.generator_loss.append(float(metrics.get("generator_loss", np.nan)))
        self.wasserstein.append(float(metrics.get("wasserstein", np.nan)))
        self.gradient_penalty.append(float(metrics.get("gradient_penalty", np.nan)))
        self.real_score.append(float(metrics.get("real_score", np.nan)))
        self.fake_score.append(float(metrics.get("fake_score", np.nan)))

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "epochs": list(self.epochs),
            "critic_loss": list(self.critic_loss),
            "generator_loss": list(self.generator_loss),
            "wasserstein": list(self.wasserstein),
            "gradient_penalty": list(self.gradient_penalty),
            "real_score": list(self.real_score),
            "fake_score": list(self.fake_score),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Sequence[float]]) -> "TrainingHistory":
        history = cls()
        history.epochs = [int(value) for value in state.get("epochs", [])]
        history.critic_loss = [float(value) for value in state.get("critic_loss", [])]
        history.generator_loss = [float(value) for value in state.get("generator_loss", [])]
        history.wasserstein = [float(value) for value in state.get("wasserstein", [])]
        history.gradient_penalty = [float(value) for value in state.get("gradient_penalty", [])]
        history.real_score = [float(value) for value in state.get("real_score", [])]
        history.fake_score = [float(value) for value in state.get("fake_score", [])]
        return history


@dataclass
class TrainingArtifact:
    """Everything needed to generate paths after training."""

    model: WGANSequential
    model_config: WGANConfig
    training_config: TrainingConfig
    scaler: FinancialFeatureScaler
    feature_columns: list[str]
    start_close: Any
    start_volume: Any
    history: TrainingHistory
    checkpoint_path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config.to_dict(),
            "training_config": _serialize_training_config(self.training_config),
            "scaler": self.scaler.to_dict(),
            "feature_columns": list(self.feature_columns),
            "start_close": self.start_close,
            "start_volume": self.start_volume,
            "history": self.history.to_dict(),
            "metadata": dict(self.metadata),
        }

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_checkpoint_dict(), checkpoint_path)
        self.checkpoint_path = checkpoint_path
        return checkpoint_path


class WGANTrainer:
    """Adversarial trainer for recurrent WGAN-GP market generators."""

    def __init__(
        self,
        model_config: Optional[WGANConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model_config = model_config or WGANConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = resolve_device(self.training_config.device)
        self.model = WGANSequential(self.model_config).to(self.device)
        self.history = TrainingHistory()

    def fit(self, prepared: PreparedData) -> TrainingArtifact:
        """Train on prepared sliding windows and return a generation artifact."""

        set_seed(self.training_config.seed)
        windows = prepared.windows
        if windows.ndim != 3:
            raise ValueError("prepared.windows must have shape (n_windows, sequence_length, n_features).")
        if windows.shape[-1] != self.model_config.feature_dim:
            raise ValueError(
                "Model feature_dim="
                f"{self.model_config.feature_dim} does not match data feature_dim={windows.shape[-1]}."
            )

        dataset = TimeSeriesWindowDataset(windows)
        batch_size = min(self.training_config.batch_size, len(dataset))
        drop_last = self.training_config.drop_last and len(dataset) >= self.training_config.batch_size
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=self.training_config.num_workers,
        )
        if len(loader) == 0:
            raise ValueError("Training DataLoader produced no batches.")

        generator_optimizer = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=self.training_config.lr,
            betas=self.training_config.betas,
        )
        critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=self.training_config.lr,
            betas=self.training_config.betas,
        )

        global_step = 0
        for epoch in range(1, self.training_config.epochs + 1):
            epoch_metrics: dict[str, list[float]] = {
                "critic_loss": [],
                "generator_loss": [],
                "wasserstein": [],
                "gradient_penalty": [],
                "real_score": [],
                "fake_score": [],
            }

            for real_sequences in loader:
                global_step += 1
                real_sequences = real_sequences.to(self.device)
                batch, sequence_length, _ = real_sequences.shape

                critic_optimizer.zero_grad(set_to_none=True)
                fake_sequences = self.model.generator.sample(
                    batch_size=batch,
                    sequence_length=sequence_length,
                    device=self.device,
                    dtype=real_sequences.dtype,
                )
                critic_loss, critic_metrics = critic_loss_wgan_gp(
                    self.model.critic,
                    real_sequences,
                    fake_sequences,
                    gradient_penalty_weight=self.training_config.gradient_penalty_weight,
                )
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.training_config.grad_clip_norm)
                critic_optimizer.step()

                _extend_metrics(epoch_metrics, critic_metrics)

                if global_step % self.training_config.n_critic == 0:
                    generator_optimizer.zero_grad(set_to_none=True)
                    fake_for_generator = self.model.generator.sample(
                        batch_size=batch,
                        sequence_length=sequence_length,
                        device=self.device,
                        dtype=real_sequences.dtype,
                    )
                    generator_loss, generator_metrics = generator_loss_wgan(self.model.critic, fake_for_generator)
                    generator_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.generator.parameters(), self.training_config.grad_clip_norm)
                    generator_optimizer.step()
                    _extend_metrics(epoch_metrics, generator_metrics)

            averaged = {key: float(np.nanmean(values)) if values else np.nan for key, values in epoch_metrics.items()}
            self.history.append(epoch, averaged)

        feature_columns = list(prepared.scaled_features.columns)
        if not feature_columns:
            feature_columns = list(FEATURE_COLUMNS)
        metadata = dict(getattr(prepared, "metadata", {}) or {})
        start_close: Any = metadata.get("start_close_by_asset")
        start_volume: Any = metadata.get("start_volume_by_asset")
        if start_close is None:
            start_close = float(prepared.raw_ohlcv["Close"].iloc[-1])
        if start_volume is None:
            start_volume = float(prepared.raw_ohlcv["Volume"].iloc[-1])
        artifact = TrainingArtifact(
            model=self.model,
            model_config=self.model_config,
            training_config=self.training_config,
            scaler=prepared.scaler,
            feature_columns=feature_columns,
            start_close=start_close,
            start_volume=start_volume,
            history=self.history,
            metadata=metadata,
        )
        if self.training_config.checkpoint_path is not None:
            artifact.save_checkpoint(self.training_config.checkpoint_path)
        return artifact


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` to the best available torch device."""

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_dict(path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> dict[str, Any]:
    """Load a SynthMarket checkpoint across PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extend_metrics(target: dict[str, list[float]], metrics: dict[str, torch.Tensor]) -> None:
    for key, value in metrics.items():
        if key in target:
            target[key].append(float(value.detach().cpu().item()))


def _serialize_training_config(config: TrainingConfig) -> dict[str, Any]:
    state = asdict(config)
    if state["checkpoint_path"] is not None:
        state["checkpoint_path"] = str(state["checkpoint_path"])
    return state
