"""Sequential WGAN-GP architecture for synthetic market paths.

The V1 model is a recurrent Wasserstein GAN with gradient penalty. It is
purposefully compact: the trainer will own optimization details, while this file
contains only reusable PyTorch modules and architecture-level utilities.

Why WGAN-GP for V1?
    * The critic outputs an unconstrained Wasserstein score, avoiding sigmoid
      saturation that can destabilize vanilla GANs.
    * Gradient penalty regularizes the critic on interpolated real/fake
      sequences, which is especially useful for continuous financial features.
    * Recurrent generator and critic layers can model volatility regimes and
      temporal dependence without copying raw price paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional, Union

import torch
from torch import Tensor, nn

RNNType = Literal["gru", "lstm"]
ActivationName = Literal["linear", "tanh"]


@dataclass(frozen=True)
class WGANConfig:
    """Configuration shared by the recurrent generator and critic."""

    feature_dim: int = 5
    noise_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.10
    rnn_type: RNNType = "gru"
    generator_output_activation: ActivationName = "linear"
    critic_bidirectional: bool = True

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if self.noise_dim <= 0:
            raise ValueError("noise_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'.")
        if self.generator_output_activation not in {"linear", "tanh"}:
            raise ValueError("generator_output_activation must be 'linear' or 'tanh'.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize model configuration for checkpoints."""

        return asdict(self)

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "WGANConfig":
        """Build a config from checkpoint state."""

        return cls(**state)


@dataclass(frozen=True)
class MultiAssetWGANConfig:
    """Convenience config for correlated multi-asset WGAN feature dimensions."""

    asset_count: int
    features_per_asset: int = 5
    noise_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.10
    rnn_type: RNNType = "gru"
    generator_output_activation: ActivationName = "linear"
    critic_bidirectional: bool = True

    def __post_init__(self) -> None:
        if self.asset_count <= 1:
            raise ValueError("asset_count must be greater than 1 for correlated multi-asset generation.")
        if self.features_per_asset <= 0:
            raise ValueError("features_per_asset must be positive.")

    @property
    def feature_dim(self) -> int:
        return self.asset_count * self.features_per_asset

    def to_wgan_config(self) -> WGANConfig:
        return WGANConfig(
            feature_dim=self.feature_dim,
            noise_dim=self.noise_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rnn_type=self.rnn_type,
            generator_output_activation=self.generator_output_activation,
            critic_bidirectional=self.critic_bidirectional,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "MultiAssetWGANConfig":
        return cls(**state)


class RecurrentGenerator(nn.Module):
    """Noise-to-sequence generator.

    The generator receives an independent noise vector at each time step. This
    preserves stochasticity throughout long generated paths instead of forcing a
    single latent vector to explain the entire trajectory.
    """

    def __init__(
        self,
        feature_dim: int,
        noise_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.10,
        rnn_type: RNNType = "gru",
        output_activation: ActivationName = "linear",
    ) -> None:
        super().__init__()
        _validate_common_dims(feature_dim, hidden_dim, num_layers, dropout)
        if noise_dim <= 0:
            raise ValueError("noise_dim must be positive.")
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'.")
        if output_activation not in {"linear", "tanh"}:
            raise ValueError("output_activation must be 'linear' or 'tanh'.")

        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.output_activation = output_activation

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_projection = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.apply(init_recurrent_weights)

    def forward(self, noise: Tensor) -> Tensor:
        """Generate scaled feature sequences from noise.

        Args:
            noise: Tensor with shape ``(batch, sequence_length, noise_dim)``.

        Returns:
            Tensor with shape ``(batch, sequence_length, feature_dim)``.
        """

        if noise.ndim != 3:
            raise ValueError("noise must have shape (batch, sequence_length, noise_dim).")
        if noise.shape[-1] != self.noise_dim:
            raise ValueError(f"Expected noise_dim={self.noise_dim}, got {noise.shape[-1]}.")

        projected = self.input_projection(noise)
        recurrent, _ = self.rnn(projected)
        generated = self.output_head(recurrent)
        if self.output_activation == "tanh":
            generated = torch.tanh(generated)
        return generated

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Convenience wrapper for drawing fresh generated sequences."""

        noise = sample_noise(batch_size, sequence_length, self.noise_dim, device=device, dtype=dtype)
        return self.forward(noise)


class RecurrentCritic(nn.Module):
    """Sequence-to-Wasserstein-score critic.

    The critic intentionally avoids batch normalization because WGAN-GP computes
    per-sample gradients; batch-dependent normalization makes those gradients
    noisy and can weaken the Lipschitz regularization.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.10,
        rnn_type: RNNType = "gru",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        _validate_common_dims(feature_dim, hidden_dim, num_layers, dropout)
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'.")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )

        pooled_dim = hidden_dim * self.num_directions * 2
        self.score_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_recurrent_weights)

    def forward(self, sequence: Tensor) -> Tensor:
        """Score real or generated sequences.

        Args:
            sequence: Tensor with shape ``(batch, sequence_length, feature_dim)``.

        Returns:
            Tensor with shape ``(batch,)``. Higher scores indicate more real-like
            sequences under the WGAN convention used by the trainer.
        """

        if sequence.ndim != 3:
            raise ValueError("sequence must have shape (batch, sequence_length, feature_dim).")
        if sequence.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {sequence.shape[-1]}.")

        projected = self.input_projection(sequence)
        recurrent, _ = self.rnn(projected)

        # Last hidden state captures path endpoint context; mean pooling keeps
        # information from earlier shocks/regimes visible to the critic.
        last_state = recurrent[:, -1, :]
        mean_state = recurrent.mean(dim=1)
        pooled = torch.cat([last_state, mean_state], dim=-1)
        return self.score_head(pooled).squeeze(-1)


class WGANSequential(nn.Module):
    """Thin module wrapper bundling generator and critic."""

    def __init__(self, config: WGANConfig) -> None:
        super().__init__()
        self.config = config
        self.generator = RecurrentGenerator(
            feature_dim=config.feature_dim,
            noise_dim=config.noise_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            rnn_type=config.rnn_type,
            output_activation=config.generator_output_activation,
        )
        self.critic = RecurrentCritic(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            rnn_type=config.rnn_type,
            bidirectional=config.critic_bidirectional,
        )

    @torch.no_grad()
    def generate(
        self,
        n_paths: int,
        length: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Generate scaled synthetic feature paths in eval mode."""

        parameter = next(self.generator.parameters())
        device = parameter.device if device is None else torch.device(device)
        dtype = parameter.dtype if dtype is None else dtype
        was_training = self.generator.training
        self.generator.eval()
        generated = self.generator.sample(n_paths, length, device=device, dtype=dtype)
        if was_training:
            self.generator.train()
        return generated


def sample_noise(
    batch_size: int,
    sequence_length: int,
    noise_dim: int,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Draw standard-normal sequence noise for the generator."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if noise_dim <= 0:
        raise ValueError("noise_dim must be positive.")
    return torch.randn(batch_size, sequence_length, noise_dim, device=device, dtype=dtype or torch.float32)


def critic_loss_wgan_gp(
    critic: nn.Module,
    real_sequences: Tensor,
    fake_sequences: Tensor,
    gradient_penalty_weight: float = 10.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute critic loss components for WGAN-GP.

    The trainer can call this directly while retaining control of optimizer
    stepping frequency, mixed precision, logging, and gradient clipping.
    """

    if gradient_penalty_weight <= 0:
        raise ValueError("gradient_penalty_weight must be positive.")
    fake_sequences = fake_sequences.detach()

    real_scores = critic(real_sequences)
    fake_scores = critic(fake_sequences)
    wasserstein = fake_scores.mean() - real_scores.mean()
    gp = gradient_penalty(critic, real_sequences, fake_sequences)
    loss = wasserstein + gradient_penalty_weight * gp

    metrics = {
        "critic_loss": loss.detach(),
        "wasserstein": (-wasserstein).detach(),
        "gradient_penalty": gp.detach(),
        "real_score": real_scores.mean().detach(),
        "fake_score": fake_scores.mean().detach(),
    }
    return loss, metrics


def generator_loss_wgan(critic: nn.Module, fake_sequences: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute generator loss for the WGAN objective."""

    fake_scores = critic(fake_sequences)
    loss = -fake_scores.mean()
    metrics = {
        "generator_loss": loss.detach(),
        "fake_score_for_generator": fake_scores.mean().detach(),
    }
    return loss, metrics


def gradient_penalty(critic: nn.Module, real_sequences: Tensor, fake_sequences: Tensor) -> Tensor:
    """Gradient penalty enforcing an approximate 1-Lipschitz critic."""

    if real_sequences.shape != fake_sequences.shape:
        raise ValueError("real_sequences and fake_sequences must have the same shape.")

    batch_size = real_sequences.shape[0]
    alpha_shape = (batch_size,) + (1,) * (real_sequences.ndim - 1)
    alpha = torch.rand(alpha_shape, device=real_sequences.device, dtype=real_sequences.dtype)
    interpolated = alpha * real_sequences + (1.0 - alpha) * fake_sequences
    interpolated.requires_grad_(True)

    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def init_recurrent_weights(module: nn.Module) -> None:
    """Initialize recurrent and linear layers with stable defaults."""

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.GRU, nn.LSTM)):
        for name, parameter in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter.data)
            elif "bias" in name:
                nn.init.zeros_(parameter.data)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count module parameters for experiment logging."""

    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def _validate_common_dims(feature_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
