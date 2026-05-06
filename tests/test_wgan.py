from __future__ import annotations

import torch

from synthmarket.models.wgan import WGANConfig, WGANSequential, critic_loss_wgan_gp, generator_loss_wgan


def test_wgan_shapes_and_finite_losses() -> None:
    model = WGANSequential(WGANConfig(feature_dim=5, noise_dim=4, hidden_dim=8, num_layers=1, dropout=0.0))
    real = torch.randn(3, 7, 5)
    fake = model.generate(3, 7, device="cpu")

    critic_loss, critic_metrics = critic_loss_wgan_gp(model.critic, real, fake)
    generator_loss, generator_metrics = generator_loss_wgan(model.critic, fake)

    assert fake.shape == (3, 7, 5)
    assert torch.isfinite(critic_loss)
    assert torch.isfinite(generator_loss)
    assert "gradient_penalty" in critic_metrics
    assert "generator_loss" in generator_metrics


def test_wgan_config_and_state_dict_round_trip(tmp_path) -> None:
    config = WGANConfig(feature_dim=5, noise_dim=4, hidden_dim=8, num_layers=1, dropout=0.0)
    model = WGANSequential(config)
    path = tmp_path / "state.pt"
    torch.save({"config": config.to_dict(), "state": model.state_dict()}, path)

    checkpoint = torch.load(path, map_location="cpu")
    restored = WGANSequential(WGANConfig.from_dict(checkpoint["config"]))
    restored.load_state_dict(checkpoint["state"])

    assert restored.generate(2, 6).shape == (2, 6, 5)

