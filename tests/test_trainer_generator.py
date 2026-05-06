from __future__ import annotations

import numpy as np
import pandas as pd

from synthmarket.data_utils import WindowConfig, prepare_market_data
from synthmarket.generator import SyntheticMarketGenerator
from synthmarket.models.wgan import WGANConfig
from synthmarket.trainer import TrainingConfig, WGANTrainer


def toy_ohlcv(rows: int = 35) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=rows, freq="B")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
    return pd.DataFrame(
        {
            "Open": close * np.exp(rng.normal(0, 0.001, rows)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1000, 2000, rows),
        },
        index=index,
    )


def test_trainer_generator_and_checkpoint_smoke(tmp_path) -> None:
    prepared = prepare_market_data(toy_ohlcv(), WindowConfig(window_size=8, stride=2))
    checkpoint = tmp_path / "model.pt"
    trainer = WGANTrainer(
        WGANConfig(feature_dim=5, noise_dim=4, hidden_dim=8, num_layers=1, dropout=0.0, critic_bidirectional=False),
        TrainingConfig(
            epochs=1,
            batch_size=4,
            n_critic=1,
            device="cpu",
            checkpoint_path=checkpoint,
            grad_clip_norm=5.0,
        ),
    )

    artifact = trainer.fit(prepared)
    synthetic = SyntheticMarketGenerator.from_artifact(artifact, device="cpu").generate_paths(n_paths=3, length=6)
    loaded = SyntheticMarketGenerator.load_checkpoint(checkpoint, device="cpu").generate_paths(n_paths=2, length=5)

    assert checkpoint.exists()
    assert synthetic.index.names == ["path_id", "date"]
    assert synthetic.shape == (18, 5)
    assert loaded.shape == (10, 5)
    assert (synthetic["High"] >= synthetic[["Open", "Close"]].max(axis=1)).all()
    assert (synthetic["Low"] <= synthetic[["Open", "Close"]].min(axis=1)).all()
    assert (synthetic["Volume"] >= 0).all()

