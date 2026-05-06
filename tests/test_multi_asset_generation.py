from __future__ import annotations

import numpy as np
import pandas as pd

from synthmarket.data_utils import WindowConfig, prepare_multi_asset_market_data
from synthmarket.generator import MultiAssetSyntheticMarketGenerator
from synthmarket.models.wgan import MultiAssetWGANConfig
from synthmarket.multi_asset import evaluate_cross_asset_realism
from synthmarket.trainer import TrainingConfig, WGANTrainer


def make_panel(rows: int = 36) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=rows, freq="B")
    frames = []
    for asset, base, slope in [("SPY", 100.0, 0.5), ("QQQ", 80.0, 0.65)]:
        close = base + np.arange(rows) * slope + np.sin(np.arange(rows) / 3.0)
        frame = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(rows, 1000),
            },
            index=dates,
        )
        frame["asset"] = asset
        frames.append(frame.set_index("asset", append=True).reorder_levels(["asset", None]))
    return pd.concat(frames).sort_index()


def test_prepare_multi_asset_market_data_shapes() -> None:
    prepared = prepare_multi_asset_market_data(make_panel(), WindowConfig(window_size=8))

    assert prepared.windows.shape[-1] == 10
    assert prepared.metadata["assets"] == ["QQQ", "SPY"] or prepared.metadata["assets"] == ["SPY", "QQQ"]
    assert set(prepared.metadata["start_close_by_asset"]) == {"SPY", "QQQ"}


def test_multi_asset_generator_outputs_path_asset_date_index() -> None:
    prepared = prepare_multi_asset_market_data(make_panel(), WindowConfig(window_size=8))
    model_config = MultiAssetWGANConfig(asset_count=2, hidden_dim=8, noise_dim=4, num_layers=1).to_wgan_config()
    trainer = WGANTrainer(
        model_config,
        TrainingConfig(epochs=1, batch_size=4, n_critic=1, device="cpu", drop_last=False),
    )
    artifact = trainer.fit(prepared)
    generator = MultiAssetSyntheticMarketGenerator.from_artifact(artifact, device="cpu")
    synthetic = generator.generate_paths(n_paths=2, length=12, batch_size=2)

    assert synthetic.index.names == ["path_id", "asset", "date"]
    assert set(synthetic.index.get_level_values("asset")) == {"SPY", "QQQ"}
    assert (synthetic[["Open", "High", "Low", "Close"]] > 0).all().all()


def test_cross_asset_realism_payload() -> None:
    panel = make_panel()
    prepared = prepare_multi_asset_market_data(panel, WindowConfig(window_size=8))
    model_config = MultiAssetWGANConfig(asset_count=2, hidden_dim=8, noise_dim=4, num_layers=1).to_wgan_config()
    artifact = WGANTrainer(
        model_config,
        TrainingConfig(epochs=1, batch_size=4, n_critic=1, device="cpu", drop_last=False),
    ).fit(prepared)
    synthetic = MultiAssetSyntheticMarketGenerator.from_artifact(artifact, device="cpu").generate_paths(
        n_paths=1, length=12, batch_size=1
    )

    metrics = evaluate_cross_asset_realism(panel, synthetic)
    assert "correlation_distance" in metrics
    assert metrics["assets"]
