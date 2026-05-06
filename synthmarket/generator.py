"""User-facing synthetic path generation interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
import torch

from .data_utils import FEATURE_COLUMNS, FinancialFeatureScaler, features_to_ohlcv
from .models.wgan import WGANConfig, WGANSequential
from .trainer import TrainingArtifact, load_checkpoint_dict, resolve_device


class SyntheticMarketGenerator:
    """Generate backtesting-ready synthetic OHLCV paths from a trained artifact."""

    def __init__(
        self,
        model: WGANSequential,
        scaler: FinancialFeatureScaler,
        feature_columns: Sequence[str],
        start_close: float,
        start_volume: float,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.scaler = scaler
        self.feature_columns = list(feature_columns)
        self.start_close = float(start_close)
        self.start_volume = float(start_volume)

    @classmethod
    def from_artifact(cls, artifact: TrainingArtifact, device: str = "auto") -> "SyntheticMarketGenerator":
        if artifact.metadata.get("mode") == "correlated_multi_asset" or isinstance(artifact.start_close, dict):
            return MultiAssetSyntheticMarketGenerator.from_artifact(artifact, device=device)
        return cls(
            model=artifact.model,
            scaler=artifact.scaler,
            feature_columns=artifact.feature_columns,
            start_close=artifact.start_close,
            start_volume=artifact.start_volume,
            device=device,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: str = "auto",
        map_location: Union[str, torch.device] = "cpu",
    ) -> "SyntheticMarketGenerator":
        checkpoint = load_checkpoint_dict(path, map_location=map_location)
        if checkpoint.get("metadata", {}).get("mode") == "correlated_multi_asset" or isinstance(
            checkpoint.get("start_close"), dict
        ):
            return MultiAssetSyntheticMarketGenerator.load_checkpoint(path, device=device, map_location=map_location)
        model_config = WGANConfig.from_dict(checkpoint["model_config"])
        model = WGANSequential(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = FinancialFeatureScaler.from_dict(checkpoint["scaler"])
        return cls(
            model=model,
            scaler=scaler,
            feature_columns=checkpoint.get("feature_columns", list(FEATURE_COLUMNS)),
            start_close=float(checkpoint["start_close"]),
            start_volume=float(checkpoint["start_volume"]),
            device=device,
        )

    def generate_paths(
        self,
        n_paths: int = 1000,
        length: int = 252,
        start_close: Optional[float] = None,
        start_volume: Optional[float] = None,
        start_date: Union[str, pd.Timestamp] = "2000-01-03",
        freq: str = "B",
        date_index: Optional[Sequence[pd.Timestamp]] = None,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV paths as a MultiIndex DataFrame."""

        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if length <= 0:
            raise ValueError("length must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        index = _build_date_index(length, start_date=start_date, freq=freq, date_index=date_index)
        seed_close = self.start_close if start_close is None else float(start_close)
        seed_volume = self.start_volume if start_volume is None else float(start_volume)

        frames: list[pd.DataFrame] = []
        generated = 0
        with torch.no_grad():
            while generated < n_paths:
                current_batch = min(batch_size, n_paths - generated)
                scaled = self.model.generate(current_batch, length, device=self.device).detach().cpu().numpy()
                for batch_index in range(current_batch):
                    features = self.scaler.inverse_transform(scaled[batch_index], columns=self.feature_columns)
                    ohlcv = features_to_ohlcv(
                        features,
                        start_close=seed_close,
                        start_volume=seed_volume,
                        index=index,
                    )
                    ohlcv.insert(0, "path_id", generated + batch_index)
                    frames.append(ohlcv)
                generated += current_batch

        combined = pd.concat(frames)
        combined = combined.set_index("path_id", append=True).reorder_levels(["path_id", combined.index.name])
        combined.index = combined.index.set_names(["path_id", "date"])
        return combined.loc[:, ["Open", "High", "Low", "Close", "Volume"]].sort_index()


class MultiAssetSyntheticMarketGenerator:
    """Generate correlated synthetic OHLCV panels from a joint WGAN artifact."""

    def __init__(
        self,
        model: WGANSequential,
        scaler: FinancialFeatureScaler,
        feature_columns: Sequence[str],
        assets: Sequence[str],
        start_close_by_asset: dict[str, float],
        start_volume_by_asset: dict[str, float],
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.scaler = scaler
        self.feature_columns = list(feature_columns)
        self.assets = [str(asset) for asset in assets]
        self.start_close_by_asset = {str(asset): float(value) for asset, value in start_close_by_asset.items()}
        self.start_volume_by_asset = {str(asset): float(value) for asset, value in start_volume_by_asset.items()}

    @classmethod
    def from_artifact(cls, artifact: TrainingArtifact, device: str = "auto") -> "MultiAssetSyntheticMarketGenerator":
        metadata = dict(artifact.metadata or {})
        assets = metadata.get("assets") or list((artifact.start_close or {}).keys())
        return cls(
            model=artifact.model,
            scaler=artifact.scaler,
            feature_columns=artifact.feature_columns,
            assets=assets,
            start_close_by_asset=dict(artifact.start_close),
            start_volume_by_asset=dict(artifact.start_volume),
            device=device,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: str = "auto",
        map_location: Union[str, torch.device] = "cpu",
    ) -> "MultiAssetSyntheticMarketGenerator":
        checkpoint = load_checkpoint_dict(path, map_location=map_location)
        model_config = WGANConfig.from_dict(checkpoint["model_config"])
        model = WGANSequential(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = FinancialFeatureScaler.from_dict(checkpoint["scaler"])
        metadata = dict(checkpoint.get("metadata") or {})
        start_close = dict(checkpoint["start_close"])
        start_volume = dict(checkpoint["start_volume"])
        return cls(
            model=model,
            scaler=scaler,
            feature_columns=checkpoint.get("feature_columns", list(FEATURE_COLUMNS)),
            assets=metadata.get("assets") or list(start_close),
            start_close_by_asset=start_close,
            start_volume_by_asset=start_volume,
            device=device,
        )

    def generate_paths(
        self,
        n_paths: int = 1000,
        length: int = 252,
        start_close_by_asset: Optional[dict[str, float]] = None,
        start_volume_by_asset: Optional[dict[str, float]] = None,
        start_date: Union[str, pd.Timestamp] = "2000-01-03",
        freq: str = "B",
        date_index: Optional[Sequence[pd.Timestamp]] = None,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if length <= 0:
            raise ValueError("length must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        index = _build_date_index(length, start_date=start_date, freq=freq, date_index=date_index)
        close_seed = {**self.start_close_by_asset, **(start_close_by_asset or {})}
        volume_seed = {**self.start_volume_by_asset, **(start_volume_by_asset or {})}

        frames: list[pd.DataFrame] = []
        generated = 0
        with torch.no_grad():
            while generated < n_paths:
                current_batch = min(batch_size, n_paths - generated)
                scaled = self.model.generate(current_batch, length, device=self.device).detach().cpu().numpy()
                for batch_index in range(current_batch):
                    wide_features = self.scaler.inverse_transform(scaled[batch_index], columns=self.feature_columns)
                    for asset in self.assets:
                        asset_features = _extract_asset_features(wide_features, asset)
                        ohlcv = features_to_ohlcv(
                            asset_features,
                            start_close=float(close_seed[asset]),
                            start_volume=float(volume_seed[asset]),
                            index=index,
                        )
                        ohlcv.insert(0, "path_id", generated + batch_index)
                        ohlcv.insert(1, "asset", asset)
                        frames.append(ohlcv)
                generated += current_batch

        combined = pd.concat(frames)
        combined = combined.set_index(["path_id", "asset"], append=True)
        combined = combined.reorder_levels(["path_id", "asset", combined.index.names[0]])
        combined.index = combined.index.set_names(["path_id", "asset", "date"])
        return combined.loc[:, ["Open", "High", "Low", "Close", "Volume"]].sort_index()


def _extract_asset_features(features: pd.DataFrame, asset: str) -> pd.DataFrame:
    columns = [f"{asset}__{feature}" for feature in FEATURE_COLUMNS]
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ValueError(f"Generated feature frame is missing columns for {asset}: {missing}.")
    asset_features = features.loc[:, columns].copy()
    asset_features.columns = list(FEATURE_COLUMNS)
    return asset_features


def load_checkpoint(path: Union[str, Path], device: str = "auto") -> SyntheticMarketGenerator:
    """Convenience wrapper for ``SyntheticMarketGenerator.load_checkpoint``."""

    return SyntheticMarketGenerator.load_checkpoint(path, device=device)


def _build_date_index(
    length: int,
    start_date: Union[str, pd.Timestamp],
    freq: str,
    date_index: Optional[Sequence[pd.Timestamp]],
) -> pd.Index:
    if date_index is not None:
        index = pd.Index(date_index)
        if len(index) != length:
            raise ValueError("date_index length must match generated path length.")
        return index
    return pd.date_range(start=pd.Timestamp(start_date), periods=length, freq=freq, name="date")
