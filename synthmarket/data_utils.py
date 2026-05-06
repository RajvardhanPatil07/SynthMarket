"""Data utilities for preparing financial time-series for generative models.

This module deliberately keeps market-data handling separate from model code.
The neural network should see stationary, scaled sequences, while downstream
users should still be able to reconstruct ordinary OHLCV bars for backtesting.

The default feature representation is return-based:

    open_return   = log(Open_t / Close_{t-1})
    high_return   = log(High_t / Close_{t-1})
    low_return    = log(Low_t / Close_{t-1})
    close_return  = log(Close_t / Close_{t-1})
    volume_return = log1p(Volume_t) - log1p(Volume_{t-1})

This makes the training target closer to stationary than raw prices, which is a
basic guardrail against a generator memorizing absolute historical price levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ScalingMethod = Literal["standard", "robust", "minmax"]

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
FEATURE_COLUMNS: tuple[str, ...] = (
    "open_return",
    "high_return",
    "low_return",
    "close_return",
    "volume_return",
)


@dataclass(frozen=True)
class MarketDataConfig:
    """Configuration for downloading sample historical data via yfinance."""

    ticker: str
    start: Optional[str] = None
    end: Optional[str] = None
    period: Optional[str] = "10y"
    interval: str = "1d"
    auto_adjust: bool = True


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window configuration for sequence model training."""

    window_size: int = 252
    stride: int = 1
    drop_incomplete: bool = True

    def __post_init__(self) -> None:
        if self.window_size <= 1:
            raise ValueError("window_size must be greater than 1.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")


@dataclass
class PreparedData:
    """Container returned by :func:`prepare_market_data`.

    Attributes:
        raw_ohlcv: Cleaned OHLCV input bars.
        features: Return-based, unscaled model features.
        scaled_features: Scaled model features as a DataFrame.
        windows: Numpy array with shape ``(n_windows, window_size, n_features)``.
        scaler: Fitted scaler that can transform/inverse-transform features.
    """

    raw_ohlcv: pd.DataFrame
    features: pd.DataFrame
    scaled_features: pd.DataFrame
    windows: np.ndarray
    scaler: "FinancialFeatureScaler"
    metadata: dict[str, Any] = field(default_factory=dict)


class FinancialFeatureScaler:
    """Small, dependency-light scaler for financial model features.

    ``sklearn`` is intentionally not required by the V1 dependency list. This
    class mirrors the small subset of scaler behavior SynthMarket needs, while
    keeping feature names and inverse transforms explicit.
    """

    def __init__(
        self,
        method: ScalingMethod = "standard",
        clip: Optional[float] = 8.0,
        eps: float = 1e-8,
    ) -> None:
        if method not in {"standard", "robust", "minmax"}:
            raise ValueError("method must be one of: standard, robust, minmax.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.method = method
        self.clip = clip
        self.eps = eps
        self.columns_: Optional[list[str]] = None
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self.columns_ is not None and self.center_ is not None and self.scale_ is not None

    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        columns: Optional[Sequence[str]] = None,
    ) -> "FinancialFeatureScaler":
        frame = _as_feature_frame(data, columns)
        values = frame.to_numpy(dtype=np.float64)

        if self.method == "standard":
            center = np.nanmean(values, axis=0)
            scale = np.nanstd(values, axis=0)
        elif self.method == "robust":
            center = np.nanmedian(values, axis=0)
            q75 = np.nanpercentile(values, 75, axis=0)
            q25 = np.nanpercentile(values, 25, axis=0)
            scale = q75 - q25
        else:
            self.data_min_ = np.nanmin(values, axis=0)
            self.data_max_ = np.nanmax(values, axis=0)
            center = self.data_min_
            scale = self.data_max_ - self.data_min_

        scale = np.where(np.abs(scale) < self.eps, 1.0, scale)
        self.columns_ = list(frame.columns)
        self.center_ = center
        self.scale_ = scale
        return self

    def transform(self, data: Union[pd.DataFrame, np.ndarray], columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        self._require_fitted()
        frame = _as_feature_frame(data, self.columns_ if columns is None else columns)
        values = frame[self.columns_].to_numpy(dtype=np.float64)

        if self.method == "minmax":
            scaled = 2.0 * ((values - self.center_) / (self.scale_ + self.eps)) - 1.0
        else:
            scaled = (values - self.center_) / (self.scale_ + self.eps)

        effective_clip = self._effective_clip()
        if effective_clip is not None:
            scaled = np.clip(scaled, -effective_clip, effective_clip)

        return pd.DataFrame(scaled, index=frame.index, columns=self.columns_)

    def inverse_transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        self._require_fitted()
        frame = _as_feature_frame(data, self.columns_ if columns is None else columns)
        values = frame[self.columns_].to_numpy(dtype=np.float64)

        if self.method == "minmax":
            restored = ((values + 1.0) / 2.0) * (self.scale_ + self.eps) + self.center_
        else:
            restored = values * (self.scale_ + self.eps) + self.center_

        return pd.DataFrame(restored, index=frame.index, columns=self.columns_)

    def fit_transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.fit(data, columns=columns).transform(data, columns=columns)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted scaler state into checkpoint-friendly primitives."""

        return {
            "method": self.method,
            "clip": self.clip,
            "eps": self.eps,
            "columns": None if self.columns_ is None else list(self.columns_),
            "center": None if self.center_ is None else self.center_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "data_min": None if self.data_min_ is None else self.data_min_.tolist(),
            "data_max": None if self.data_max_ is None else self.data_max_.tolist(),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "FinancialFeatureScaler":
        """Rehydrate a scaler saved by :meth:`to_dict`."""

        scaler = cls(
            method=state.get("method", "standard"),
            clip=state.get("clip", 8.0),
            eps=state.get("eps", 1e-8),
        )
        scaler.columns_ = None if state.get("columns") is None else list(state["columns"])
        scaler.center_ = None if state.get("center") is None else np.asarray(state["center"], dtype=np.float64)
        scaler.scale_ = None if state.get("scale") is None else np.asarray(state["scale"], dtype=np.float64)
        scaler.data_min_ = None if state.get("data_min") is None else np.asarray(state["data_min"], dtype=np.float64)
        scaler.data_max_ = None if state.get("data_max") is None else np.asarray(state["data_max"], dtype=np.float64)
        return scaler

    def _effective_clip(self) -> Optional[float]:
        if self.clip is None:
            return None
        if self.method == "minmax":
            return min(float(self.clip), 1.0)
        return float(self.clip)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("FinancialFeatureScaler must be fitted before use.")


class TimeSeriesWindowDataset(Dataset[torch.Tensor]):
    """Torch dataset wrapping sliding windows with shape ``(T, F)``.

    The dataset owns a contiguous float32 copy so DataLoader workers do not pay
    repeated conversion costs during adversarial training.
    """

    def __init__(self, windows: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(windows, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
        else:
            tensor = windows.detach().clone().float().contiguous()
        if tensor.ndim != 3:
            raise ValueError("windows must have shape (n_windows, sequence_length, n_features).")
        self.windows = tensor

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


def fetch_yfinance_ohlcv(config: MarketDataConfig) -> pd.DataFrame:
    """Fetch and clean OHLCV data for one ticker using yfinance.

    ``yfinance`` is imported lazily so the rest of the library remains usable in
    offline settings or production systems that inject their own data.
    """

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        raise ImportError("Install yfinance to fetch sample data: pip install yfinance") from exc

    if config.start or config.end:
        downloaded = yf.download(
            config.ticker,
            start=config.start,
            end=config.end,
            interval=config.interval,
            auto_adjust=config.auto_adjust,
            progress=False,
            group_by="column",
        )
    else:
        downloaded = yf.download(
            config.ticker,
            period=config.period,
            interval=config.interval,
            auto_adjust=config.auto_adjust,
            progress=False,
            group_by="column",
        )

    if downloaded.empty:
        raise ValueError(f"No data returned for ticker {config.ticker!r}.")

    return clean_ohlcv(downloaded)


def parse_tickers(value: Union[str, Sequence[str]]) -> list[str]:
    """Normalize comma-separated or sequence ticker input."""

    if isinstance(value, str):
        raw = value.replace(";", ",").split(",")
    else:
        raw = list(value)
    tickers = []
    for ticker in raw:
        normalized = str(ticker).strip().upper()
        if normalized:
            tickers.append(normalized)
    if not tickers:
        raise ValueError("At least one ticker is required.")
    return list(dict.fromkeys(tickers))


def fetch_yfinance_ohlcv_panel(config: MarketDataConfig) -> pd.DataFrame:
    """Fetch aligned OHLCV panels for one or more tickers.

    The returned frame is indexed by ``asset, date``. Dates missing from any
    asset are dropped so a correlated sequence model receives synchronized bars.
    """

    tickers = parse_tickers(config.ticker)
    pieces: list[pd.DataFrame] = []
    for ticker in tickers:
        asset_frame = fetch_yfinance_ohlcv(
            MarketDataConfig(
                ticker=ticker,
                start=config.start,
                end=config.end,
                period=config.period,
                interval=config.interval,
                auto_adjust=config.auto_adjust,
            )
        )
        asset_frame.index = pd.Index(asset_frame.index, name="date")
        asset_frame["asset"] = ticker
        pieces.append(asset_frame.set_index("asset", append=True).reorder_levels(["asset", "date"]))

    panel = pd.concat(pieces).sort_index()
    return clean_ohlcv_panel(panel)


def clean_ohlcv_panel(data: pd.DataFrame) -> pd.DataFrame:
    """Clean a multi-asset OHLCV frame indexed by ``asset, date``."""

    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels < 2:
        raise ValueError("multi-asset OHLCV data must use a MultiIndex with asset and date levels.")
    frame = data.copy()
    frame.index = frame.index.set_names(["asset", "date"] + list(frame.index.names[2:]))
    if frame.index.nlevels > 2:
        frame = frame.droplevel(list(range(2, frame.index.nlevels)))

    cleaned = []
    common_dates: Optional[pd.Index] = None
    for asset, group in frame.groupby(level="asset"):
        asset_frame = clean_ohlcv(group.droplevel("asset"))
        asset_frame.index = pd.Index(asset_frame.index, name="date")
        common_dates = asset_frame.index if common_dates is None else common_dates.intersection(asset_frame.index)
        asset_frame["asset"] = str(asset)
        cleaned.append(asset_frame.set_index("asset", append=True).reorder_levels(["asset", "date"]))

    if common_dates is None or len(common_dates) == 0:
        raise ValueError("No shared dates remain across assets.")
    aligned = pd.concat(cleaned).sort_index()
    aligned = aligned.loc[(slice(None), common_dates.sort_values()), :]
    return aligned.loc[:, list(OHLCV_COLUMNS)]


def prepare_multi_asset_market_data(
    ohlcv_panel: pd.DataFrame,
    window: Optional[WindowConfig] = None,
    scaler: Optional[FinancialFeatureScaler] = None,
) -> PreparedData:
    """Prepare synchronized multi-asset return features for correlated WGANs."""

    window = window or WindowConfig()
    scaler = scaler or FinancialFeatureScaler(method="standard")
    raw = clean_ohlcv_panel(ohlcv_panel)
    features = ohlcv_panel_to_features(raw)
    scaled = scaler.fit_transform(features)
    windows = make_sliding_windows(
        scaled,
        window_size=window.window_size,
        stride=window.stride,
        drop_incomplete=window.drop_incomplete,
    )
    assets = [str(asset) for asset in raw.index.get_level_values("asset").unique()]
    metadata = {
        "mode": "correlated_multi_asset",
        "assets": assets,
        "feature_columns_by_asset": {
            asset: [f"{asset}__{feature}" for feature in FEATURE_COLUMNS] for asset in assets
        },
        "start_close_by_asset": {
            asset: float(raw.xs(asset, level="asset")["Close"].iloc[-1]) for asset in assets
        },
        "start_volume_by_asset": {
            asset: float(raw.xs(asset, level="asset")["Volume"].iloc[-1]) for asset in assets
        },
    }
    return PreparedData(
        raw_ohlcv=raw,
        features=features,
        scaled_features=scaled,
        windows=windows,
        scaler=scaler,
        metadata=metadata,
    )


def ohlcv_panel_to_features(ohlcv_panel: pd.DataFrame) -> pd.DataFrame:
    """Convert synchronized multi-asset OHLCV bars to a wide feature frame."""

    panel = clean_ohlcv_panel(ohlcv_panel)
    feature_frames: list[pd.DataFrame] = []
    common_index: Optional[pd.Index] = None
    for asset, group in panel.groupby(level="asset"):
        asset_features = ohlcv_to_features(group.droplevel("asset"))
        asset_features.columns = [f"{asset}__{column}" for column in asset_features.columns]
        common_index = asset_features.index if common_index is None else common_index.intersection(asset_features.index)
        feature_frames.append(asset_features)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No shared feature dates remain across assets.")
    aligned = [frame.loc[common_index] for frame in feature_frames]
    return pd.concat(aligned, axis=1).replace([np.inf, -np.inf], np.nan).dropna()


def clean_ohlcv(data: pd.DataFrame, columns: Sequence[str] = OHLCV_COLUMNS) -> pd.DataFrame:
    """Normalize common OHLCV DataFrame shapes into a clean single-index frame."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")

    frame = data.copy()

    # yfinance returns MultiIndex columns for some ticker/query combinations.
    if isinstance(frame.columns, pd.MultiIndex):
        if set(columns).issubset(set(frame.columns.get_level_values(0))):
            frame = frame.droplevel([level for level in range(1, frame.columns.nlevels)], axis=1)
        elif set(columns).issubset(set(frame.columns.get_level_values(-1))):
            frame.columns = frame.columns.get_level_values(-1)
        else:
            # For multi-ticker input, keep the first ticker. Multi-asset support
            # belongs in a portfolio-level dataset wrapper, not this OHLCV helper.
            frame = frame.xs(frame.columns.get_level_values(-1)[0], axis=1, level=-1)

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}.")

    frame = frame.loc[:, list(columns)].sort_index()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    frame = frame.ffill().dropna()

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        raise ValueError("No numeric OHLCV rows remain after cleaning.")

    if (numeric[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive.")
    if (numeric["Volume"] < 0).any():
        raise ValueError("Volume must be non-negative.")

    # Enforce economically sane high/low bounds after vendor adjustments.
    high_floor = numeric[["Open", "Close", "High"]].max(axis=1)
    low_cap = numeric[["Open", "Close", "Low"]].min(axis=1)
    numeric["High"] = np.maximum(numeric["High"], high_floor)
    numeric["Low"] = np.minimum(numeric["Low"], low_cap)

    return numeric


def ohlcv_to_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLCV bars to stationary log-return features."""

    frame = clean_ohlcv(ohlcv)
    previous_close = frame["Close"].shift(1)
    previous_volume = frame["Volume"].shift(1)

    features = pd.DataFrame(index=frame.index)
    features["open_return"] = np.log(frame["Open"] / previous_close)
    features["high_return"] = np.log(frame["High"] / previous_close)
    features["low_return"] = np.log(frame["Low"] / previous_close)
    features["close_return"] = np.log(frame["Close"] / previous_close)
    features["volume_return"] = np.log1p(frame["Volume"]) - np.log1p(previous_volume)

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features.loc[:, list(FEATURE_COLUMNS)]


def features_to_ohlcv(
    features: Union[pd.DataFrame, np.ndarray],
    start_close: float,
    start_volume: float,
    index: Optional[Iterable[pd.Timestamp]] = None,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Reconstruct OHLCV bars from return-based features.

    Generated neural outputs can violate high/low ordering. This function
    repairs those microstructure constraints while preserving the generated
    open and close levels.
    """

    if start_close <= 0:
        raise ValueError("start_close must be positive.")
    if start_volume < 0:
        raise ValueError("start_volume must be non-negative.")

    frame = _as_feature_frame(features, columns)
    if index is not None:
        frame = frame.copy()
        frame.index = pd.Index(index)

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns required for reconstruction: {missing}.")

    previous_close = float(start_close)
    previous_log_volume = float(np.log1p(start_volume))
    rows: list[dict[str, float]] = []

    for _, row in frame.loc[:, FEATURE_COLUMNS].iterrows():
        open_price = previous_close * float(np.exp(row["open_return"]))
        high_price = previous_close * float(np.exp(row["high_return"]))
        low_price = previous_close * float(np.exp(row["low_return"]))
        close_price = previous_close * float(np.exp(row["close_return"]))

        # OHLC bars consumed by backtesting engines should satisfy these bounds.
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        previous_log_volume = previous_log_volume + float(row["volume_return"])
        volume = max(float(np.expm1(previous_log_volume)), 0.0)

        rows.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
            }
        )
        previous_close = close_price

    return pd.DataFrame(rows, index=frame.index, columns=list(OHLCV_COLUMNS))


def make_sliding_windows(
    data: Union[pd.DataFrame, np.ndarray],
    window_size: int,
    stride: int = 1,
    drop_incomplete: bool = True,
) -> np.ndarray:
    """Create contiguous sliding windows for sequence models.

    Returns:
        Array with shape ``(n_windows, window_size, n_features)``.
    """

    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    values = data.to_numpy(dtype=np.float32) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("data must have shape (n_observations, n_features).")
    if len(values) < window_size:
        raise ValueError("Not enough observations to create one full window.")

    starts = range(0, len(values), stride) if not drop_incomplete else range(0, len(values) - window_size + 1, stride)
    windows: list[np.ndarray] = []
    for start in starts:
        end = start + window_size
        window = values[start:end]
        if len(window) == window_size:
            windows.append(window)
        elif not drop_incomplete:
            padding = np.repeat(window[-1:], window_size - len(window), axis=0)
            windows.append(np.concatenate([window, padding], axis=0))

    if not windows:
        raise ValueError("No windows were created. Check window_size and stride.")

    return np.stack(windows).astype(np.float32, copy=False)


def prepare_market_data(
    ohlcv: pd.DataFrame,
    window: Optional[WindowConfig] = None,
    scaler: Optional[FinancialFeatureScaler] = None,
) -> PreparedData:
    """Clean OHLCV data, build return features, scale them, and create windows."""

    window = window or WindowConfig()
    scaler = scaler or FinancialFeatureScaler(method="standard")

    raw = clean_ohlcv(ohlcv)
    features = ohlcv_to_features(raw)
    scaled = scaler.fit_transform(features)
    windows = make_sliding_windows(
        scaled,
        window_size=window.window_size,
        stride=window.stride,
        drop_incomplete=window.drop_incomplete,
    )

    return PreparedData(
        raw_ohlcv=raw,
        features=features,
        scaled_features=scaled,
        windows=windows,
        scaler=scaler,
        metadata={"mode": "single_asset", "assets": ["asset_0"]},
    )


def train_val_split_windows(
    windows: np.ndarray,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological train/validation split for overlapping time-series windows."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, sequence_length, n_features).")

    split = int(np.floor(len(windows) * (1.0 - val_fraction)))
    split = min(max(split, 1), len(windows) - 1)
    return windows[:split], windows[split:]


def _as_feature_frame(data: Union[pd.DataFrame, np.ndarray], columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Coerce array-like feature data into a named DataFrame."""

    if isinstance(data, pd.DataFrame):
        return data.copy()

    values = np.asarray(data)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("feature data must be 1D or 2D.")

    if columns is None:
        columns = [f"feature_{i}" for i in range(values.shape[1])]
    if len(columns) != values.shape[1]:
        raise ValueError("Number of columns does not match feature dimension.")

    return pd.DataFrame(values, columns=list(columns))
