from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthmarket.data_utils import (
    FEATURE_COLUMNS,
    FinancialFeatureScaler,
    WindowConfig,
    clean_ohlcv,
    features_to_ohlcv,
    make_sliding_windows,
    ohlcv_to_features,
    prepare_market_data,
    train_val_split_windows,
)


def toy_ohlcv(rows: int = 40) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = 100 * np.exp(np.cumsum(np.linspace(-0.002, 0.003, rows)))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.linspace(1000, 2000, rows),
        },
        index=index,
    )


def test_clean_ohlcv_accepts_yfinance_multiindex() -> None:
    base = toy_ohlcv()
    multi = base.copy()
    multi.columns = pd.MultiIndex.from_product([multi.columns, ["SPY"]])

    cleaned = clean_ohlcv(multi)

    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(cleaned) == len(base)


def test_feature_round_trip_reconstructs_original_ohlcv() -> None:
    base = toy_ohlcv()
    features = ohlcv_to_features(base)
    restored = features_to_ohlcv(
        features,
        start_close=float(base["Close"].iloc[0]),
        start_volume=float(base["Volume"].iloc[0]),
        index=features.index,
    )

    expected = base.loc[features.index]
    np.testing.assert_allclose(restored[["Open", "High", "Low", "Close"]], expected[["Open", "High", "Low", "Close"]])
    np.testing.assert_allclose(restored["Volume"], expected["Volume"])


def test_scaler_serializes_and_minmax_clips_to_unit_interval() -> None:
    features = ohlcv_to_features(toy_ohlcv())
    scaler = FinancialFeatureScaler(method="minmax").fit(features)
    transformed = scaler.transform(features * 100)
    restored_scaler = FinancialFeatureScaler.from_dict(scaler.to_dict())
    round_trip = restored_scaler.inverse_transform(restored_scaler.transform(features))

    assert transformed.max().max() <= 1.0
    assert transformed.min().min() >= -1.0
    np.testing.assert_allclose(round_trip.to_numpy(), features.to_numpy(), atol=1e-8)


def test_sliding_windows_and_train_val_split() -> None:
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    windows = make_sliding_windows(values, window_size=4, stride=2)
    train, val = train_val_split_windows(windows, val_fraction=0.25)

    assert windows.shape == (4, 4, 2)
    assert len(train) == 3
    assert len(val) == 1


def test_prepare_market_data_and_invalid_ohlcv() -> None:
    prepared = prepare_market_data(toy_ohlcv(), WindowConfig(window_size=8, stride=4))

    assert prepared.windows.shape[-1] == len(FEATURE_COLUMNS)
    bad = toy_ohlcv()
    bad.loc[bad.index[0], "Close"] = -1
    with pytest.raises(ValueError, match="strictly positive"):
        clean_ohlcv(bad)

