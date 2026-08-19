import numpy as np
import pandas as pd
import pytest

from synthmarket.validation import chronological_split, validate_ohlcv


def _frame(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    index = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, rows)))
    open_price = close * (1 + rng.normal(0, 0.002, rows))
    high = np.maximum.reduce([open_price, close, close * (1 + np.abs(rng.normal(0, 0.006, rows)))])
    low = np.minimum.reduce([open_price, close, close * (1 - np.abs(rng.normal(0, 0.006, rows)))])
    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": 1_000_000.0},
        index=index,
    )


def test_split_is_disjoint_ordered_and_shuffle_safe() -> None:
    frame = _frame()
    split = chronological_split(frame.sample(frac=1.0, random_state=3), min_train_rows=50)
    assert len(split.train) + len(split.validation) + len(split.test) == len(frame)
    assert split.train.index.max() < split.validation.index.min() < split.test.index.min()
    assert set(split.train.index).isdisjoint(split.validation.index)
    assert set(split.validation.index).isdisjoint(split.test.index)


def test_valid_ohlcv_passes() -> None:
    report = validate_ohlcv(_frame())
    assert report.is_valid
    assert report.rows == 400


def test_ohlcv_reports_injected_violations() -> None:
    frame = _frame()
    frame.iloc[3, frame.columns.get_loc("High")] = frame.iloc[3]["Low"] * 0.5
    frame.iloc[7, frame.columns.get_loc("Volume")] = -10.0
    frame.iloc[11, frame.columns.get_loc("Close")] = np.nan
    report = validate_ohlcv(frame)
    assert not report.is_valid
    assert report.violations["high_below_low"] >= 1
    assert report.violations["negative_volume"] == 1
    assert report.violations["non_finite"] == 1
    with pytest.raises(ValueError):
        validate_ohlcv(frame, raise_on_error=True)


def test_ohlcv_requires_all_columns() -> None:
    with pytest.raises(ValueError, match="Missing required"):
        validate_ohlcv(pd.DataFrame({"Close": [1.0]}))
