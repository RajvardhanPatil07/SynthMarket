from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthmarket.backtester import SMACrossoverConfig, run_sma_crossover


def make_trending_ohlcv(rows: int = 80, start: float = 100.0, stop: float = 140.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = np.linspace(start, stop, rows)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(rows, 1000),
        },
        index=dates,
    )


def test_sma_backtest_generates_metrics_and_equity_curve() -> None:
    report = run_sma_crossover(make_trending_ohlcv(), SMACrossoverConfig(short_window=3, long_window=8, fee_bps=0))

    row = report.per_path.loc[0]
    assert row["final_equity"] > 10000
    assert row["trade_count"] >= 1
    assert row["max_drawdown"] <= 0
    assert report.equity_curves.index.names == ["path_id", "date"]
    assert "robustness_score" in report.aggregate


def test_sma_backtest_handles_multi_path_input_and_fees() -> None:
    up = make_trending_ohlcv(start=100, stop=140)
    down = make_trending_ohlcv(start=140, stop=100)
    up["path_id"] = 0
    down["path_id"] = 1
    combined = pd.concat([up, down])
    combined = combined.set_index("path_id", append=True).reorder_levels(["path_id", None])
    combined.index = combined.index.set_names(["path_id", "date"])

    report = run_sma_crossover(combined, SMACrossoverConfig(short_window=3, long_window=8, fee_bps=5))

    assert len(report.per_path) == 2
    assert report.aggregate["return_p05"] <= report.aggregate["return_p95"]
    assert 0 <= report.aggregate["percent_profitable"] <= 1


def test_sma_config_validation() -> None:
    with pytest.raises(ValueError, match="long_window"):
        SMACrossoverConfig(short_window=10, long_window=5)

