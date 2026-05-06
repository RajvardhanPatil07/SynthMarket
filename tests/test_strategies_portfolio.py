from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthmarket.backtester import PortfolioSpec, run_portfolio_backtest, run_strategy_backtest
from synthmarket.strategies import StrategySpec, generate_strategy_signal, list_strategy_templates, make_strategy_spec


def make_ohlcv(rows: int = 80, start: float = 100.0, stop: float = 130.0) -> pd.DataFrame:
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


def make_panel() -> pd.DataFrame:
    spy = make_ohlcv(start=100, stop=130)
    qqq = make_ohlcv(start=80, stop=120)
    spy["asset"] = "SPY"
    qqq["asset"] = "QQQ"
    panel = pd.concat([spy, qqq])
    return panel.set_index("asset", append=True).reorder_levels(["asset", None])


def test_strategy_templates_and_validation() -> None:
    templates = list_strategy_templates()
    assert {template["id"] for template in templates} >= {"buy_hold", "sma_crossover", "donchian_breakout"}

    with pytest.raises(ValueError, match="long_window"):
        StrategySpec(name="Bad", template="sma_crossover", parameters={"short_window": 20, "long_window": 10})


def test_strategy_signal_and_single_asset_backtest() -> None:
    strategy = make_strategy_spec("ema_crossover", {"short_window": 3, "long_window": 8})
    signal = generate_strategy_signal(make_ohlcv(), strategy)
    report = run_strategy_backtest(make_ohlcv(), strategy=strategy, config=PortfolioSpec(fee_bps=0))

    assert signal.max() == 1.0
    assert report.per_path.loc[0, "final_equity"] > 10000
    assert "robustness_score" in report.aggregate


def test_portfolio_backtest_uses_multi_asset_weights() -> None:
    strategy = make_strategy_spec("buy_hold")
    report = run_portfolio_backtest(
        make_panel(),
        strategy=strategy,
        config=PortfolioSpec(weights={"SPY": 0.6, "QQQ": 0.4}, rebalance_frequency="weekly", fee_bps=0),
    )

    assert report.per_path.loc[0, "asset_count"] == 2
    assert report.per_path.loc[0, "final_equity"] > 10000
    assert report.equity_curves.index.names == ["path_id", "date"]
