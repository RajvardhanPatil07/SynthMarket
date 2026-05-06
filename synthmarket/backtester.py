"""Simple strategy stress testing across real or synthetic OHLCV paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_utils import clean_ohlcv
from .strategies import StrategySpec, generate_strategy_signal, make_strategy_spec


@dataclass(frozen=True)
class SMACrossoverConfig:
    """Configuration for a long-only SMA crossover backtest."""

    short_window: int = 20
    long_window: int = 50
    initial_cash: float = 10000.0
    fee_bps: float = 1.0
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.short_window <= 0:
            raise ValueError("short_window must be positive.")
        if self.long_window <= self.short_window:
            raise ValueError("long_window must be greater than short_window.")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative.")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "initial_cash": self.initial_cash,
            "fee_bps": self.fee_bps,
            "periods_per_year": self.periods_per_year,
        }


@dataclass
class BacktestReport:
    """SMA backtest outputs for all paths."""

    config: Any
    per_path: pd.DataFrame
    equity_curves: pd.DataFrame
    aggregate: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
            "aggregate": dict(self.aggregate),
            "per_path": self.per_path.reset_index().to_dict(orient="records"),
        }


@dataclass(frozen=True)
class PortfolioSpec:
    """Configuration for long-only strategy stress tests across assets."""

    initial_cash: float = 10000.0
    fee_bps: float = 1.0
    weights: dict[str, float] | None = None
    rebalance_frequency: str = "daily"
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative.")
        if self.rebalance_frequency not in {"none", "daily", "weekly", "monthly"}:
            raise ValueError("rebalance_frequency must be one of: none, daily, weekly, monthly.")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")
        if self.weights is not None:
            if not self.weights:
                raise ValueError("weights must not be empty.")
            if any(float(weight) < 0 for weight in self.weights.values()):
                raise ValueError("weights must be non-negative.")
            if sum(float(weight) for weight in self.weights.values()) <= 0:
                raise ValueError("at least one weight must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "fee_bps": self.fee_bps,
            "weights": None if self.weights is None else dict(self.weights),
            "rebalance_frequency": self.rebalance_frequency,
            "periods_per_year": self.periods_per_year,
        }


def run_sma_crossover(ohlcv: pd.DataFrame, config: SMACrossoverConfig | None = None) -> BacktestReport:
    """Run a long-only SMA crossover strategy on one or many OHLCV paths."""

    config = config or SMACrossoverConfig()
    frame = clean_ohlcv(ohlcv)
    path_groups = _iter_path_groups(frame)

    metrics: list[dict[str, float]] = []
    equity_frames: list[pd.DataFrame] = []
    for path_id, group in path_groups:
        path_metrics, path_equity = _run_single_path(group, config)
        path_metrics["path_id"] = path_id
        metrics.append(path_metrics)
        path_equity = path_equity.copy()
        path_equity["path_id"] = path_id
        equity_frames.append(path_equity)

    per_path = pd.DataFrame(metrics).set_index("path_id").sort_index()
    equity_curves = pd.concat(equity_frames)
    equity_curves = equity_curves.set_index("path_id", append=True)
    equity_curves = equity_curves.reorder_levels([1, 0])
    equity_curves.index = equity_curves.index.set_names(["path_id", "date"])
    return BacktestReport(
        config=config,
        per_path=per_path,
        equity_curves=equity_curves.sort_index(),
        aggregate=_aggregate_metrics(per_path),
    )


def run_strategy_backtest(
    ohlcv: pd.DataFrame,
    strategy: StrategySpec | None = None,
    config: PortfolioSpec | None = None,
) -> BacktestReport:
    """Run a declarative strategy on one asset or synthetic path bundle."""

    strategy = strategy or make_strategy_spec("sma_crossover")
    config = config or PortfolioSpec()
    if _looks_like_multi_asset(ohlcv):
        return run_portfolio_backtest(ohlcv, strategy=strategy, config=config)

    frame = clean_ohlcv(ohlcv)
    metrics: list[dict[str, float]] = []
    equity_frames: list[pd.DataFrame] = []
    for path_id, group in _iter_path_groups(frame):
        path_metrics, path_equity = _run_single_path_strategy(group, strategy, config)
        path_metrics["path_id"] = path_id
        metrics.append(path_metrics)
        path_equity = path_equity.copy()
        path_equity["path_id"] = path_id
        equity_frames.append(path_equity)

    per_path = pd.DataFrame(metrics).set_index("path_id").sort_index()
    equity_curves = pd.concat(equity_frames)
    equity_curves = equity_curves.set_index("path_id", append=True)
    equity_curves = equity_curves.reorder_levels([1, 0])
    equity_curves.index = equity_curves.index.set_names(["path_id", "date"])
    return BacktestReport(
        config=config,
        per_path=per_path,
        equity_curves=equity_curves.sort_index(),
        aggregate=_aggregate_metrics(per_path),
    )


def run_portfolio_backtest(
    ohlcv: pd.DataFrame,
    strategy: StrategySpec | None = None,
    config: PortfolioSpec | None = None,
) -> BacktestReport:
    """Run a long-only strategy over weighted multi-asset OHLCV paths."""

    strategy = strategy or make_strategy_spec("sma_crossover")
    config = config or PortfolioSpec()
    frame = _normalize_portfolio_frame(ohlcv)

    metrics: list[dict[str, float]] = []
    equity_frames: list[pd.DataFrame] = []
    for path_id, path_frame in frame.groupby(level="path_id"):
        path_metrics, path_equity = _run_single_portfolio_path(path_frame.droplevel("path_id"), strategy, config)
        path_metrics["path_id"] = int(path_id)
        metrics.append(path_metrics)
        path_equity = path_equity.copy()
        path_equity["path_id"] = int(path_id)
        equity_frames.append(path_equity)

    per_path = pd.DataFrame(metrics).set_index("path_id").sort_index()
    equity_curves = pd.concat(equity_frames)
    equity_curves = equity_curves.set_index("path_id", append=True)
    equity_curves = equity_curves.reorder_levels([1, 0])
    equity_curves.index = equity_curves.index.set_names(["path_id", "date"])
    return BacktestReport(
        config=config,
        per_path=per_path,
        equity_curves=equity_curves.sort_index(),
        aggregate=_aggregate_metrics(per_path),
    )


def _run_single_path(path: pd.DataFrame, config: SMACrossoverConfig) -> tuple[dict[str, float], pd.DataFrame]:
    path = path.sort_index()
    close = path["Close"].astype(float)
    short_sma = close.rolling(config.short_window, min_periods=config.short_window).mean()
    long_sma = close.rolling(config.long_window, min_periods=config.long_window).mean()
    raw_signal = (short_sma > long_sma).astype(float)
    target_position = raw_signal.shift(1).fillna(0.0)

    returns = close.pct_change().fillna(0.0)
    fee_rate = config.fee_bps / 10000.0
    trades = target_position.diff().abs().fillna(target_position.abs())
    strategy_returns = target_position * returns - trades * fee_rate
    equity = config.initial_cash * (1.0 + strategy_returns).cumprod()

    drawdown = _drawdown(equity)
    trade_count = int(trades.sum())
    round_trips = _round_trip_returns(close, target_position, fee_rate)
    total_return = float(equity.iloc[-1] / config.initial_cash - 1.0)
    annualized_return = _annualized_return(total_return, len(equity), config.periods_per_year)
    annualized_volatility = float(strategy_returns.std(ddof=1) * np.sqrt(config.periods_per_year))
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "trade_count": float(trade_count),
        "win_rate": float(np.mean(np.asarray(round_trips) > 0)) if round_trips else 0.0,
        "final_equity": float(equity.iloc[-1]),
    }
    curve = pd.DataFrame(
        {
            "Close": close,
            "short_sma": short_sma,
            "long_sma": long_sma,
            "position": target_position,
            "strategy_return": strategy_returns,
            "equity": equity,
            "drawdown": drawdown,
        },
        index=path.index,
    )
    return metrics, curve


def _run_single_path_strategy(
    path: pd.DataFrame,
    strategy: StrategySpec,
    config: PortfolioSpec,
) -> tuple[dict[str, float], pd.DataFrame]:
    path = path.sort_index()
    close = path["Close"].astype(float)
    raw_signal = generate_strategy_signal(path, strategy)
    target_position = raw_signal.shift(1).fillna(0.0).clip(0.0, 1.0)

    returns = close.pct_change().fillna(0.0)
    fee_rate = config.fee_bps / 10000.0
    trades = target_position.diff().abs().fillna(target_position.abs())
    strategy_returns = target_position * returns - trades * fee_rate
    equity = config.initial_cash * (1.0 + strategy_returns).cumprod()

    drawdown = _drawdown(equity)
    round_trips = _round_trip_returns(close, target_position, fee_rate)
    total_return = float(equity.iloc[-1] / config.initial_cash - 1.0)
    annualized_return = _annualized_return(total_return, len(equity), config.periods_per_year)
    annualized_volatility = float(strategy_returns.std(ddof=1) * np.sqrt(config.periods_per_year))
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "trade_count": float(trades.sum()),
        "win_rate": float(np.mean(np.asarray(round_trips) > 0)) if round_trips else 0.0,
        "final_equity": float(equity.iloc[-1]),
    }
    curve = pd.DataFrame(
        {
            "Close": close,
            "position": target_position,
            "strategy_return": strategy_returns,
            "equity": equity,
            "drawdown": drawdown,
        },
        index=path.index,
    )
    return metrics, curve


def _run_single_portfolio_path(
    path_frame: pd.DataFrame,
    strategy: StrategySpec,
    config: PortfolioSpec,
) -> tuple[dict[str, float], pd.DataFrame]:
    assets = {
        str(asset): clean_ohlcv(group.droplevel("asset")).sort_index()
        for asset, group in path_frame.groupby(level="asset")
    }
    if not assets:
        raise ValueError("portfolio input contains no assets.")

    common_index: pd.Index | None = None
    for frame in assets.values():
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    if common_index is None or len(common_index) < 2:
        raise ValueError("portfolio assets need at least two aligned observations.")
    common_index = common_index.sort_values()

    weights = _normalize_weights(config.weights, list(assets))
    returns_by_asset: dict[str, pd.Series] = {}
    exposure_by_asset: dict[str, pd.Series] = {}
    for asset, frame in assets.items():
        aligned = frame.loc[common_index]
        returns_by_asset[asset] = aligned["Close"].pct_change().fillna(0.0)
        signal = generate_strategy_signal(aligned, strategy).shift(1).fillna(0.0).clip(0.0, 1.0)
        exposure_by_asset[asset] = signal * weights[asset]

    returns = pd.DataFrame(returns_by_asset, index=common_index)
    target_exposure = pd.DataFrame(exposure_by_asset, index=common_index)
    target_exposure = _apply_rebalance_frequency(target_exposure, config.rebalance_frequency)
    turnover = target_exposure.diff().abs().sum(axis=1).fillna(target_exposure.abs().sum(axis=1))
    portfolio_returns = (target_exposure * returns).sum(axis=1) - turnover * (config.fee_bps / 10000.0)
    equity = config.initial_cash * (1.0 + portfolio_returns).cumprod()
    drawdown = _drawdown(equity)

    total_return = float(equity.iloc[-1] / config.initial_cash - 1.0)
    annualized_return = _annualized_return(total_return, len(equity), config.periods_per_year)
    annualized_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(config.periods_per_year))
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "trade_count": float(np.count_nonzero(turnover.to_numpy(dtype=float) > 1e-12)),
        "win_rate": float(np.mean(portfolio_returns.to_numpy(dtype=float) > 0.0)),
        "final_equity": float(equity.iloc[-1]),
        "asset_count": float(len(assets)),
    }
    curve = pd.DataFrame(
        {
            "portfolio_return": portfolio_returns,
            "turnover": turnover,
            "equity": equity,
            "drawdown": drawdown,
        },
        index=common_index,
    )
    return metrics, curve


def _aggregate_metrics(per_path: pd.DataFrame) -> dict[str, float]:
    returns = per_path["total_return"].to_numpy(dtype=float)
    drawdowns = per_path["max_drawdown"].to_numpy(dtype=float)
    sharpes = per_path["sharpe"].to_numpy(dtype=float)
    percent_profitable = float(np.mean(returns > 0.0))
    downside_penalty = float(np.mean(np.maximum(-returns, 0.0)))
    robustness_score = float(np.clip(percent_profitable * (1.0 + np.nanmedian(sharpes)) - downside_penalty, 0.0, 1.0))
    return {
        "median_return": float(np.nanmedian(returns)),
        "return_p05": float(np.nanquantile(returns, 0.05)),
        "return_p95": float(np.nanquantile(returns, 0.95)),
        "worst_drawdown": float(np.nanmin(drawdowns)),
        "percent_profitable": percent_profitable,
        "median_sharpe": float(np.nanmedian(sharpes)),
        "robustness_score": robustness_score,
    }


def _iter_path_groups(frame: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    if isinstance(frame.index, pd.MultiIndex):
        return [(int(path_id), group.droplevel(0)) for path_id, group in frame.groupby(level=0)]
    return [(0, frame)]


def _looks_like_multi_asset(frame: pd.DataFrame) -> bool:
    return isinstance(frame.index, pd.MultiIndex) and (
        "asset" in [str(name) for name in frame.index.names] or frame.index.nlevels >= 3
    )


def _normalize_portfolio_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce portfolio input to index levels path_id, asset, date."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("ohlcv must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("ohlcv must not be empty.")

    if not isinstance(frame.index, pd.MultiIndex):
        single = clean_ohlcv(frame).copy()
        single.index = pd.Index(single.index, name="date")
        single["path_id"] = 0
        single["asset"] = "asset_0"
        return single.set_index(["path_id", "asset"], append=True).reorder_levels(["path_id", "asset", "date"])

    names = [str(name) for name in frame.index.names]
    if frame.index.nlevels >= 3:
        normalized = clean_ohlcv(frame).copy()
        normalized.index = normalized.index.set_names(["path_id", "asset", "date"] + names[3:])
        if normalized.index.nlevels > 3:
            normalized = normalized.droplevel(list(range(3, normalized.index.nlevels)))
        return normalized.sort_index()

    if "asset" in names:
        normalized = clean_ohlcv(frame).copy()
        normalized.index = normalized.index.set_names(["asset", "date"])
        normalized["path_id"] = 0
        return normalized.set_index("path_id", append=True).reorder_levels(["path_id", "asset", "date"]).sort_index()

    pieces: list[pd.DataFrame] = []
    for path_id, group in frame.groupby(level=0):
        single = clean_ohlcv(group.droplevel(0)).copy()
        single.index = pd.Index(single.index, name="date")
        single["path_id"] = int(path_id)
        single["asset"] = "asset_0"
        pieces.append(
            single.set_index(["path_id", "asset"], append=True).reorder_levels(["path_id", "asset", "date"])
        )
    combined = pd.concat(pieces)
    combined.index = combined.index.set_names(["path_id", "asset", "date"])
    return combined.sort_index()


def _normalize_weights(weights: dict[str, float] | None, assets: list[str]) -> dict[str, float]:
    if weights is None:
        return {asset: 1.0 / len(assets) for asset in assets}
    normalized_keys = {str(asset).upper(): float(weight) for asset, weight in weights.items()}
    selected = {asset: normalized_keys.get(asset.upper(), 0.0) for asset in assets}
    total = sum(selected.values())
    if total <= 0:
        return {asset: 1.0 / len(assets) for asset in assets}
    return {asset: weight / total for asset, weight in selected.items()}


def _apply_rebalance_frequency(exposure: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "daily":
        return exposure
    if frequency == "none":
        mask = pd.Series(False, index=exposure.index)
        mask.iloc[0] = True
    elif frequency == "weekly":
        periods = exposure.index.to_series().dt.to_period("W")
        mask = periods.ne(periods.shift(1))
    elif frequency == "monthly":
        periods = exposure.index.to_series().dt.to_period("M")
        mask = periods.ne(periods.shift(1))
    else:
        raise ValueError("Unsupported rebalance frequency.")
    return exposure.where(mask, np.nan).ffill().fillna(0.0)


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _annualized_return(total_return: float, periods: int, periods_per_year: int) -> float:
    if periods <= 1:
        return 0.0
    growth = max(1.0 + total_return, 1e-12)
    return float(growth ** (periods_per_year / periods) - 1.0)


def _round_trip_returns(close: pd.Series, position: pd.Series, fee_rate: float) -> list[float]:
    entries = (position.diff().fillna(position) > 0).to_numpy()
    exits = (position.diff().fillna(position) < 0).to_numpy()
    prices = close.to_numpy(dtype=float)
    trade_returns: list[float] = []
    entry_price: float | None = None

    for index, price in enumerate(prices):
        if entries[index] and entry_price is None:
            entry_price = price
        elif exits[index] and entry_price is not None:
            trade_returns.append(float(price / entry_price - 1.0 - 2.0 * fee_rate))
            entry_price = None
    if entry_price is not None:
        trade_returns.append(float(prices[-1] / entry_price - 1.0 - fee_rate))
    return trade_returns
