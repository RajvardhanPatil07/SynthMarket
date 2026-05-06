"""SynthMarket: synthetic financial time-series generation tools."""

from .backtester import BacktestReport, PortfolioSpec, SMACrossoverConfig, run_portfolio_backtest, run_sma_crossover
from .generator import MultiAssetSyntheticMarketGenerator, SyntheticMarketGenerator
from .strategies import StrategySpec, list_strategy_templates, make_strategy_spec
from .trainer import TrainingArtifact, TrainingConfig, TrainingHistory, WGANTrainer

__all__ = [
    "BacktestReport",
    "PortfolioSpec",
    "SMACrossoverConfig",
    "MultiAssetSyntheticMarketGenerator",
    "StrategySpec",
    "SyntheticMarketGenerator",
    "TrainingArtifact",
    "TrainingConfig",
    "TrainingHistory",
    "WGANTrainer",
    "list_strategy_templates",
    "make_strategy_spec",
    "run_portfolio_backtest",
    "run_sma_crossover",
]
