"""SynthMarket: reproducible synthetic financial time-series research tools."""

from .backtester import BacktestReport, PortfolioSpec, SMACrossoverConfig, run_portfolio_backtest, run_sma_crossover
from .data.versioning import DatasetManifest, build_manifest, dataframe_fingerprint
from .evaluation import (
    UtilityComparison,
    classify_volatility_regimes,
    compare_tail_risk,
    expected_shortfall,
    maximum_drawdown,
    regime_report,
    tail_risk_report,
    transition_matrix,
    value_at_risk,
)
from .experiments.manifest import ExperimentManifest, current_git_commit
from .generator import MultiAssetSyntheticMarketGenerator, SyntheticMarketGenerator
from .models import ModelMetadata, SyntheticModel, get_model, list_models, register_model
from .strategies import StrategySpec, list_strategy_templates, make_strategy_spec
from .trainer import TrainingArtifact, TrainingConfig, TrainingHistory, WGANTrainer
from .validation.temporal import TemporalSplit, chronological_split

__all__ = [
    "BacktestReport",
    "DatasetManifest",
    "ExperimentManifest",
    "ModelMetadata",
    "MultiAssetSyntheticMarketGenerator",
    "PortfolioSpec",
    "SMACrossoverConfig",
    "StrategySpec",
    "SyntheticModel",
    "SyntheticMarketGenerator",
    "TemporalSplit",
    "TrainingArtifact",
    "TrainingConfig",
    "TrainingHistory",
    "UtilityComparison",
    "WGANTrainer",
    "build_manifest",
    "chronological_split",
    "classify_volatility_regimes",
    "compare_tail_risk",
    "current_git_commit",
    "dataframe_fingerprint",
    "expected_shortfall",
    "get_model",
    "list_models",
    "list_strategy_templates",
    "make_strategy_spec",
    "maximum_drawdown",
    "regime_report",
    "register_model",
    "run_portfolio_backtest",
    "run_sma_crossover",
    "tail_risk_report",
    "transition_matrix",
    "value_at_risk",
]
