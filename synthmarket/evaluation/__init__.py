"""Evaluation utilities for synthetic-market research."""

from .regimes import classify_volatility_regimes, regime_report, transition_matrix
from .tail_risk import compare_tail_risk, expected_shortfall, maximum_drawdown, tail_risk_report, value_at_risk
from .utility import UtilityComparison, compare_metric

__all__ = [
    "UtilityComparison",
    "classify_volatility_regimes",
    "compare_metric",
    "compare_tail_risk",
    "expected_shortfall",
    "maximum_drawdown",
    "regime_report",
    "tail_risk_report",
    "transition_matrix",
    "value_at_risk",
]
