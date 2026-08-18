"""Financial tail-risk diagnostics for real and synthetic returns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def returns_from_close(close: pd.Series) -> np.ndarray:
    values = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    returns = np.log(values / values.shift(1)).dropna()
    return returns.to_numpy(dtype=float)


def value_at_risk(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Historical VaR reported as a positive loss magnitude."""

    _validate_confidence(confidence)
    return float(-np.quantile(np.asarray(returns, dtype=float), 1.0 - confidence))


def expected_shortfall(returns: np.ndarray, confidence: float = 0.99) -> float:
    """Historical expected shortfall reported as a positive loss magnitude."""

    _validate_confidence(confidence)
    values = np.asarray(returns, dtype=float)
    threshold = np.quantile(values, 1.0 - confidence)
    tail = values[values <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-np.mean(tail))


def maximum_drawdown(prices: pd.Series) -> float:
    """Return maximum drawdown as a positive fraction."""

    values = pd.to_numeric(prices, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(values)
    drawdowns = 1.0 - (values / np.maximum(running_max, 1e-12))
    return float(np.max(drawdowns))


def tail_risk_report(returns: np.ndarray, confidence_levels: tuple[float, ...] = (0.95, 0.99)) -> dict[str, Any]:
    """Return a compact, serializable tail-risk report."""

    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        raise ValueError("At least 10 finite returns are required for tail-risk analysis.")
    return {
        "count": int(len(values)),
        "levels": {
            str(level): {
                "var": value_at_risk(values, level),
                "expected_shortfall": expected_shortfall(values, level),
            }
            for level in confidence_levels
        },
        "worst_return": float(-np.min(values)),
        "extreme_loss_rate_3pct": float(np.mean(values <= -0.03)),
    }


def compare_tail_risk(real_returns: np.ndarray, synthetic_returns: np.ndarray) -> dict[str, Any]:
    """Compare key tail statistics between real and synthetic returns."""

    real = tail_risk_report(real_returns)
    synthetic = tail_risk_report(synthetic_returns)
    return {"real": real, "synthetic": synthetic}


def _validate_confidence(confidence: float) -> None:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0.")
