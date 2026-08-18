"""Simple, deterministic market-regime diagnostics for generated paths."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def classify_volatility_regimes(returns: pd.Series, quantiles: tuple[float, float] = (0.33, 0.66)) -> pd.Series:
    """Label observations as low, medium, or high volatility."""

    if len(returns.dropna()) < 9:
        raise ValueError("At least 9 returns are required for regime classification.")
    rolling = returns.astype(float).rolling(21, min_periods=5).std()
    finite = rolling.dropna()
    low, high = finite.quantile(quantiles).tolist()
    return pd.cut(
        rolling,
        bins=[-np.inf, low, high, np.inf],
        labels=["low_vol", "medium_vol", "high_vol"],
        include_lowest=True,
    ).astype("string")


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """Compute a row-normalized regime transition matrix."""

    sequence = labels.dropna().astype(str).tolist()
    if len(sequence) < 2:
        raise ValueError("At least two regime labels are required.")
    states = sorted(set(sequence))
    counts = pd.DataFrame(0.0, index=states, columns=states)
    for current, nxt in zip(sequence, sequence[1:]):
        counts.loc[current, nxt] += 1.0
    totals = counts.sum(axis=1).replace(0.0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0)


def regime_report(close: pd.Series) -> dict[str, Any]:
    """Return regime frequencies and transition probabilities for a price series."""

    returns = np.log(close.astype(float) / close.astype(float).shift(1)).dropna()
    labels = classify_volatility_regimes(returns)
    distribution = labels.value_counts(normalize=True, dropna=True).to_dict()
    matrix = transition_matrix(labels)
    return {
        "distribution": {str(k): float(v) for k, v in distribution.items()},
        "transition_matrix": matrix.to_dict(orient="index"),
    }
