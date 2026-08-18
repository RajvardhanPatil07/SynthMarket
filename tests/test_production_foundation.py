"""Tests for the production/research foundation added in v0.2."""

import numpy as np
import pandas as pd

from synthmarket import (
    build_manifest,
    chronological_split,
    expected_shortfall,
    maximum_drawdown,
    regime_report,
    value_at_risk,
)


def _prices(n: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
    return pd.DataFrame({"Close": close}, index=index)


def test_chronological_split_is_ordered() -> None:
    frame = _prices(120)
    split = chronological_split(frame, validation_fraction=0.2, test_fraction=0.2, min_train_rows=10)
    assert split.train.index.max() < split.validation.index.min()
    assert split.validation.index.max() < split.test.index.min()


def test_manifest_fingerprint_is_stable() -> None:
    frame = _prices(20)
    first = build_manifest(frame, dataset_id="demo", version="1", source="test")
    second = build_manifest(frame.copy(), dataset_id="demo", version="1", source="test")
    assert first.fingerprint == second.fingerprint


def test_tail_metrics_are_positive_losses() -> None:
    returns = np.array([-0.10, -0.04, -0.02, 0.01, 0.02] * 10, dtype=float)
    assert value_at_risk(returns, 0.95) > 0
    assert expected_shortfall(returns, 0.95) >= value_at_risk(returns, 0.95)


def test_maximum_drawdown() -> None:
    prices = pd.Series([100.0, 110.0, 88.0, 100.0])
    assert maximum_drawdown(prices) == 0.2


def test_regime_report_is_serializable() -> None:
    frame = _prices(100)
    report = regime_report(frame["Close"])
    assert "distribution" in report
    assert "transition_matrix" in report
