from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthmarket.backtester import SMACrossoverConfig, run_sma_crossover
from synthmarket.web import build_preview, coerce_params, compact_backtest, json_safe


def test_coerce_params_validates_dashboard_inputs() -> None:
    params = coerce_params({"ticker": " spy ", "epochs": "2", "n_paths": "10"})

    assert params["ticker"] == "SPY"
    assert params["epochs"] == 2
    assert params["n_paths"] == 10
    assert params["sma_short"] == 20
    assert params["sma_long"] == 50
    with pytest.raises(ValueError, match="epochs"):
        coerce_params({"ticker": "SPY", "epochs": "0"})
    with pytest.raises(ValueError, match="sma_long"):
        coerce_params({"ticker": "SPY", "sma_short": "50", "sma_long": "20"})


def test_build_preview_and_json_safe_handle_numpy_values() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="B")
    close = np.linspace(100, 112, len(dates))
    real = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.arange(1000, 1012),
        },
        index=dates,
    )
    synthetic = real.copy()
    synthetic["path_id"] = 0
    synthetic = synthetic.set_index("path_id", append=True).reorder_levels(["path_id", None])
    synthetic.index = synthetic.index.set_names(["path_id", "date"])

    preview = build_preview(real, synthetic)
    payload = json_safe({"preview": preview, "bad": np.nan, "value": np.float64(1.5)})

    assert payload["bad"] is None
    assert payload["value"] == 1.5
    assert len(payload["preview"]["synthetic_paths"]) == 1


def test_compact_backtest_payload_contains_summary_rows() -> None:
    real = pd.DataFrame(
        {
            "Open": np.linspace(100, 120, 40),
            "High": np.linspace(101, 121, 40),
            "Low": np.linspace(99, 119, 40),
            "Close": np.linspace(100, 120, 40),
            "Volume": np.full(40, 1000),
        },
        index=pd.date_range("2021-01-01", periods=40, freq="B"),
    )
    report = run_sma_crossover(real, SMACrossoverConfig(short_window=3, long_window=8))
    payload = compact_backtest(report)

    assert "aggregate" in payload
    assert "equity_preview" in payload
    assert "return_histogram" in payload
    assert payload["per_path"][0]["path_id"] == 0
