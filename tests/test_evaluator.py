from __future__ import annotations

import numpy as np
import pandas as pd

from synthmarket.evaluator import StylizedFactsEvaluator


def make_paths(n_paths: int = 3, rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frames = []
    dates = pd.date_range("2022-01-03", periods=rows, freq="B")
    for path_id in range(n_paths):
        close = 100 * np.exp(np.cumsum(rng.standard_t(df=5, size=rows) * 0.01))
        frame = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": rng.integers(1000, 3000, rows),
                "path_id": path_id,
            },
            index=dates,
        )
        frames.append(frame)
    combined = pd.concat(frames)
    combined = combined.set_index("path_id", append=True).reorder_levels(["path_id", None])
    combined.index = combined.index.set_names(["path_id", "date"])
    return combined


def test_evaluator_returns_report_with_core_metrics() -> None:
    real = make_paths(n_paths=1).droplevel("path_id")
    synthetic = make_paths(n_paths=3)

    report = StylizedFactsEvaluator(real, synthetic, max_lag=5, rolling_window=10).evaluate()

    assert report.status in {"pass", "warn", "fail"}
    assert "ks_statistic" in report.metrics
    assert "squared_return_acf_mse" in report.metrics
    assert "memorization" in report.metrics

