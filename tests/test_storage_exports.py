from __future__ import annotations

import numpy as np
import pandas as pd

from synthmarket.integrations import to_vectorbt_close_matrix, write_backtrader_csv_bundle
from synthmarket.storage import SynthMarketStore
from synthmarket.strategies import make_strategy_spec


def make_synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=6, freq="B")
    frames = []
    for path_id in [0, 1]:
        for asset, offset in [("SPY", 0.0), ("QQQ", 10.0)]:
            close = np.linspace(100 + offset, 105 + offset, len(dates))
            frame = pd.DataFrame(
                {
                    "Open": close,
                    "High": close * 1.01,
                    "Low": close * 0.99,
                    "Close": close,
                    "Volume": np.full(len(dates), 1000),
                    "path_id": path_id,
                    "asset": asset,
                },
                index=dates,
            )
            frames.append(frame)
    panel = pd.concat(frames)
    return panel.set_index(["path_id", "asset"], append=True).reorder_levels(["path_id", "asset", None])


def test_vectorbt_and_backtrader_exports(tmp_path) -> None:
    panel = make_synthetic_panel()
    matrix = to_vectorbt_close_matrix(panel)
    bundle = write_backtrader_csv_bundle(panel, tmp_path / "bt_bundle")

    assert matrix.shape[1] == 4
    assert bundle.exists()
    assert bundle.suffix == ".zip"


def test_sqlite_store_saves_strategy_and_run(tmp_path) -> None:
    store = SynthMarketStore(tmp_path / "synthmarket.db")
    project = store.create_project("Lab")
    strategy = make_strategy_spec("buy_hold", name="Hold")
    saved_strategy = store.save_strategy(strategy, project_id=project["project_id"])
    saved_run = store.save_run(
        "run-1",
        params={"ticker": "SPY"},
        metrics={"status": "pass"},
        backtest={"aggregate": {"median_return": 0.1, "robustness_score": 0.8}},
        files={"synthetic_csv": "synthetic.csv"},
        project_id=project["project_id"],
        strategy=strategy,
    )

    assert saved_strategy["name"] == "Hold"
    assert saved_run["run_id"] == "run-1"
    assert store.compare_runs(["run-1"])["runs"][0]["robustness_score"] == 0.8
