"""VectorBT-friendly export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_vectorbt_close_matrix(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return a wide close-price matrix suitable for VectorBT inputs."""

    if "Close" not in ohlcv.columns:
        raise ValueError("OHLCV frame must contain a Close column.")
    close = ohlcv["Close"].astype(float)
    if not isinstance(close.index, pd.MultiIndex):
        return close.to_frame("asset_0")

    names = [str(name) for name in close.index.names]
    if "path_id" in names and "asset" in names:
        matrix = close.unstack(["path_id", "asset"])
        matrix.columns = pd.MultiIndex.from_tuples(matrix.columns, names=["path_id", "asset"])
        return matrix.sort_index(axis=1)
    if "path_id" in names:
        matrix = close.unstack("path_id")
        matrix.columns = pd.MultiIndex.from_product([matrix.columns, ["asset_0"]], names=["path_id", "asset"])
        return matrix.sort_index(axis=1)
    if "asset" in names:
        return close.unstack("asset").sort_index(axis=1)
    return close.unstack(level=0).sort_index(axis=1)


def write_vectorbt_close_matrix_csv(ohlcv: pd.DataFrame, path: str | Path) -> Path:
    """Write a VectorBT-ready close matrix CSV and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    to_vectorbt_close_matrix(ohlcv).to_csv(output)
    return output
