"""Multi-asset diagnostics for correlated synthetic market paths."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_utils import clean_ohlcv_panel


def evaluate_cross_asset_realism(real_panel: pd.DataFrame, synthetic_panel: pd.DataFrame) -> dict[str, Any]:
    """Compare real and synthetic cross-asset correlation/covariance structure."""

    real_returns = _returns_matrix(clean_ohlcv_panel(real_panel))
    synthetic_returns = _synthetic_return_matrices(synthetic_panel)
    if real_returns.shape[1] < 2 or not synthetic_returns:
        return {
            "status": "warn",
            "correlation_distance": None,
            "covariance_distance": None,
            "warnings": ["Cross-asset realism needs at least two assets."],
        }

    real_corr = real_returns.corr().to_numpy(dtype=float)
    real_cov = real_returns.cov().to_numpy(dtype=float)
    corr_distances = []
    cov_distances = []
    for matrix in synthetic_returns:
        aligned = matrix.loc[:, real_returns.columns].dropna()
        if len(aligned) < 3:
            continue
        corr_distances.append(_frobenius_distance(real_corr, aligned.corr().to_numpy(dtype=float)))
        cov_distances.append(_frobenius_distance(real_cov, aligned.cov().to_numpy(dtype=float)))

    warnings = []
    corr_distance = float(np.nanmedian(corr_distances)) if corr_distances else np.nan
    cov_distance = float(np.nanmedian(cov_distances)) if cov_distances else np.nan
    if not np.isfinite(corr_distance):
        warnings.append("Synthetic paths were too short for cross-asset correlation diagnostics.")
        status = "warn"
    elif corr_distance > 0.75:
        warnings.append("Synthetic cross-asset correlations are far from the real sample.")
        status = "warn"
    else:
        status = "pass"

    return {
        "status": status,
        "assets": list(real_returns.columns),
        "correlation_distance": corr_distance,
        "covariance_distance": cov_distance,
        "real_correlation": _matrix_payload(real_returns.corr()),
        "synthetic_median_correlation_distance": corr_distance,
        "warnings": warnings,
    }


def _returns_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel["Close"].unstack("asset").astype(float)
    return np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()


def _synthetic_return_matrices(panel: pd.DataFrame) -> list[pd.DataFrame]:
    if not isinstance(panel.index, pd.MultiIndex):
        return []
    names = [str(name) for name in panel.index.names]
    if "path_id" not in names or "asset" not in names:
        return []
    matrices = []
    for _, group in panel.groupby(level="path_id"):
        asset_panel = group.droplevel("path_id")
        close = asset_panel["Close"].unstack("asset").astype(float)
        returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        matrices.append(returns)
    return matrices


def _frobenius_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, ord="fro") / max(left.shape[0], 1))


def _matrix_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {"columns": list(frame.columns), "values": frame.to_numpy(dtype=float).tolist()}
