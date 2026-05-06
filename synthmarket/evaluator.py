"""Stylized-fact diagnostics for synthetic financial time-series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .data_utils import clean_ohlcv


@dataclass
class EvaluationReport:
    """Practical V1 realism report."""

    status: str
    metrics: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "metrics": self.metrics, "warnings": list(self.warnings)}


class StylizedFactsEvaluator:
    """Compare real and synthetic OHLCV paths using practical market diagnostics."""

    def __init__(
        self,
        real_ohlcv: pd.DataFrame,
        synthetic_ohlcv: pd.DataFrame,
        max_lag: int = 20,
        rolling_window: int = 21,
    ) -> None:
        self.real_ohlcv = clean_ohlcv(real_ohlcv)
        self.synthetic_ohlcv = clean_ohlcv(synthetic_ohlcv)
        self.max_lag = max_lag
        self.rolling_window = rolling_window
        if max_lag <= 0:
            raise ValueError("max_lag must be positive.")
        if rolling_window <= 1:
            raise ValueError("rolling_window must be greater than 1.")

    def evaluate(self) -> EvaluationReport:
        real_returns = _log_returns(self.real_ohlcv)
        synthetic_returns = _log_returns(self.synthetic_ohlcv)

        real_values = real_returns.to_numpy(dtype=float)
        synthetic_values = synthetic_returns.to_numpy(dtype=float)
        if len(real_values) < self.max_lag + 2 or len(synthetic_values) < self.max_lag + 2:
            raise ValueError("Not enough returns to evaluate stylized facts.")

        real_acf = _average_grouped_acf(real_returns, self.max_lag)
        synthetic_acf = _average_grouped_acf(synthetic_returns, self.max_lag)
        real_squared_acf = _average_grouped_acf(real_returns.pow(2), self.max_lag)
        synthetic_squared_acf = _average_grouped_acf(synthetic_returns.pow(2), self.max_lag)

        real_rolling_vol = _rolling_volatility(real_returns, self.rolling_window)
        synthetic_rolling_vol = _rolling_volatility(synthetic_returns, self.rolling_window)
        memorization = nearest_neighbor_memorization(
            real_returns,
            synthetic_returns,
            window_size=min(20, max(3, self.max_lag)),
        )

        ks_result = stats.ks_2samp(real_values, synthetic_values)
        metrics: dict[str, Any] = {
            "real_return_stats": _summary_stats(real_values),
            "synthetic_return_stats": _summary_stats(synthetic_values),
            "tail_quantiles": {"real": _quantiles(real_values), "synthetic": _quantiles(synthetic_values)},
            "ks_statistic": float(ks_result.statistic),
            "ks_pvalue": float(ks_result.pvalue),
            "wasserstein_distance": float(stats.wasserstein_distance(real_values, synthetic_values)),
            "return_acf": {"real": real_acf.tolist(), "synthetic": synthetic_acf.tolist()},
            "squared_return_acf": {"real": real_squared_acf.tolist(), "synthetic": synthetic_squared_acf.tolist()},
            "return_acf_mse": float(np.mean((real_acf - synthetic_acf) ** 2)),
            "squared_return_acf_mse": float(np.mean((real_squared_acf - synthetic_squared_acf) ** 2)),
            "ljung_box_returns": {
                "real": _ljung_box(real_values, self.max_lag),
                "synthetic": _ljung_box(synthetic_values, self.max_lag),
            },
            "ljung_box_squared_returns": {
                "real": _ljung_box(real_values**2, self.max_lag),
                "synthetic": _ljung_box(synthetic_values**2, self.max_lag),
            },
            "rolling_volatility": {
                "real_mean": float(np.nanmean(real_rolling_vol)),
                "synthetic_mean": float(np.nanmean(synthetic_rolling_vol)),
                "wasserstein_distance": float(stats.wasserstein_distance(real_rolling_vol, synthetic_rolling_vol)),
            },
            "memorization": memorization,
        }
        status, warnings = _quality_gate(metrics)
        return EvaluationReport(status=status, metrics=metrics, warnings=warnings)

    def plot_return_distribution(self, path: Optional[Union[str, Path]] = None):
        import matplotlib.pyplot as plt

        real = _log_returns(self.real_ohlcv)
        synthetic = _log_returns(self.synthetic_ohlcv)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(real, bins=60, density=True, alpha=0.55, label="Real")
        ax.hist(synthetic, bins=60, density=True, alpha=0.55, label="Synthetic")
        ax.set_title("Daily log-return distribution")
        ax.set_xlabel("Log return")
        ax.set_ylabel("Density")
        ax.legend()
        return _save_or_return(fig, path)

    def plot_qq(self, path: Optional[Union[str, Path]] = None):
        import matplotlib.pyplot as plt

        real = _log_returns(self.real_ohlcv).to_numpy(dtype=float)
        synthetic = _log_returns(self.synthetic_ohlcv).to_numpy(dtype=float)
        quantiles = np.linspace(0.01, 0.99, 99)
        real_q = np.quantile(real, quantiles)
        synthetic_q = np.quantile(synthetic, quantiles)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(real_q, synthetic_q, s=16, alpha=0.75)
        lower = float(min(real_q.min(), synthetic_q.min()))
        upper = float(max(real_q.max(), synthetic_q.max()))
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        ax.set_title("Real vs synthetic return QQ")
        ax.set_xlabel("Real quantiles")
        ax.set_ylabel("Synthetic quantiles")
        return _save_or_return(fig, path)

    def plot_acf(self, squared: bool = False, path: Optional[Union[str, Path]] = None):
        import matplotlib.pyplot as plt

        real = _log_returns(self.real_ohlcv)
        synthetic = _log_returns(self.synthetic_ohlcv)
        if squared:
            real = real.pow(2)
            synthetic = synthetic.pow(2)
        real_acf = _average_grouped_acf(real, self.max_lag)
        synthetic_acf = _average_grouped_acf(synthetic, self.max_lag)
        lags = np.arange(1, self.max_lag + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lags, real_acf, marker="o", label="Real")
        ax.plot(lags, synthetic_acf, marker="o", label="Synthetic")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Squared-return ACF" if squared else "Return ACF")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.legend()
        return _save_or_return(fig, path)

    def plot_rolling_volatility(self, path: Optional[Union[str, Path]] = None):
        import matplotlib.pyplot as plt

        real = _rolling_volatility(_log_returns(self.real_ohlcv), self.rolling_window)
        synthetic = _rolling_volatility(_log_returns(self.synthetic_ohlcv), self.rolling_window)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(real, bins=50, density=True, alpha=0.55, label="Real")
        ax.hist(synthetic, bins=50, density=True, alpha=0.55, label="Synthetic")
        ax.set_title(f"{self.rolling_window}-day rolling volatility")
        ax.set_xlabel("Rolling volatility")
        ax.set_ylabel("Density")
        ax.legend()
        return _save_or_return(fig, path)

    def plot_path_preview(self, path: Optional[Union[str, Path]] = None, max_paths: int = 20):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(self.real_ohlcv["Close"].reset_index(drop=True), color="black", linewidth=2, label="Real")
        if isinstance(self.synthetic_ohlcv.index, pd.MultiIndex):
            for _, group in list(self.synthetic_ohlcv.groupby(level=0))[:max_paths]:
                ax.plot(group["Close"].reset_index(drop=True), alpha=0.35, linewidth=1)
        else:
            ax.plot(self.synthetic_ohlcv["Close"].reset_index(drop=True), alpha=0.55, label="Synthetic")
        ax.set_title("Synthetic close path preview")
        ax.set_xlabel("Step")
        ax.set_ylabel("Close")
        ax.legend()
        return _save_or_return(fig, path)

    def plot_all(self, output_dir: Union[str, Path]) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "return_distribution": output / "return_distribution.png",
            "qq": output / "qq.png",
            "return_acf": output / "return_acf.png",
            "squared_return_acf": output / "squared_return_acf.png",
            "rolling_volatility": output / "rolling_volatility.png",
            "path_preview": output / "path_preview.png",
        }
        self.plot_return_distribution(paths["return_distribution"])
        self.plot_qq(paths["qq"])
        self.plot_acf(False, paths["return_acf"])
        self.plot_acf(True, paths["squared_return_acf"])
        self.plot_rolling_volatility(paths["rolling_volatility"])
        self.plot_path_preview(paths["path_preview"])
        return paths


def nearest_neighbor_memorization(
    real_returns: pd.Series,
    synthetic_returns: pd.Series,
    window_size: int = 20,
    max_windows: int = 2000,
) -> dict[str, float]:
    """Estimate whether generated return windows are near copies of training windows."""

    real_windows = _return_windows(real_returns, window_size, max_windows)
    synthetic_windows = _return_windows(synthetic_returns, window_size, max_windows)
    if len(real_windows) == 0 or len(synthetic_windows) == 0:
        return {"min_distance": np.nan, "median_distance": np.nan, "near_duplicate_rate": np.nan}

    distances = []
    for synthetic_window in synthetic_windows:
        delta = real_windows - synthetic_window
        norm = np.sqrt(np.mean(delta * delta, axis=1))
        distances.append(float(np.min(norm)))
    distances_array = np.asarray(distances, dtype=float)
    return {
        "min_distance": float(np.min(distances_array)),
        "median_distance": float(np.median(distances_array)),
        "near_duplicate_rate": float(np.mean(distances_array < 1e-8)),
    }


def _log_returns(ohlcv: pd.DataFrame) -> pd.Series:
    close = ohlcv["Close"].astype(float)
    if isinstance(ohlcv.index, pd.MultiIndex):
        returns = close.groupby(level=0, group_keys=False).apply(lambda values: np.log(values / values.shift(1)))
    else:
        returns = np.log(close / close.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "skew": float(stats.skew(values, bias=False)),
        "kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(values, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)}


def _average_grouped_acf(series: pd.Series, max_lag: int) -> np.ndarray:
    if isinstance(series.index, pd.MultiIndex):
        acfs = [
            _acf(group.to_numpy(dtype=float), max_lag)
            for _, group in series.groupby(level=0)
            if len(group) > max_lag
        ]
        if acfs:
            return np.nanmean(np.vstack(acfs), axis=0)
    return _acf(series.to_numpy(dtype=float), max_lag)


def _acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values - np.mean(values)
    denominator = float(np.dot(values, values))
    if denominator <= 0:
        return np.zeros(max_lag, dtype=float)
    acf_values = []
    for lag in range(1, max_lag + 1):
        if lag >= len(values):
            acf_values.append(np.nan)
        else:
            acf_values.append(float(np.dot(values[:-lag], values[lag:]) / denominator))
    return np.asarray(acf_values, dtype=float)


def _ljung_box(values: np.ndarray, lags: int) -> dict[str, float]:
    acf = _acf(values, lags)
    n = len(values)
    valid_lags = np.arange(1, len(acf) + 1)
    statistic = n * (n + 2) * np.nansum((acf**2) / np.maximum(n - valid_lags, 1))
    return {"statistic": float(statistic), "pvalue": float(stats.chi2.sf(statistic, len(valid_lags)))}


def _rolling_volatility(returns: pd.Series, window: int) -> np.ndarray:
    if isinstance(returns.index, pd.MultiIndex):
        rolling = returns.groupby(level=0, group_keys=False).rolling(window).std().reset_index(level=0, drop=True)
    else:
        rolling = returns.rolling(window).std()
    return rolling.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _return_windows(series: pd.Series, window_size: int, max_windows: int) -> np.ndarray:
    groups = [series]
    if isinstance(series.index, pd.MultiIndex):
        groups = [group for _, group in series.groupby(level=0)]
    windows = []
    for group in groups:
        values = group.to_numpy(dtype=float)
        for start in range(0, max(0, len(values) - window_size + 1)):
            windows.append(values[start : start + window_size])
    if not windows:
        return np.empty((0, window_size), dtype=float)
    array = np.asarray(windows, dtype=float)
    if len(array) > max_windows:
        positions = np.linspace(0, len(array) - 1, max_windows).astype(int)
        array = array[positions]
    return array


def _quality_gate(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    fail = False

    if metrics["memorization"]["near_duplicate_rate"] > 0.05:
        fail = True
        warnings.append("Synthetic return windows contain too many exact near-duplicates of real windows.")
    if metrics["ks_statistic"] > 0.35:
        fail = True
        warnings.append("Synthetic return distribution is far from the real return distribution.")
    elif metrics["ks_statistic"] > 0.20:
        warnings.append("Synthetic return distribution differs noticeably from the real return distribution.")

    real_vol = metrics["rolling_volatility"]["real_mean"]
    synthetic_vol = metrics["rolling_volatility"]["synthetic_mean"]
    if np.isfinite(real_vol) and real_vol > 0:
        ratio = synthetic_vol / real_vol
        if ratio < 0.3 or ratio > 3.0:
            fail = True
            warnings.append("Synthetic rolling volatility is outside the practical V1 tolerance band.")
        elif ratio < 0.5 or ratio > 2.0:
            warnings.append("Synthetic rolling volatility differs materially from real rolling volatility.")

    if metrics["squared_return_acf_mse"] > 0.05:
        warnings.append("Synthetic volatility clustering differs materially from real volatility clustering.")

    if fail:
        return "fail", warnings
    if warnings:
        return "warn", warnings
    return "pass", warnings


def _save_or_return(fig, path: Optional[Union[str, Path]]):
    if path is None:
        return fig
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=160)
    return fig
