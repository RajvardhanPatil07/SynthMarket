"""Benchmark built-in baselines against a held-out return series."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from synthmarket.evaluation.tail_risk import (
    expected_shortfall,
    returns_from_close,
    value_at_risk,
)
from synthmarket.models.registry import get_model


def _simulated_market(length: int = 2500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 4e-6, 0.09, 0.88
    variance = omega / (1.0 - alpha - beta)
    output = np.empty(length)
    for step in range(length):
        shock = rng.standard_normal() * np.sqrt(variance)
        output[step] = 0.0003 + shock
        variance = omega + alpha * shock**2 + beta * variance
    return output


def _squared_autocorrelation(returns: np.ndarray, lag: int) -> float:
    squared = returns**2
    left = squared[:-lag] - squared[:-lag].mean()
    right = squared[lag:] - squared[lag:].mean()
    denominator = np.sqrt(np.sum(left**2) * np.sum(right**2))
    if denominator == 0:
        return 0.0
    return float(np.sum(left * right) / denominator)


def _describe(returns: np.ndarray) -> dict[str, float]:
    return {
        "std": float(np.std(returns)),
        "var_99": value_at_risk(returns, confidence=0.99),
        "es_97_5": expected_shortfall(returns, confidence=0.975),
        "sq_autocorr_lag1": _squared_autocorrelation(returns, 1),
        "sq_autocorr_lag5": _squared_autocorrelation(returns, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--close-column", default="Close")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--paths", type=int, default=50)
    args = parser.parse_args()

    if args.csv:
        returns = returns_from_close(pd.read_csv(args.csv)[args.close_column])
    else:
        returns = _simulated_market()
    split_at = int(len(returns) * 0.70)
    train, test = returns[:split_at], returns[split_at:]
    rows = [("held-out real", _describe(test))]
    for name in ("block-bootstrap", "garch"):
        model = get_model(name)
        model.fit(train.reshape(1, -1, 1))
        generated = model.generate(args.paths, len(test), seed=args.seed)
        synthetic = generated[:, :, 0].ravel()
        rows.append((name, _describe(synthetic)))

    metrics = list(rows[0][1])
    print("| source | " + " | ".join(metrics) + " |")
    print("|" + "---|" * (len(metrics) + 1))
    for name, statistics in rows:
        values = " | ".join(f"{statistics[metric]:.5f}" for metric in metrics)
        print(f"| {name} | {values} |")


if __name__ == "__main__":
    main()
