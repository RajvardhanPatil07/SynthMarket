"""Reproducible benchmark helpers for comparing synthetic-market generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class BenchmarkResult:
    """One model's benchmark outcome."""

    model: str
    seed: int
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "seed": self.seed, "metrics": dict(self.metrics)}


def summarize_results(results: Iterable[BenchmarkResult]) -> list[dict[str, Any]]:
    """Aggregate repeated runs into mean/std metrics per model."""

    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)

    summary: list[dict[str, Any]] = []
    for model, model_results in sorted(grouped.items()):
        metric_names = sorted({name for result in model_results for name in result.metrics})
        metrics: dict[str, dict[str, float]] = {}
        for name in metric_names:
            values = [result.metrics[name] for result in model_results if name in result.metrics]
            array = np.asarray(values, dtype=float)
            metrics[name] = {
                "mean": float(np.nanmean(array)),
                "std": float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0,
            }
        summary.append({"model": model, "runs": len(model_results), "metrics": metrics})
    return summary


def run_benchmark(
    model_names: Iterable[str],
    evaluator: Callable[[str, int], Mapping[str, float]],
    *,
    seeds: Iterable[int] = (1, 2, 3),
) -> list[BenchmarkResult]:
    """Run the same evaluator across models and repeated seeds."""

    results: list[BenchmarkResult] = []
    for model in model_names:
        for seed in seeds:
            metrics = dict(evaluator(model, int(seed)))
            results.append(BenchmarkResult(model=model, seed=int(seed), metrics=metrics))
    return results
