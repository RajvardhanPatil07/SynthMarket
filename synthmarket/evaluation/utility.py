"""Synthetic-to-real utility diagnostics for trading research."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class UtilityComparison:
    """Compact comparison of a strategy trained on synthetic versus real data."""

    synthetic_metric: float
    real_metric: float

    @property
    def absolute_gap(self) -> float:
        return float(self.synthetic_metric - self.real_metric)

    @property
    def relative_gap(self) -> float:
        denominator = max(abs(self.real_metric), 1e-12)
        return float(self.absolute_gap / denominator)

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "absolute_gap": self.absolute_gap, "relative_gap": self.relative_gap}


def compare_metric(synthetic_values: Iterable[float], real_values: Iterable[float]) -> UtilityComparison:
    """Compare average strategy outcomes on synthetic and real out-of-sample data."""

    synthetic = np.asarray(list(synthetic_values), dtype=float)
    real = np.asarray(list(real_values), dtype=float)
    synthetic = synthetic[np.isfinite(synthetic)]
    real = real[np.isfinite(real)]
    if len(synthetic) == 0 or len(real) == 0:
        raise ValueError("Both synthetic and real metric collections must contain finite values.")
    return UtilityComparison(float(np.mean(synthetic)), float(np.mean(real)))
