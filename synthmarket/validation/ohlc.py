"""Structural OHLCV validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class OhlcValidationReport:
    rows: int
    violations: dict[str, int]

    @property
    def is_valid(self) -> bool:
        return all(count == 0 for count in self.violations.values())

    def to_dict(self) -> dict[str, object]:
        return {"rows": self.rows, "is_valid": self.is_valid, "violations": dict(self.violations)}


def validate_ohlcv(frame: pd.DataFrame, *, raise_on_error: bool = False) -> OhlcValidationReport:
    """Validate positivity, finiteness, volume, and OHLC ordering constraints."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")

    numeric = frame[list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    prices = numeric[["Open", "High", "Low", "Close"]]
    violations = {
        "non_finite": int((~np.isfinite(numeric.to_numpy())).any(axis=1).sum()),
        "non_positive_price": int((prices <= 0).any(axis=1).sum()),
        "negative_volume": int((numeric["Volume"] < 0).sum()),
        "high_below_low": int((numeric["High"] < numeric["Low"]).sum()),
        "high_below_open_or_close": int(
            (numeric["High"] < numeric[["Open", "Close"]].max(axis=1)).sum()
        ),
        "low_above_open_or_close": int(
            (numeric["Low"] > numeric[["Open", "Close"]].min(axis=1)).sum()
        ),
    }
    report = OhlcValidationReport(rows=len(frame), violations=violations)
    if raise_on_error and not report.is_valid:
        raise ValueError(f"OHLCV validation failed: {report.to_dict()}")
    return report
