"""Leakage-safe temporal dataset splits for financial time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological train/validation/test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def chronological_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    min_train_rows: int = 100,
    cutoff: Optional[str] = None,
) -> TemporalSplit:
    """Split a time-indexed frame without shuffling or future leakage."""

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")
    if min_train_rows <= 0:
        raise ValueError("min_train_rows must be positive.")

    ordered = frame.sort_index().copy()
    if cutoff is not None:
        cutoff_ts = pd.Timestamp(cutoff)
        train = ordered.loc[ordered.index < cutoff_ts]
        remaining = ordered.loc[ordered.index >= cutoff_ts]
        if len(train) < min_train_rows:
            raise ValueError("cutoff leaves fewer than min_train_rows in training data.")
        validation_size = max(
            1,
            int(
                round(
                    len(remaining)
                    * validation_fraction
                    / (validation_fraction + test_fraction)
                )
            ),
        )
        validation = remaining.iloc[:validation_size]
        test = remaining.iloc[validation_size:]
    else:
        n = len(ordered)
        train_end = int(n * (1 - validation_fraction - test_fraction))
        validation_end = int(n * (1 - test_fraction))
        if train_end < min_train_rows:
            raise ValueError("Not enough rows for the requested train/validation/test split.")
        train = ordered.iloc[:train_end]
        validation = ordered.iloc[train_end:validation_end]
        test = ordered.iloc[validation_end:]

    if len(validation) == 0 or len(test) == 0:
        raise ValueError("Each split must contain at least one row.")
    return TemporalSplit(train=train, validation=validation, test=test)
