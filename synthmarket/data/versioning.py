"""Reproducible dataset manifests and fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable metadata describing exactly what an experiment trained on."""

    dataset_id: str
    version: str
    source: str
    fingerprint: str
    rows: int
    columns: tuple[str, ...]
    start: Optional[str] = None
    end: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["columns"] = list(self.columns)
        result["parameters"] = dict(self.parameters)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint for a tabular dataset."""

    normalized = frame.copy()
    normalized = normalized.sort_index()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    payload = pd.util.hash_pandas_object(normalized, index=True).to_numpy().tobytes()
    schema = json.dumps(
        {"columns": list(normalized.columns), "dtypes": {c: str(normalized[c].dtype) for c in normalized.columns}},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(schema + payload).hexdigest()


def build_manifest(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    version: str,
    source: str,
    parameters: Optional[Mapping[str, Any]] = None,
) -> DatasetManifest:
    """Build a manifest from the exact DataFrame used by an experiment."""

    start = end = None
    if len(frame.index):
        try:
            start = str(frame.index.min())
            end = str(frame.index.max())
        except TypeError:
            pass
    return DatasetManifest(
        dataset_id=dataset_id,
        version=version,
        source=source,
        fingerprint=dataframe_fingerprint(frame),
        rows=len(frame),
        columns=tuple(str(column) for column in frame.columns),
        start=start,
        end=end,
        parameters=dict(parameters or {}),
    )
