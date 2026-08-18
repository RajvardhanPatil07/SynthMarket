"""Reproducible experiment manifests for SynthMarket runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ExperimentManifest:
    """Portable metadata for reproducing and auditing an experiment."""

    experiment_id: str
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    model: str
    seed: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    git_commit: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["parameters"] = dict(self.parameters)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


def current_git_commit() -> Optional[str]:
    """Return the current Git commit when this package runs inside a checkout."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
