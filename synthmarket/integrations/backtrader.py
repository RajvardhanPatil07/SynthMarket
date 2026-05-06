"""Backtrader-friendly CSV bundle exports."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


def write_backtrader_csv_bundle(ohlcv: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write one OHLCV CSV per path/asset and zip the bundle."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, frame in _iter_ohlcv_slices(ohlcv):
        csv_path = output / f"{label}.csv"
        csv_frame = frame.reset_index()
        csv_frame.to_csv(csv_path, index=False)
        manifest.append({"label": label, "file": csv_path.name})

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps({"files": manifest}, indent=2), encoding="utf-8")
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, arcname=manifest_path.name)
        for item in manifest:
            archive.write(output / item["file"], arcname=item["file"])
    return zip_path


def _iter_ohlcv_slices(ohlcv: pd.DataFrame):
    if not isinstance(ohlcv.index, pd.MultiIndex):
        yield "path_0_asset_0", ohlcv
        return

    names = [str(name) for name in ohlcv.index.names]
    if "path_id" in names and "asset" in names:
        for (path_id, asset), group in ohlcv.groupby(level=["path_id", "asset"]):
            yield f"path_{path_id}_asset_{asset}", group.droplevel(["path_id", "asset"])
        return
    if "path_id" in names:
        for path_id, group in ohlcv.groupby(level="path_id"):
            yield f"path_{path_id}_asset_0", group.droplevel("path_id")
        return
    if "asset" in names:
        for asset, group in ohlcv.groupby(level="asset"):
            yield f"path_0_asset_{asset}", group.droplevel("asset")
        return
    for key, group in ohlcv.groupby(level=0):
        yield f"path_{key}_asset_0", group.droplevel(0)
