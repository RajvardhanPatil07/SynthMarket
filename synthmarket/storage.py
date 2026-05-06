"""Local SQLite persistence for SynthMarket projects, strategies, and runs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from .strategies import StrategySpec

DEFAULT_DB_PATH = Path("artifacts") / "projects" / "synthmarket.db"


class SynthMarketStore:
    """Tiny local-first persistence layer for the dashboard."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def create_project(self, name: str = "Default Project") -> dict[str, Any]:
        project_id = str(int(time.time() * 1000))
        created_at = time.time()
        with self._connect() as conn:
            conn.execute(
                "insert into projects(project_id, name, created_at) values (?, ?, ?)",
                (project_id, name.strip() or "Default Project", created_at),
            )
        return {"project_id": project_id, "name": name.strip() or "Default Project", "created_at": created_at}

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("select * from projects order by created_at desc").fetchall()
        return [_row_to_dict(row) for row in rows]

    def save_strategy(self, strategy: StrategySpec, project_id: Optional[str] = None) -> dict[str, Any]:
        strategy_id = str(int(time.time() * 1000))
        created_at = time.time()
        payload = strategy.to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                insert into strategies(strategy_id, project_id, name, template, payload_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (strategy_id, project_id, strategy.name, strategy.template, json.dumps(payload), created_at),
            )
        return {
            "strategy_id": strategy_id,
            "project_id": project_id,
            "name": strategy.name,
            "template": strategy.template,
            "payload": payload,
            "created_at": created_at,
        }

    def list_strategies(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = "select * from strategies"
        params: tuple[Any, ...] = ()
        if project_id:
            query += " where project_id = ?"
            params = (project_id,)
        query += " order by created_at desc"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_decode_json_columns(row, ("payload_json",)) for row in rows]

    def save_run(
        self,
        job_id: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        backtest: dict[str, Any],
        files: dict[str, str],
        project_id: Optional[str] = None,
        strategy: Optional[StrategySpec] = None,
    ) -> dict[str, Any]:
        created_at = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                insert or replace into runs(
                    run_id, project_id, strategy_name, strategy_template, params_json,
                    metrics_json, backtest_json, files_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    None if strategy is None else strategy.name,
                    None if strategy is None else strategy.template,
                    json.dumps(params),
                    json.dumps(metrics),
                    json.dumps(backtest),
                    json.dumps(files),
                    created_at,
                ),
            )
        return self.get_run(job_id) or {}

    def list_runs(self, limit: int = 50, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = "select * from runs"
        params: tuple[Any, ...] = ()
        if project_id:
            query += " where project_id = ?"
            params = (project_id,)
        query += " order by created_at desc limit ?"
        params = (*params, int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        json_columns = ("params_json", "metrics_json", "backtest_json", "files_json")
        return [_decode_json_columns(row, json_columns) for row in rows]

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("select * from runs where run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return _decode_json_columns(row, ("params_json", "metrics_json", "backtest_json", "files_json"))

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        runs = [run for run_id in run_ids if (run := self.get_run(run_id)) is not None]
        rows = []
        for run in runs:
            backtest = run.get("backtest_json") or {}
            aggregate = backtest.get("aggregate") or {}
            rows.append(
                {
                    "run_id": run["run_id"],
                    "strategy": run.get("strategy_name") or "-",
                    "template": run.get("strategy_template") or "-",
                    "median_return": aggregate.get("median_return"),
                    "return_p05": aggregate.get("return_p05"),
                    "return_p95": aggregate.get("return_p95"),
                    "worst_drawdown": aggregate.get("worst_drawdown"),
                    "percent_profitable": aggregate.get("percent_profitable"),
                    "robustness_score": aggregate.get("robustness_score"),
                }
            )
        rows.sort(key=lambda row: (row.get("robustness_score") is None, -(row.get("robustness_score") or 0.0)))
        return {"runs": rows, "count": len(rows)}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                create table if not exists projects (
                    project_id text primary key,
                    name text not null,
                    created_at real not null
                );

                create table if not exists strategies (
                    strategy_id text primary key,
                    project_id text,
                    name text not null,
                    template text not null,
                    payload_json text not null,
                    created_at real not null
                );

                create table if not exists runs (
                    run_id text primary key,
                    project_id text,
                    strategy_name text,
                    strategy_template text,
                    params_json text not null,
                    metrics_json text not null,
                    backtest_json text not null,
                    files_json text not null,
                    created_at real not null
                );
                """
            )


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _decode_json_columns(row: sqlite3.Row, columns: tuple[str, ...]) -> dict[str, Any]:
    payload = _row_to_dict(row)
    for column in columns:
        payload[column] = json.loads(payload[column]) if payload.get(column) else None
    return payload
