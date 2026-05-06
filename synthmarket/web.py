"""Local web dashboard for end-to-end SynthMarket workflows."""

from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import socket
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

from .backtester import PortfolioSpec, run_portfolio_backtest, run_strategy_backtest
from .data_utils import (
    MarketDataConfig,
    WindowConfig,
    fetch_yfinance_ohlcv,
    fetch_yfinance_ohlcv_panel,
    parse_tickers,
    prepare_market_data,
    prepare_multi_asset_market_data,
)
from .evaluator import EvaluationReport, StylizedFactsEvaluator
from .generator import MultiAssetSyntheticMarketGenerator, SyntheticMarketGenerator
from .integrations import write_backtrader_csv_bundle, write_vectorbt_close_matrix_csv
from .models.wgan import MultiAssetWGANConfig, WGANConfig
from .multi_asset import evaluate_cross_asset_realism
from .storage import DEFAULT_DB_PATH, SynthMarketStore
from .strategies import (
    StrategySpec,
    list_strategy_templates,
    make_strategy_spec,
    strategy_from_payload,
    strategy_summary,
)
from .trainer import TrainingConfig, WGANTrainer

STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_OUTPUT_DIR = Path("artifacts") / "web"


@dataclass
class WebJob:
    """In-memory state for one dashboard run."""

    job_id: str
    params: dict[str, Any]
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting to start"
    progress: float = 0.0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    output_dir: Optional[str] = None
    files: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    preview: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_seconds"] = round((self.finished_at or time.time()) - self.started_at, 2)
        return payload


class JobManager:
    """Owns background training jobs for the local dashboard."""

    def __init__(self, output_dir: Path, store: Optional[SynthMarketStore] = None) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or SynthMarketStore(DEFAULT_DB_PATH)
        self._lock = threading.Lock()
        self._jobs: dict[str, WebJob] = {}

    def latest(self) -> Optional[WebJob]:
        with self._lock:
            if not self._jobs:
                return None
            return max(self._jobs.values(), key=lambda job: job.started_at)

    def get(self, job_id: str) -> Optional[WebJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, raw_params: dict[str, Any]) -> WebJob:
        params = coerce_params(raw_params)
        with self._lock:
            running = [job for job in self._jobs.values() if job.status in {"queued", "running"}]
            if running:
                raise RuntimeError(
                    "A SynthMarket run is already active. Wait for it to finish before starting another."
                )
            job = WebJob(job_id=str(int(time.time() * 1000)), params=params)
            job.output_dir = str(self.output_dir / job.job_id)
            self._jobs[job.job_id] = job

        thread = threading.Thread(target=self._run, args=(job.job_id,), daemon=True)
        thread.start()
        return job

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return

        try:
            run_pipeline(job, self.store)
        except Exception as exc:  # noqa: BLE001 - this is the process boundary for dashboard jobs.
            job.status = "failed"
            job.stage = "failed"
            job.message = str(exc)
            job.error = traceback.format_exc()
            job.finished_at = time.time()


def coerce_params(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize dashboard input."""

    ticker_text = str(raw.get("tickers") or raw.get("ticker") or "SPY").strip().upper()
    tickers = parse_tickers(ticker_text)
    generation_mode = str(
        raw.get("generation_mode") or ("correlated_multi_asset" if len(tickers) > 1 else "single_asset")
    )
    if generation_mode not in {"single_asset", "correlated_multi_asset"}:
        raise ValueError("generation_mode must be single_asset or correlated_multi_asset.")

    strategy = _coerce_strategy(raw)
    portfolio = _coerce_portfolio(raw, tickers)
    params = {
        "ticker": tickers[0],
        "tickers": tickers,
        "ticker_text": ",".join(tickers),
        "generation_mode": generation_mode,
        "period": str(raw.get("period", "5y")),
        "epochs": _bounded_int(raw.get("epochs", 3), "epochs", 1, 500),
        "window_size": _bounded_int(raw.get("window_size", 60), "window_size", 8, 512),
        "n_paths": _bounded_int(raw.get("n_paths", 25), "n_paths", 1, 5000),
        "length": _bounded_int(raw.get("length", 126), "length", 5, 2000),
        "hidden_dim": _bounded_int(raw.get("hidden_dim", 32), "hidden_dim", 8, 512),
        "noise_dim": _bounded_int(raw.get("noise_dim", 16), "noise_dim", 4, 128),
        "num_layers": _bounded_int(raw.get("num_layers", 1), "num_layers", 1, 4),
        "batch_size": _bounded_int(raw.get("batch_size", 32), "batch_size", 1, 512),
        "n_critic": _bounded_int(raw.get("n_critic", 1), "n_critic", 1, 10),
        "sma_short": _bounded_int(raw.get("sma_short", 20), "sma_short", 2, 252),
        "sma_long": _bounded_int(raw.get("sma_long", 50), "sma_long", 3, 512),
        "initial_cash": portfolio.initial_cash,
        "fee_bps": portfolio.fee_bps,
        "rebalance_frequency": portfolio.rebalance_frequency,
        "portfolio_weights": portfolio.weights,
        "device": str(raw.get("device", "cpu")),
        "project_id": str(raw.get("project_id") or "") or None,
        "project_name": str(raw.get("project_name") or "Default Project"),
        "strategy_spec": strategy.to_dict(),
        "strategy_summary": strategy_summary(strategy),
        "portfolio_spec": portfolio.to_dict(),
    }
    if params["sma_long"] <= params["sma_short"]:
        raise ValueError("sma_long must be greater than sma_short.")
    if params["generation_mode"] == "correlated_multi_asset" and len(params["tickers"]) < 2:
        params["generation_mode"] = "single_asset"
    if params["window_size"] >= 2520:
        raise ValueError("window_size is too large for the dashboard defaults.")
    return params


def _coerce_strategy(raw: dict[str, Any]) -> StrategySpec:
    payload = raw.get("strategy_spec")
    if isinstance(payload, str) and payload.strip():
        payload = json.loads(payload)
    if isinstance(payload, dict) and payload:
        return strategy_from_payload(payload)

    template = str(raw.get("strategy_template") or "sma_crossover")
    parameters: dict[str, Any] = {}
    for key, value in raw.items():
        if str(key).startswith("strategy_param_"):
            parameters[str(key).removeprefix("strategy_param_")] = value
    if template == "sma_crossover":
        parameters.setdefault("short_window", raw.get("sma_short", 20))
        parameters.setdefault("long_window", raw.get("sma_long", 50))
        if int(parameters["long_window"]) <= int(parameters["short_window"]):
            raise ValueError("sma_long must be greater than sma_short.")
    return make_strategy_spec(
        template=template,
        parameters=parameters,
        name=str(raw.get("strategy_name") or ""),
        description=str(raw.get("strategy_description") or ""),
        advanced_code=str(raw.get("advanced_code") or ""),
    )


def _coerce_portfolio(raw: dict[str, Any], tickers: list[str]) -> PortfolioSpec:
    payload = raw.get("portfolio_spec")
    if isinstance(payload, str) and payload.strip():
        payload = json.loads(payload)
    payload = payload if isinstance(payload, dict) else {}
    weights = payload.get("weights") or _parse_weights(raw.get("portfolio_weights"), tickers)
    return PortfolioSpec(
        initial_cash=_bounded_float(
            payload.get("initial_cash", raw.get("initial_cash", 10000.0)),
            "initial_cash",
            100.0,
            100000000.0,
        ),
        fee_bps=_bounded_float(payload.get("fee_bps", raw.get("fee_bps", 1.0)), "fee_bps", 0.0, 1000.0),
        weights=weights,
        rebalance_frequency=str(payload.get("rebalance_frequency", raw.get("rebalance_frequency", "daily"))),
    )


def _parse_weights(value: Any, tickers: list[str]) -> dict[str, float] | None:
    if isinstance(value, dict):
        return {str(key).upper(): float(weight) for key, weight in value.items()}
    if not value:
        return None
    parsed: dict[str, float] = {}
    parts = str(value).replace(";", ",").split(",")
    for index, part in enumerate(parts):
        if not part.strip():
            continue
        if ":" in part:
            ticker, weight = part.split(":", 1)
            parsed[ticker.strip().upper()] = float(weight)
        elif index < len(tickers):
            parsed[tickers[index]] = float(part)
    return parsed or None


def run_pipeline(job: WebJob, store: Optional[SynthMarketStore] = None) -> None:
    """Fetch, train, generate, evaluate, and persist a dashboard run."""

    import matplotlib

    matplotlib.use("Agg", force=True)

    params = job.params
    output_dir = Path(job.output_dir or DEFAULT_OUTPUT_DIR / job.job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy = StrategySpec.from_dict(params["strategy_spec"])
    portfolio = PortfolioSpec(**params["portfolio_spec"])
    is_multi_asset = params["generation_mode"] == "correlated_multi_asset"
    slug = "_".join(params["tickers"]).lower()

    _update(job, "running", "fetch", 0.08, f"Fetching {params['ticker_text']} market data")
    if is_multi_asset:
        real_ohlcv = fetch_yfinance_ohlcv_panel(MarketDataConfig(ticker=params["ticker_text"], period=params["period"]))
    else:
        real_ohlcv = fetch_yfinance_ohlcv(MarketDataConfig(ticker=params["ticker"], period=params["period"]))

    _update(job, "running", "prepare", 0.18, "Preparing return features and training windows")
    if is_multi_asset:
        prepared = prepare_multi_asset_market_data(
            real_ohlcv,
            WindowConfig(window_size=params["window_size"], stride=1),
        )
    else:
        prepared = prepare_market_data(real_ohlcv, WindowConfig(window_size=params["window_size"], stride=1))

    _update(job, "running", "train", 0.32, f"Training WGAN-GP for {params['epochs']} epoch(s)")
    checkpoint_path = output_dir / f"{slug}_wgan.pt"
    model_config = (
        MultiAssetWGANConfig(
            asset_count=len(params["tickers"]),
            features_per_asset=prepared.windows.shape[-1] // len(params["tickers"]),
            noise_dim=params["noise_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=0.05,
        ).to_wgan_config()
        if is_multi_asset
        else WGANConfig(
            feature_dim=prepared.windows.shape[-1],
            noise_dim=params["noise_dim"],
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=0.05,
        )
    )
    trainer = WGANTrainer(
        model_config,
        TrainingConfig(
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            n_critic=params["n_critic"],
            device=params["device"],
            checkpoint_path=checkpoint_path,
        ),
    )
    artifact = trainer.fit(prepared)

    _update(job, "running", "generate", 0.72, f"Generating {params['n_paths']} synthetic path(s)")
    if is_multi_asset:
        generator = MultiAssetSyntheticMarketGenerator.from_artifact(artifact, device=params["device"])
    else:
        generator = SyntheticMarketGenerator.from_artifact(artifact, device=params["device"])
    synthetic = generator.generate_paths(n_paths=params["n_paths"], length=params["length"], batch_size=128)
    csv_path = output_dir / f"{slug}_synthetic_paths.csv"
    synthetic.to_csv(csv_path)

    _update(job, "running", "evaluate", 0.86, "Evaluating stylized facts")
    if is_multi_asset:
        eval_asset = params["tickers"][0]
        real_for_eval = real_ohlcv.xs(eval_asset, level="asset")
        synthetic_for_eval = synthetic.xs(eval_asset, level="asset")
    else:
        real_for_eval = real_ohlcv
        synthetic_for_eval = synthetic
    evaluator = StylizedFactsEvaluator(
        real_ohlcv=real_for_eval,
        synthetic_ohlcv=synthetic_for_eval,
        max_lag=min(20, max(3, params["length"] // 10)),
        rolling_window=min(21, max(3, params["length"] // 6)),
    )
    report = evaluator.evaluate()
    plot_paths = evaluator.plot_all(output_dir / "plots")
    cross_asset_metrics: dict[str, Any] = {}
    if is_multi_asset:
        cross_asset_metrics = evaluate_cross_asset_realism(real_ohlcv, synthetic)

    _update(job, "running", "backtest", 0.94, f"Stress-testing {strategy.name} across synthetic paths")
    if is_multi_asset:
        backtest_report = run_portfolio_backtest(synthetic, strategy=strategy, config=portfolio)
    else:
        backtest_report = run_strategy_backtest(synthetic, strategy=strategy, config=portfolio)
    backtest_csv_path = output_dir / "backtest_results.csv"
    backtest_report.per_path.to_csv(backtest_csv_path)
    strategy_json_path = output_dir / "strategy.json"
    strategy_json_path.write_text(json.dumps(strategy.to_dict(), indent=2), encoding="utf-8")
    vectorbt_csv_path = write_vectorbt_close_matrix_csv(synthetic, output_dir / "vectorbt_close_matrix.csv")
    backtrader_zip_path = write_backtrader_csv_bundle(synthetic, output_dir / "backtrader_bundle")

    job.files = {
        "checkpoint": str(checkpoint_path),
        "synthetic_csv": str(csv_path),
        "backtest_csv": str(backtest_csv_path),
        "strategy_json": str(strategy_json_path),
        "vectorbt_csv": str(vectorbt_csv_path),
        "backtrader_zip": str(backtrader_zip_path),
        **{name: str(path) for name, path in plot_paths.items()},
    }
    job.metrics = compact_metrics(report)
    if cross_asset_metrics:
        job.metrics["cross_asset"] = cross_asset_metrics
    job.warnings = list(report.warnings) + list(cross_asset_metrics.get("warnings", []))
    job.preview = build_preview(real_for_eval, synthetic_for_eval)
    job.preview["assets"] = list(params["tickers"])
    job.backtest = compact_backtest(backtest_report)
    if store is not None:
        project_id = _resolve_project_id(store, params)
        store.save_run(
            job.job_id,
            params=json_safe(params),
            metrics=json_safe(job.metrics),
            backtest=json_safe(job.backtest),
            files=dict(job.files),
            project_id=project_id,
            strategy=strategy,
        )
    job.status = "succeeded"
    job.stage = "done"
    job.progress = 1.0
    job.message = f"Finished with evaluation status: {report.status}"
    job.finished_at = time.time()


def compact_metrics(report: EvaluationReport) -> dict[str, Any]:
    """Keep the dashboard payload small while preserving the key diagnostics."""

    metrics = report.metrics
    real_vol = metrics["rolling_volatility"]["real_mean"]
    synthetic_vol = metrics["rolling_volatility"]["synthetic_mean"]
    vol_ratio = float(synthetic_vol / real_vol) if real_vol else np.nan
    return {
        "status": report.status,
        "ks_statistic": metrics["ks_statistic"],
        "ks_pvalue": metrics["ks_pvalue"],
        "wasserstein_distance": metrics["wasserstein_distance"],
        "return_acf_mse": metrics["return_acf_mse"],
        "squared_return_acf_mse": metrics["squared_return_acf_mse"],
        "rolling_volatility_ratio": vol_ratio,
        "memorization": metrics["memorization"],
        "real_return_stats": metrics["real_return_stats"],
        "synthetic_return_stats": metrics["synthetic_return_stats"],
        "tail_quantiles": metrics["tail_quantiles"],
    }


def _resolve_project_id(store: SynthMarketStore, params: dict[str, Any]) -> str:
    project_id = params.get("project_id")
    if project_id:
        return str(project_id)
    projects = store.list_projects()
    if projects:
        return str(projects[0]["project_id"])
    return str(store.create_project(str(params.get("project_name") or "Default Project"))["project_id"])


def compact_backtest(report) -> dict[str, Any]:
    per_path = report.per_path.sort_values("total_return").reset_index()
    equity_preview = []
    for path_id, group in list(report.equity_curves.groupby(level=0))[:8]:
        path_group = group.droplevel(0)
        equity_preview.append(
            {
                "path_id": int(path_id),
                "equity": _downsample(path_group["equity"]),
                "drawdown": _downsample(path_group["drawdown"]),
            }
        )
    return {
        "config": report.config.to_dict(),
        "aggregate": report.aggregate,
        "per_path": per_path.to_dict(orient="records"),
        "worst_paths": per_path.head(8).to_dict(orient="records"),
        "equity_preview": equity_preview,
        "return_histogram": _histogram(report.per_path["total_return"].to_numpy(dtype=float)),
        "drawdown_histogram": _histogram(report.per_path["max_drawdown"].to_numpy(dtype=float)),
    }


def build_preview(real_ohlcv: pd.DataFrame, synthetic_ohlcv: pd.DataFrame, max_paths: int = 8) -> dict[str, Any]:
    """Prepare lightweight chart series for the browser."""

    real_close = real_ohlcv["Close"].tail(252).reset_index(drop=True)
    synthetic_paths = []
    for path_id, group in list(synthetic_ohlcv.groupby(level=0))[:max_paths]:
        close = group["Close"].reset_index(drop=True)
        synthetic_paths.append({"path_id": int(path_id), "close": _downsample(close)})

    real_returns = np.log(real_ohlcv["Close"] / real_ohlcv["Close"].shift(1)).dropna()
    synthetic_returns = synthetic_ohlcv["Close"].groupby(level=0).apply(lambda values: np.log(values / values.shift(1)))
    synthetic_returns = synthetic_returns.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "real_close": _downsample(real_close),
        "synthetic_paths": synthetic_paths,
        "real_returns": _histogram(real_returns.to_numpy(dtype=float)),
        "synthetic_returns": _histogram(synthetic_returns.to_numpy(dtype=float)),
    }


def make_handler(manager: JobManager) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one JobManager."""

    class SynthMarketHandler(BaseHTTPRequestHandler):
        server_version = "SynthMarketWeb/0.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            self._route_get(send_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API.
            self._route_get(send_body=False)

        def _route_get(self, send_body: bool) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/":
                self._serve_file(STATIC_DIR / "index.html", send_body=send_body)
            elif path.startswith("/static/"):
                self._serve_file(STATIC_DIR / path.removeprefix("/static/"), send_body=send_body)
            elif path == "/api/health":
                self._json({"ok": True})
            elif path == "/api/strategies/templates":
                self._json({"templates": list_strategy_templates()})
            elif path == "/api/strategies":
                self._json({"strategies": manager.store.list_strategies()})
            elif path == "/api/projects":
                self._json({"projects": manager.store.list_projects()})
            elif path == "/api/runs":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["50"])[0])
                self._json({"runs": manager.store.list_runs(limit=limit)})
            elif path == "/api/compare":
                query = parse_qs(parsed.query)
                run_ids = ",".join(query.get("run_ids", [])).split(",")
                self._json({"comparison": manager.store.compare_runs([run_id for run_id in run_ids if run_id])})
            elif path.startswith("/api/export/"):
                self._serve_export(path)
            elif path == "/api/latest":
                latest = manager.latest()
                self._json({"job": None if latest is None else latest.to_dict()})
            elif path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                job = manager.get(job_id)
                if job is None:
                    self._json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                else:
                    self._json({"job": job.to_dict()})
            elif path.startswith("/api/files/"):
                self._serve_job_file(path)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            parsed = urlparse(self.path)
            if parsed.path == "/api/run":
                try:
                    payload = self._read_json()
                    job = manager.start(payload)
                except Exception as exc:  # noqa: BLE001 - user input/API boundary.
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._json({"job": job.to_dict()}, HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/strategies":
                try:
                    payload = self._read_json()
                    strategy = strategy_from_payload(payload.get("strategy") or payload)
                    saved = manager.store.save_strategy(strategy, project_id=payload.get("project_id"))
                except Exception as exc:  # noqa: BLE001 - user input/API boundary.
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._json({"strategy": saved}, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/projects":
                try:
                    payload = self._read_json()
                    project = manager.store.create_project(str(payload.get("name") or "Default Project"))
                except Exception as exc:  # noqa: BLE001 - user input/API boundary.
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._json({"project": project}, HTTPStatus.CREATED)
                return
            if parsed.path == "/api/compare":
                try:
                    payload = self._read_json()
                    run_ids = [str(value) for value in payload.get("run_ids", [])]
                except Exception as exc:  # noqa: BLE001 - user input/API boundary.
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._json({"comparison": manager.store.compare_runs(run_ids)})
                return
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _serve_job_file(self, path: str) -> None:
            parts = path.split("/")
            if len(parts) < 5:
                self._json({"error": "invalid file route"}, HTTPStatus.BAD_REQUEST)
                return
            job_id, key = parts[3], parts[4]
            job = manager.get(job_id)
            files = job.files if job is not None else {}
            if key not in files:
                saved_run = manager.store.get_run(job_id)
                files = saved_run.get("files_json", {}) if saved_run else {}
            if key not in files:
                self._json({"error": "file not found"}, HTTPStatus.NOT_FOUND)
                return
            self._serve_file(Path(files[key]))

        def _serve_export(self, path: str) -> None:
            parts = path.split("/")
            if len(parts) < 5:
                self._json({"error": "invalid export route"}, HTTPStatus.BAD_REQUEST)
                return
            run_id, export_format = parts[3], parts[4]
            key = {
                "synthetic": "synthetic_csv",
                "synthetic_csv": "synthetic_csv",
                "backtest": "backtest_csv",
                "backtest_csv": "backtest_csv",
                "strategy": "strategy_json",
                "strategy_json": "strategy_json",
                "vectorbt": "vectorbt_csv",
                "vectorbt_csv": "vectorbt_csv",
                "backtrader": "backtrader_zip",
                "backtrader_zip": "backtrader_zip",
            }.get(export_format)
            if key is None:
                self._json({"error": "unknown export format"}, HTTPStatus.BAD_REQUEST)
                return
            job = manager.get(run_id)
            files = job.files if job is not None else {}
            if key not in files:
                saved_run = manager.store.get_run(run_id)
                files = saved_run.get("files_json", {}) if saved_run else {}
            if key not in files:
                self._json({"error": "export not found"}, HTTPStatus.NOT_FOUND)
                return
            self._serve_file(Path(files[key]))

        def _read_json(self) -> dict[str, Any]:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            return json.loads(body.decode("utf-8") or "{}")

        def _serve_file(self, path: Path, send_body: bool = True) -> None:
            resolved = path.resolve()
            if not resolved.exists() or not resolved.is_file():
                self._json({"error": "file not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            if resolved.suffix == ".csv":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f'attachment; filename="{resolved.name}"')
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(resolved.stat().st_size))
            self.end_headers()
            if send_body:
                with resolved.open("rb") as file:
                    self.wfile.write(file.read())

        def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return SynthMarketHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SynthMarket local web dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Use the next available port if the requested one is busy.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    manager = JobManager(Path(args.output_dir), store=SynthMarketStore(Path(args.db_path)))
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(manager))
    except OSError as exc:
        address_in_use = exc.errno in {errno.EADDRINUSE, 48, 98} or "Address already in use" in str(exc)
        if not args.auto_port:
            print(
                f"Port {args.port} is already in use. Open http://{args.host}:{args.port} if the dashboard is "
                f"already running, stop that process, or rerun with --port {args.port + 1}. "
                "You can also add --auto-port."
            )
            raise SystemExit(1) from exc
        if not address_in_use:
            print(f"Could not bind to {args.host}:{args.port}; trying the next available port.")
        fallback_port = find_available_port(args.host, args.port + 1)
        server = ThreadingHTTPServer((args.host, fallback_port), make_handler(manager))

    host, port = server.server_address
    print(f"SynthMarket web dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping SynthMarket web dashboard")
    finally:
        server.server_close()


def _update(job: WebJob, status: str, stage: str, progress: float, message: str) -> None:
    job.status = status
    job.stage = stage
    job.progress = progress
    job.message = message


def find_available_port(host: str, start_port: int, attempts: int = 100) -> int:
    """Find an open localhost port for dashboard startup fallback."""

    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No available port found from {start_port} to {start_port + attempts - 1}.")


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values and non-finite floats into JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _downsample(values: pd.Series, max_points: int = 220) -> list[float]:
    array = values.to_numpy(dtype=float)
    if len(array) > max_points:
        positions = np.linspace(0, len(array) - 1, max_points).astype(int)
        array = array[positions]
    return [float(value) for value in array if np.isfinite(value)]


def _histogram(values: np.ndarray, bins: int = 44) -> dict[str, list[float]]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"x": [], "y": []}
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {"x": centers.astype(float).tolist(), "y": counts.astype(float).tolist()}


if __name__ == "__main__":
    main()
