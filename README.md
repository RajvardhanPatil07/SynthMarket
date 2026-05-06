# SynthMarket

Synthetic market histories for strategy research, robustness testing, and backtest stress scenarios.

SynthMarket is a local-first Python library and browser dashboard for generating statistically plausible synthetic OHLCV market data with PyTorch. It trains recurrent WGAN-GP models on historical asset data, generates alternative market paths, evaluates stylized facts, and backtests trading strategies across those paths.

The goal is not price prediction. The goal is to ask better research questions:

- What happens to this strategy across thousands of plausible market histories?
- Does the generated data preserve return distributions, volatility behavior, and cross-asset relationships?
- Which strategy parameters survive adverse synthetic scenarios?
- Can the results be exported into a standard backtesting workflow?

## Highlights

- Recurrent WGAN-GP generator and critic built with PyTorch `nn.Module`.
- Single-asset and correlated multi-asset OHLCV flows.
- yfinance data fetching plus robust OHLCV cleaning, scaling, and sliding-window sequencing.
- Strategy templates and no-code strategy specs:
  - Buy and Hold
  - SMA Crossover
  - EMA Crossover
  - RSI Mean Reversion
  - Bollinger Mean Reversion
  - Donchian Breakout
- Portfolio backtesting across generated paths.
- Stylized-fact evaluator for return distributions, tails, volatility clustering, autocorrelation, and memorization checks.
- Local SQLite persistence for saved runs and strategy specs.
- Browser dashboard for training, generation, evaluation, backtesting, comparison, exports, and previous-result viewing.
- Export adapters for VectorBT-ready close matrices and Backtrader-ready OHLCV bundles.

## What SynthMarket Builds

```text
historical OHLCV
    -> clean and align
    -> scale and window
    -> train recurrent WGAN-GP
    -> generate synthetic OHLCV paths
    -> repair market constraints
    -> evaluate realism
    -> backtest strategies
    -> compare saved runs
    -> export CSV / ZIP / checkpoint artifacts
```

## Install

SynthMarket targets Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Runtime dependencies are declared in `pyproject.toml`: PyTorch, Pandas, NumPy, SciPy, yfinance, and Matplotlib.

## Quickstart: Python API

```python
from synthmarket.data_utils import MarketDataConfig, WindowConfig, fetch_yfinance_ohlcv, prepare_market_data
from synthmarket.evaluator import StylizedFactsEvaluator
from synthmarket.generator import SyntheticMarketGenerator
from synthmarket.models.wgan import WGANConfig
from synthmarket.trainer import TrainingConfig, WGANTrainer

ohlcv = fetch_yfinance_ohlcv(MarketDataConfig(ticker="SPY", period="10y"))
prepared = prepare_market_data(ohlcv, WindowConfig(window_size=252, stride=1))

trainer = WGANTrainer(
    WGANConfig(feature_dim=prepared.windows.shape[-1]),
    TrainingConfig(
        epochs=100,
        batch_size=64,
        checkpoint_path="artifacts/spy_wgan.pt",
        device="auto",
    ),
)

artifact = trainer.fit(prepared)

generator = SyntheticMarketGenerator.from_artifact(artifact, device="auto")
synthetic = generator.generate_paths(n_paths=1000, length=252)

report = StylizedFactsEvaluator(real_ohlcv=ohlcv, synthetic_ohlcv=synthetic).evaluate()
print(report.status)
print(report.metrics["ks_statistic"])
```

The generated output is a Pandas DataFrame with a `path_id, date` MultiIndex and standard OHLCV columns:

```text
Open, High, Low, Close, Volume
```

That shape is intentionally easy to adapt for VectorBT, Backtrader, Zipline-style loaders, custom Pandas backtests, and research notebooks.

## Run the Local Dashboard

```bash
source .venv/bin/activate
python -m synthmarket.web --host 127.0.0.1 --port 8765 --auto-port
```

Then open:

```text
http://127.0.0.1:8765/
```

The dashboard lets you:

- fetch single or multi-asset data,
- train a WGAN-GP model,
- generate synthetic paths,
- evaluate stylized facts,
- build strategies from templates,
- backtest across generated paths,
- inspect previous backtest results,
- compare saved runs,
- download synthetic data, backtest CSVs, strategy JSON, model checkpoints, VectorBT CSVs, and Backtrader ZIP bundles.

## Fast Smoke Run

For a quick CPU-friendly dashboard test:

```text
Tickers: SPY,QQQ
Period: 1y
Epochs: 1
Window: 10
Paths: 2
Length: 30
Mode: Correlated Multi-Asset
```

This is not a quality model; it is only meant to verify that the pipeline works end to end.

## Example Script

```bash
python examples/train_spy_wgan.py \
  --ticker SPY \
  --period 5y \
  --epochs 20 \
  --n-paths 100 \
  --length 252
```

Outputs are written under `artifacts/` by default. This directory is ignored by Git because it can contain generated data, plots, checkpoints, and local SQLite state.

## Project Structure

```text
synthmarket/
  data_utils.py              # fetching, cleaning, scaling, sequencing
  models/wgan.py             # recurrent WGAN-GP architectures
  trainer.py                 # adversarial training loop and checkpoints
  generator.py               # synthetic OHLCV generation facade
  evaluator.py               # stylized-fact metrics and plots
  backtester.py              # SMA and portfolio backtesting
  strategies.py              # declarative strategy specs/templates
  multi_asset.py             # aligned multi-asset data shaping
  storage.py                 # local SQLite persistence
  integrations/
    vectorbt.py              # VectorBT-ready exports
    backtrader.py            # Backtrader-ready exports
  static/                    # local web UI assets
  web.py                     # dependency-light HTTP dashboard
tests/                       # unit and smoke tests
examples/                    # runnable examples
```

## Evaluator Metrics

SynthMarket does not just generate paths and hope they look good. The evaluator reports practical diagnostics:

- return mean, standard deviation, skew, and kurtosis,
- tail quantiles,
- Kolmogorov-Smirnov statistic,
- Wasserstein distance,
- raw-return autocorrelation,
- squared-return autocorrelation,
- rolling volatility comparison,
- correlation and covariance distance for multi-asset runs,
- nearest-neighbor memorization checks,
- pass / warn / fail quality gates.

The quality gate is a research diagnostic, not a proof that synthetic paths are true market samples.

## Backtesting

The built-in V1 backtester is intentionally deterministic and educational. It supports:

- long-only strategies,
- close-to-close execution,
- basic transaction fees,
- SMA crossover,
- strategy specs from templates,
- portfolio weights and rebalancing,
- per-path metrics:
  - total return,
  - annualized return,
  - annualized volatility,
  - Sharpe ratio,
  - max drawdown,
  - trade count,
  - win rate,
  - final equity.

Aggregate output includes median return, 5th/95th percentile return, worst drawdown, percent profitable, and a robustness score.

## Exports

The dashboard writes and serves:

- synthetic OHLCV CSV,
- backtest results CSV,
- strategy JSON,
- model checkpoint,
- VectorBT-ready close matrix CSV,
- Backtrader OHLCV CSV bundle ZIP,
- evaluator plots.

VectorBT and Backtrader are optional. SynthMarket does not require them at runtime.

## Development

Run the full local checks:

```bash
ruff check . --no-cache
python -m pytest -p no:cacheprovider
python -m compileall synthmarket tests examples
```

The current test suite covers:

- data cleaning and transforms,
- WGAN tensor shapes and serialization behavior,
- trainer/generator smoke paths,
- evaluator metrics,
- strategy templates,
- portfolio backtesting,
- SQLite persistence,
- export adapters,
- web helper behavior.

## Roadmap

- Advanced Python strategy sandbox.
- More strategy templates and parameter sweeps.
- Better portfolio-level reporting.
- TimeGAN and diffusion model backends.
- Stronger multi-asset correlation modeling.
- Optional VectorBT and Backtrader execution adapters.
- Notebook examples and benchmark datasets.
- Model cards for generated datasets.

## Research Caveats

Synthetic financial data can make backtests more robust, but it can also create false confidence. Always compare synthetic results against real historical regimes, out-of-sample tests, transaction-cost assumptions, liquidity constraints, and common-sense market structure.

SynthMarket is research software. It is not financial advice, not an investment recommendation, and not a production trading system.

## License

MIT License. See `LICENSE`.
