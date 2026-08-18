# SynthMarket

Synthetic market histories for strategy research, robustness testing, and backtest stress scenarios.

SynthMarket is a local-first Python library and optional API/browser service for generating statistically plausible synthetic OHLCV market data with PyTorch. The current baseline is a recurrent WGAN-GP, while the v0.2 architecture adds pluggable model backends, reproducible dataset and experiment manifests, leakage-safe temporal validation, tail-risk diagnostics, volatility-regime analysis, benchmark helpers, and a containerized FastAPI service layer.

> **Research software, not financial advice.** Synthetic market data must never replace real out-of-sample validation, and generated paths are not historical ground truth.

## What SynthMarket Builds

```text
historical OHLCV
    -> validate / clean / version
    -> leakage-safe train / validation / test split
    -> train a registered generative model
    -> generate synthetic market paths
    -> validate OHLC constraints and memorization
    -> evaluate statistical / temporal / dependence / tail fidelity
    -> compare synthetic-to-real trading utility
    -> backtest and stress-test strategies
    -> persist reproducible experiment metadata
    -> export research artifacts and reports
```

## Why v0.2

The project is moving from a single-model demo into a reproducible research and deployment platform. The WGAN-GP baseline remains intentionally available so newer models can be compared against it rather than replacing it without evidence.

### v0.2 additions

- Pluggable `SyntheticModel` protocol and model registry.
- Dataset manifests with deterministic SHA-256 fingerprints.
- Chronological train/validation/test splitting with no shuffling.
- VaR, expected shortfall, maximum drawdown, and extreme-loss diagnostics.
- Volatility-regime frequency and transition analysis.
- Synthetic-to-real utility comparison primitives.
- Experiment manifests capturing dataset identity, random seed, Python/platform, Git SHA, and configuration hash.
- Repeated-seed benchmark utilities.
- Optional FastAPI `/health` and `/models` service endpoints.
- Docker and Docker Compose deployment scaffolding.
- Multi-version GitHub Actions CI and dependency auditing.
- Model card, data card, production protocol, and licensing guidance.

## Install

SynthMarket targets Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For the hosted API:

```bash
python -m pip install -e ".[api]"
uvicorn synthmarket.api:app --host 127.0.0.1 --port 8000
```

For the full development environment:

```bash
python -m pip install -e ".[all]"
```

## Research quickstart

```python
import pandas as pd
from synthmarket import build_manifest, chronological_split, regime_report

# `ohlcv` should be the exact cleaned DataFrame used for the experiment.
manifest = build_manifest(
    ohlcv,
    dataset_id="SPY_DAILY",
    version="2026.08.18",
    source="provider-or-internal-source",
)

split = chronological_split(ohlcv, validation_fraction=0.15, test_fraction=0.15)
print(manifest.to_json())
print(split.sizes)
print(regime_report(split.test["Close"]))
```

The existing WGAN-GP training API continues to work for end-to-end synthetic generation. See `examples/` for runnable workflows.

## Evaluation protocol

For publishable results, use the same protocol for every model:

1. Record the exact dataset manifest and fingerprint.
2. Split chronologically into train, validation, and test periods.
3. Fit scalers and models only on training data.
4. Tune model parameters using training/validation periods only.
5. Generate synthetic paths without accessing the held-out test period.
6. Evaluate distributional, temporal, cross-asset, tail-risk, and memorization metrics.
7. Evaluate trading utility on unseen real data.
8. Repeat stochastic models across multiple seeds and report mean/std or confidence intervals.
9. Store the experiment manifest and Git SHA with every reported result.

## Production deployment

The optional API is intentionally thin so the research core stays easy to install. A hosted deployment should place durable infrastructure around it:

```text
Web / SDK -> FastAPI -> Job Queue -> Workers -> Model Registry
                    |             |            |
                 Postgres      Object Store   PyTorch
```

The repository includes a container for the API:

```bash
docker compose up --build
```

Additional deployment work should add authentication/RBAC, durable job queues, object storage, rate limits, structured logging, metrics/tracing, and multi-tenant experiment persistence.

## Current model strategy

### WGAN-GP baseline

The existing recurrent WGAN-GP is retained as the baseline model. It supports GRU/LSTM recurrent generation, single-asset and multi-asset flows, and the existing training/checkpoint pipeline.

### Next research backends

The architecture is now ready for additional registered backends such as:

- TimeGAN
- diffusion models
- temporal Transformers
- probabilistic/statistical baselines such as block bootstrap or GARCH

These models should be added through the registry and evaluated through the same benchmark protocol.

## Data and licensing

The repository does not ship proprietary financial datasets. `yfinance` is supported as a convenience provider, but data-provider terms and redistribution restrictions remain the user's responsibility. See `docs/DATA_CARD.md` and `LICENSES.md`.

## Documentation

- `docs/PRODUCTION.md` — production architecture and research checklist.
- `docs/MODEL_CARD_WGAN_GP.md` — intended use, limitations, and evaluation protocol for the baseline model.
- `docs/DATA_CARD.md` — dataset provenance and reproducibility requirements.

## Existing capabilities

SynthMarket still includes the original research features:

- Single-asset and correlated multi-asset OHLCV flows.
- Robust data cleaning, scaling, and sliding-window sequencing.
- Strategy templates and no-code strategy specs.
- Portfolio backtesting across generated paths.
- Stylized-fact evaluation and memorization checks.
- Local SQLite persistence for saved runs and strategy specs.
- Browser dashboard for training, generation, evaluation, backtesting, comparison, and exports.
- VectorBT-ready and Backtrader-ready export adapters.

## Development

Run the local checks:

```bash
ruff check . --no-cache
python -m pytest -p no:cacheprovider
python -m compileall synthmarket tests examples
```

GitHub Actions also runs the test suite on Python 3.10, 3.11, and 3.12 and performs a dependency audit.

## Disclaimer

SynthMarket is research software. It is not financial advice, an investment recommendation, a market-data redistribution service, or a production trading engine.

## License

MIT License. See `LICENSE`.
