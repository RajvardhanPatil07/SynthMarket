# SynthMarket

[![CI](https://github.com/RajvardhanPatil07/SynthMarket/actions/workflows/ci.yml/badge.svg)](https://github.com/RajvardhanPatil07/SynthMarket/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/RajvardhanPatil07/SynthMarket)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Synthetic market histories for strategy research, robustness testing, and backtest stress scenarios.

SynthMarket is a local-first Python library and optional API/browser service for generating statistically plausible synthetic OHLCV market data. Its recurrent WGAN-GP can now be evaluated against built-in block-bootstrap and Gaussian GARCH(1,1) baselines through a common model registry. The project includes reproducible manifests, leakage-safe temporal validation, tail-risk and regime diagnostics, structural OHLCV validation, a CLI, benchmark helpers, and an optional FastAPI service.

> **Research software, not financial advice.** Synthetic data must not replace real out-of-sample validation and generated paths are not historical truth.

## What's new in v0.3

- Registered `block-bootstrap` and `garch` statistical baselines.
- Added a `synthmarket` CLI for model listing, generation, evaluation, splitting, and validation.
- Added structural OHLCV validation with detailed violation counts.
- Added `POST /generate` to the optional API.
- Corrected the WGAN-GP schedule to run `n_critic` critic updates per generator update on fresh minibatches.
- Added separate generator/critic learning rates and deterministic training mode.
- Made checkpoint loading try PyTorch's safe `weights_only=True` mode first.
- Added baseline, persistence, determinism, no-leakage, CLI, and API tests.
- Added coverage, type checking, release notes, contribution guidance, and citation metadata.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For the API:

```bash
python -m pip install -e ".[api]"
uvicorn synthmarket.api:app --host 127.0.0.1 --port 8000
```

For everything: `python -m pip install -e ".[all]"`.

## CLI

```bash
synthmarket models
synthmarket validate --csv prices.csv
synthmarket split --csv prices.csv --validation 0.15 --test 0.15
synthmarket evaluate --csv prices.csv --close-column Close
synthmarket generate --csv prices.csv --model block-bootstrap \
  --paths 25 --length 252 --seed 42 --output synthetic.csv
```

Every subcommand prints JSON, making it easy to use from notebooks and shell pipelines.

## Python quickstart

```python
from synthmarket import build_manifest, chronological_split, regime_report, validate_ohlcv

validate_ohlcv(ohlcv, raise_on_error=True)
manifest = build_manifest(
    ohlcv,
    dataset_id="SPY_DAILY",
    version="2026.08.19",
    source="your-provider",
)
split = chronological_split(ohlcv, validation_fraction=0.15, test_fraction=0.15)
print(manifest.to_json())
print(split.sizes)
print(regime_report(split.test["Close"]))
```

## Statistical baseline benchmark

Run the deterministic benchmark before claiming that a neural model improves fidelity:

```bash
python examples/benchmark_baselines.py
python examples/benchmark_baselines.py --csv prices.csv
```

Results on the included simulated volatility-clustering market (2,500 steps, 70/30 train/test, 50 paths, seed 7):

| source | std | var_99 | es_97_5 | sq_autocorr_lag1 | sq_autocorr_lag5 |
|---|---:|---:|---:|---:|---:|
| held-out real | 0.01212 | 0.03050 | 0.03074 | 0.07339 | 0.11872 |
| block-bootstrap | 0.01131 | 0.02618 | 0.02668 | 0.07170 | 0.04473 |
| garch | 0.01106 | 0.02678 | 0.02743 | 0.14715 | 0.12864 |

The baselines closely reproduce held-out volatility and tail-loss scale; GARCH also reproduces longer-lag volatility clustering. They provide a transparent minimum bar for neural generators.

## Model backends

- **WGAN-GP** — recurrent GRU/LSTM neural baseline for single- and multi-asset sequences.
- **Block bootstrap** — resamples contiguous blocks and preserves cross-feature dependence exactly.
- **Gaussian GARCH(1,1)** — deterministic NumPy-only fitting with variance targeting for volatility clustering.

Future registered backends can include TimeGAN, diffusion models, and temporal Transformers. Each should be evaluated with the same held-out protocol.

## Evaluation protocol

1. Record the exact dataset manifest and fingerprint.
2. Split chronologically into train, validation, and test periods.
3. Fit scalers and models on training data only.
4. Tune with train/validation periods only.
5. Generate without reading the held-out test period.
6. Compare distributional, temporal, dependence, tail-risk, and memorization metrics.
7. Evaluate strategy utility on unseen real data.
8. Repeat stochastic models across seeds and report uncertainty.
9. Store the experiment manifest and Git SHA with every result.

## API and deployment

The optional FastAPI service exposes `GET /health`, `GET /models`, and `POST /generate`. Start the included container with `docker compose up --build`. A production deployment should add authentication/RBAC, durable queues, object storage, rate limits, structured logging, metrics, tracing, and persistent experiment storage.

## Development

```bash
ruff check . --no-cache
mypy synthmarket
python -m pytest -p no:cacheprovider --cov=synthmarket
python -m compileall synthmarket tests examples
```

CI runs linting, tests with coverage, and compilation on Python 3.10, 3.11, and 3.12, plus dependency auditing. See `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/PRODUCTION.md`, `docs/MODEL_CARD_WGAN_GP.md`, and `docs/DATA_CARD.md`.

## Data, disclaimer, and license

The repository does not redistribute proprietary market datasets. Users are responsible for provider terms. SynthMarket is research software—not financial advice, an investment recommendation, or a production trading engine.

MIT License. See `LICENSE`.
