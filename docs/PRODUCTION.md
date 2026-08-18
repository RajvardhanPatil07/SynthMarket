# Production and Research Readiness

SynthMarket v0.2 introduces the foundation needed to evolve the project from a local WGAN-GP research tool into a reproducible financial synthetic-data platform.

## Implemented in v0.2

- Pluggable synthetic-model interface and registry.
- Reproducible dataset manifests with SHA-256 fingerprints.
- Leakage-safe chronological train/validation/test splitting.
- Tail-risk diagnostics: VaR, expected shortfall, maximum drawdown, and extreme-loss rates.
- Volatility-regime classification and transition matrices.
- Synthetic-to-real utility comparison primitives.
- Experiment manifests containing dataset identity, seed, environment, Git SHA, and configuration hash.
- Optional FastAPI service layer with health and model endpoints.
- Docker image and Compose configuration for the API.
- Multi-Python CI plus dependency auditing.

## Production architecture target

The repository now has clean seams for a hosted deployment:

```text
Web / SDK -> FastAPI -> Job Queue -> Workers -> Model Registry
                    |             |            |
                 Postgres      Object Store   PyTorch
```

The API in v0.2 is intentionally small. Authentication, multi-tenant persistence, durable job queues, and object storage should be added at deployment time rather than coupled to the research core.

## Research protocol

For publishable experiments:

1. Pin the dataset version and record its fingerprint.
2. Create chronological train/validation/test partitions.
3. Fit every scaler and model only on training data.
4. Select hyperparameters using validation data only.
5. Evaluate once on the held-out test period.
6. Repeat stochastic models across multiple seeds.
7. Report statistical fidelity, temporal fidelity, dependence, tail risk, memorization, and synthetic-to-real trading utility.
8. Record the Git SHA and experiment configuration for every reported result.

## Production checklist

The next deployment-specific layer should add:

- managed PostgreSQL for experiment metadata,
- Redis or a durable queue for long training jobs,
- S3-compatible object storage for checkpoints and datasets,
- authentication and role-based access control,
- rate limiting and request validation,
- structured logs and metrics,
- OpenTelemetry/Sentry-style error reporting,
- model cards and data cards,
- formal benchmark datasets and published baselines,
- diffusion/TimeGAN/Transformer backends.

The current WGAN-GP implementation should remain as a baseline so improvements can be measured rather than assumed.
