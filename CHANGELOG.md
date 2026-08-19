# Changelog

All notable changes to SynthMarket are documented here. This project follows Keep a Changelog and Semantic Versioning.

## [0.3.0] - 2026-08-19

### Added

- Registered block-bootstrap and Gaussian GARCH(1,1) statistical baselines.
- CLI commands for listing models, generation, evaluation, chronological splitting, and OHLCV validation.
- Structural OHLCV validation with per-constraint counts.
- Baseline generation API endpoint.
- Determinism, persistence, leakage, CLI, and API tests.
- Baseline fidelity benchmark, contribution guide, changelog, and citation metadata.
- Coverage reporting and non-blocking mypy checks in CI.

### Changed

- Corrected WGAN-GP training to use `n_critic` fresh critic minibatches per generator update.
- Added separate generator and critic learning rates plus deterministic cuDNN behavior.
- Updated package metadata to version 0.3.0 and credited Rajvardhan Patil.

### Security

- Checkpoint loading attempts `weights_only=True` first; unsafe compatibility loading can be disabled.

## [0.2.0]

- Added a model protocol and registry, reproducible manifests, leakage-safe splits, tail/regime diagnostics, API scaffolding, and CI.

## [0.1.0]

- Initial recurrent WGAN-GP research pipeline.
