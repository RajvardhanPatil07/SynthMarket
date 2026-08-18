# Model Card: Recurrent WGAN-GP Baseline

## Summary

SynthMarket's recurrent WGAN-GP is the current baseline synthetic-market generator. It is designed for research experiments, robustness testing, and alternative-path generation rather than price prediction or live execution.

## Intended use

- Generate synthetic OHLCV feature sequences for research.
- Stress-test strategy assumptions across multiple plausible paths.
- Benchmark newer financial generative models.

## Not intended for

- Investment advice.
- Autonomous live trading.
- Treating synthetic observations as factual market history.
- Replacing real out-of-sample validation.

## Architecture

- GRU or LSTM recurrent generator.
- Recurrent critic with optional bidirectionality.
- Wasserstein objective with gradient penalty.
- Return-oriented model features with explicit inverse reconstruction.

## Known limitations

- GAN training can be unstable and sensitive to hyperparameters.
- The model is a baseline, not the project's claimed state-of-the-art model.
- Financial tails and crisis regimes can remain difficult to reproduce.
- Synthetic fidelity does not guarantee trading utility.

## Evaluation protocol

Every model comparison should include distributional, temporal, dependence, tail-risk, memorization, and synthetic-to-real utility diagnostics. All model-selection decisions must happen before the held-out test period is evaluated.
