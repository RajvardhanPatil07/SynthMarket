# Contributing to SynthMarket

Thank you for improving SynthMarket.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[all]"
```

## Checks

Run the same checks as CI before opening a pull request:

```bash
ruff check . --no-cache
mypy synthmarket
python -m pytest -p no:cacheprovider --cov=synthmarket
python -m compileall synthmarket tests examples
```

## Guidelines

- Keep the research core dependency-light; place optional integrations behind extras.
- Implement new backends through the `SyntheticModel` protocol and model registry.
- Benchmark every generative backend against `block-bootstrap` and `garch` before claiming improvement.
- Fit scalers and models on training data only; never shuffle across time boundaries.
- Give every stochastic path a seed and test determinism.
- Include regression tests for fixes and positive/negative tests for features. Tests must run without a GPU.
- Update the README and changelog for user-facing changes.
- Use focused commits with imperative subjects. PRs should explain motivation, changes, and test evidence.

## Issues

Include a minimal reproduction, SynthMarket version, Python version, and operating system.

SynthMarket is research software, not financial advice.
