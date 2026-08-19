# Contributing to SynthMarket

Thanks for looking at SynthMarket. This document describes how the project is developed so
contributions (including your own future PRs) stay easy to review and trust.

## Workflow

1. Create a branch per change: `feature/<short-name>` or `fix/<short-name>`.
2. Keep commits small and scoped to one logical change. Prefer several commits with clear
   messages over one large commit ("Add gradient penalty to WGAN critic loss", not "wip" or
   "fix lint").
3. Open a pull request into `main` describing *why* the change matters, not just what changed.
4. Wait for CI (`.github/workflows/ci.yml`) to pass before merging. If CI fails, push a
   follow-up commit with the fix rather than force-pushing over the failure silently.
5. Tag releases using semantic versioning (`vX.Y.Z`) with short release notes.

## AI-assisted contributions

Parts of this project have used AI coding assistants during development. If a commit or PR
was substantially AI-generated, note it in the commit body (e.g. `Assisted-by: <tool>`) or PR
description. Review AI-generated diffs line by line before merging -- you are responsible for
code you commit, whether you or a tool wrote the first draft.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

## Reproducing research results

See `docs/RESULTS.md` and `docs/BENCHMARK_RESULTS.md` for instructions on regenerating the
stylized-facts comparison and strategy benchmark tables using `scripts/generate_results.py`.
Do not hand-edit the numbers in those files -- regenerate them from a real run so the reported
figures stay trustworthy.
