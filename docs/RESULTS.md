# Stylized-Facts Results: Real vs. Synthetic

> Status: **template -- not yet populated with a real run.**
> Do not fill in numbers here by hand. Generate this file by running
> `python scripts/generate_results.py --ticker <TICKER> --checkpoint <path>` against a real
> trained checkpoint and real historical data, then commit the regenerated file.

## Why this file exists

Architecture (a WGAN-GP generator, temporal validation, tail-risk diagnostics) is not evidence
that synthetic paths are statistically realistic. This file is where that evidence lives:
comparisons between a real historical OHLCV series and a synthetic series generated from the
same starting conditions.

## What `scripts/generate_results.py` reports

| Diagnostic | What it checks | Source |
|---|---|---|
| Return distribution (mean, std, skew, kurtosis) | Fat tails / non-normality match | `StylizedFactsEvaluator` |
| ACF of returns (lags 1-10) | Absence of naive linear autocorrelation | `StylizedFactsEvaluator` |
| ACF of squared returns (lags 1-10) | Volatility clustering | `StylizedFactsEvaluator` |
| Tail risk (VaR / CVaR at 95%, 99%) | Extreme-move realism | `compare_tail_risk` |
| Volatility regime classification | Regime-switching behavior | `classify_volatility_regimes` |

## Real vs. synthetic comparison table

_(populate by running the script -- placeholders below)_

| Metric | Real | Synthetic | Delta |
|---|---|---|---|
| Mean daily return | `TBD` | `TBD` | `TBD` |
| Std daily return | `TBD` | `TBD` | `TBD` |
| Skew | `TBD` | `TBD` | `TBD` |
| Excess kurtosis | `TBD` | `TBD` | `TBD` |
| ACF(1) of returns | `TBD` | `TBD` | `TBD` |
| ACF(1) of squared returns | `TBD` | `TBD` | `TBD` |
| VaR 95% | `TBD` | `TBD` | `TBD` |
| CVaR 95% | `TBD` | `TBD` | `TBD` |

## Plots

Regenerate and commit under `docs/assets/`:
- `return_distribution.png`
- `acf_returns.png`
- `acf_squared_returns.png`
- `wgan_training_loss.png` (critic vs. generator loss per epoch)

## Known limitations

- Synthetic data is not a substitute for real out-of-sample validation (see README caveat).
- A single ticker/date range is not sufficient evidence of general realism; repeat across at
  least 3-5 tickers and regimes before making strong claims.
