# Strategy Benchmark: Real vs. Synthetic Market Data

> Status: **template -- not yet populated with a real run.**
> Generate by running `python scripts/generate_results.py --benchmark --checkpoint <path>`,
> which drives `synthmarket.benchmark.run_benchmark` and the `backtrader`/`vectorbt`
> integrations against both a real historical series and generated synthetic paths.

## Method

1. Load real OHLCV data for the benchmark ticker(s).
2. Generate N synthetic paths of equal length from the trained WGAN-GP checkpoint via
   `SyntheticMarketGenerator`.
3. Run each strategy in `synthmarket.strategies` against real data and against the synthetic
   paths using the `backtrader` and `vectorbt` integrations.
4. Aggregate Sharpe ratio, max drawdown, and win rate; report mean +/- std across synthetic
   paths so single-path noise is visible.

## Results

_(populate by running the script -- placeholders below)_

| Strategy | Sharpe (real) | Sharpe (synthetic, mean +/- std) | Max DD (real) | Max DD (synthetic) | Win rate (real) | Win rate (synthetic) |
|---|---|---|---|---|---|---|
| `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## Interpretation guide

- Large Sharpe gaps between real and synthetic suggest the generator is not capturing the
  return/volatility dynamics the strategy actually depends on.
- Wide std across synthetic paths for the same strategy is expected and useful: it shows the
  strategy's sensitivity to path variation that a single historical backtest can't reveal.
