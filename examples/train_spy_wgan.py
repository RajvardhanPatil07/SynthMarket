"""Train a small SynthMarket WGAN-GP model and write synthetic paths plus plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthmarket.data_utils import MarketDataConfig, WindowConfig, fetch_yfinance_ohlcv, prepare_market_data
from synthmarket.evaluator import StylizedFactsEvaluator
from synthmarket.generator import SyntheticMarketGenerator
from synthmarket.models.wgan import WGANConfig
from synthmarket.trainer import TrainingConfig, WGANTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SynthMarket on yfinance OHLCV data.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=252)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--length", type=int, default=252)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ohlcv = fetch_yfinance_ohlcv(MarketDataConfig(ticker=args.ticker, period=args.period))
    prepared = prepare_market_data(ohlcv, WindowConfig(window_size=args.window_size, stride=1))

    checkpoint_path = output_dir / f"{args.ticker.lower()}_wgan.pt"
    trainer = WGANTrainer(
        WGANConfig(feature_dim=prepared.windows.shape[-1]),
        TrainingConfig(epochs=args.epochs, checkpoint_path=checkpoint_path, device=args.device),
    )
    artifact = trainer.fit(prepared)

    generator = SyntheticMarketGenerator.from_artifact(artifact, device=args.device)
    synthetic = generator.generate_paths(n_paths=args.n_paths, length=args.length)
    synthetic_path = output_dir / f"{args.ticker.lower()}_synthetic_paths.csv"
    synthetic.to_csv(synthetic_path)

    evaluator = StylizedFactsEvaluator(real_ohlcv=ohlcv, synthetic_ohlcv=synthetic)
    report = evaluator.evaluate()
    evaluator.plot_all(output_dir / "plots")

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved synthetic paths: {synthetic_path}")
    print(f"Evaluation status: {report.status}")
    for warning in report.warnings:
        print(f"- {warning}")


if __name__ == "__main__":
    main()

