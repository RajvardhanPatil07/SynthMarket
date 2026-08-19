"""Command-line workflows for SynthMarket."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _load_frame(path: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path), index_col=0, parse_dates=True)
    if frame.empty:
        raise ValueError(f"CSV file '{path}' contains no rows.")
    return frame


def _close_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {', '.join(map(str, frame.columns))}")
    close = pd.to_numeric(frame[column], errors="coerce").dropna()
    if close.empty or (close <= 0).any():
        raise ValueError(f"Column '{column}' must contain positive numeric prices.")
    return close


def _handle_models(args: argparse.Namespace) -> dict[str, object]:
    from .models.registry import list_models

    return {"models": list_models()}


def _handle_evaluate(args: argparse.Namespace) -> dict[str, object]:
    from .evaluation.regimes import regime_report
    from .evaluation.tail_risk import returns_from_close, tail_risk_report

    close = _close_series(_load_frame(args.csv), args.close_column)
    return {"tail_risk": tail_risk_report(returns_from_close(close)), "regimes": regime_report(close)}


def _handle_split(args: argparse.Namespace) -> dict[str, object]:
    from .validation.temporal import chronological_split

    split = chronological_split(
        _load_frame(args.csv),
        validation_fraction=args.validation,
        test_fraction=args.test,
        min_train_rows=args.min_train_rows,
    )

    def span(part: pd.DataFrame) -> dict[str, object]:
        return {"rows": len(part), "start": str(part.index.min()), "end": str(part.index.max())}

    return {"train": span(split.train), "validation": span(split.validation), "test": span(split.test)}


def _handle_validate(args: argparse.Namespace) -> dict[str, object]:
    from .validation.ohlc import validate_ohlcv

    return validate_ohlcv(_load_frame(args.csv)).to_dict()


def _handle_generate(args: argparse.Namespace) -> dict[str, object]:
    from .evaluation.tail_risk import returns_from_close
    from .models.registry import get_model

    close = _close_series(_load_frame(args.csv), args.close_column)
    returns = returns_from_close(close)
    if len(returns) < 30:
        raise ValueError("At least 30 historical returns are required to fit a baseline model.")
    model = get_model(args.model)
    if args.model not in {"block-bootstrap", "garch"}:
        raise ValueError(f"Model '{args.model}' is not supported by this fit/generate CLI workflow.")
    model.fit(returns.reshape(1, -1, 1))
    generated = model.generate(args.paths, args.length, seed=args.seed)
    last_close = float(close.iloc[-1])
    prices = last_close * np.exp(np.cumsum(generated[:, :, 0], axis=1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prices.T, columns=[f"path_{index + 1}" for index in range(args.paths)]).to_csv(
        output, index_label="step"
    )
    return {
        "model": args.model,
        "paths": args.paths,
        "length": args.length,
        "seed": args.seed,
        "start_price": last_close,
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthmarket", description="SynthMarket research workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models_parser = subparsers.add_parser("models", help="List registered model backends.")
    models_parser.set_defaults(handler=_handle_models)

    evaluate_parser = subparsers.add_parser("evaluate", help="Tail-risk and regime report for a price CSV.")
    evaluate_parser.add_argument("--csv", required=True)
    evaluate_parser.add_argument("--close-column", default="Close")
    evaluate_parser.set_defaults(handler=_handle_evaluate)

    split_parser = subparsers.add_parser("split", help="Leakage-safe chronological split summary.")
    split_parser.add_argument("--csv", required=True)
    split_parser.add_argument("--validation", type=float, default=0.15)
    split_parser.add_argument("--test", type=float, default=0.15)
    split_parser.add_argument("--min-train-rows", type=int, default=100)
    split_parser.set_defaults(handler=_handle_split)

    validate_parser = subparsers.add_parser("validate", help="Structural OHLCV constraint validation.")
    validate_parser.add_argument("--csv", required=True)
    validate_parser.set_defaults(handler=_handle_validate)

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic price paths with a baseline model.")
    generate_parser.add_argument("--csv", required=True)
    generate_parser.add_argument("--close-column", default="Close")
    generate_parser.add_argument("--model", default="block-bootstrap")
    generate_parser.add_argument("--paths", type=int, default=10)
    generate_parser.add_argument("--length", type=int, default=252)
    generate_parser.add_argument("--seed", type=int, default=None)
    generate_parser.add_argument("--output", default="synthetic_paths.csv")
    generate_parser.set_defaults(handler=_handle_generate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = args.handler(args)
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
