"""Declarative strategy specifications and built-in signal templates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .data_utils import clean_ohlcv


@dataclass(frozen=True)
class StrategySpec:
    """Serializable strategy contract shared by the web UI and backtester."""

    name: str
    template: str
    parameters: dict[str, Any] = field(default_factory=dict)
    long_only: bool = True
    description: str = ""
    advanced_code: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("strategy name is required.")
        if self.template not in TEMPLATE_DEFINITIONS:
            raise ValueError(f"Unknown strategy template: {self.template}.")
        if not self.long_only:
            raise ValueError("V2 strategy execution is long-only.")
        validate_strategy_parameters(self.template, self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "StrategySpec":
        return cls(
            name=str(state.get("name") or state.get("template") or "Strategy"),
            template=str(state.get("template", "sma_crossover")),
            parameters=dict(state.get("parameters") or {}),
            long_only=bool(state.get("long_only", True)),
            description=str(state.get("description", "")),
            advanced_code=str(state.get("advanced_code", "")),
        )


TEMPLATE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "buy_hold": {
        "label": "Buy and Hold",
        "description": "Stay fully invested after the first bar.",
        "parameters": {},
    },
    "sma_crossover": {
        "label": "SMA Crossover",
        "description": "Go long when the short simple moving average is above the long average.",
        "parameters": {
            "short_window": {
                "label": "Short SMA",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 252,
            },
            "long_window": {
                "label": "Long SMA",
                "type": "int",
                "default": 50,
                "min": 3,
                "max": 512,
            },
        },
    },
    "ema_crossover": {
        "label": "EMA Crossover",
        "description": "Go long when the short exponential moving average is above the long average.",
        "parameters": {
            "short_window": {
                "label": "Short EMA",
                "type": "int",
                "default": 12,
                "min": 2,
                "max": 252,
            },
            "long_window": {
                "label": "Long EMA",
                "type": "int",
                "default": 26,
                "min": 3,
                "max": 512,
            },
        },
    },
    "rsi_mean_reversion": {
        "label": "RSI Mean Reversion",
        "description": "Enter after oversold RSI and exit when momentum normalizes.",
        "parameters": {
            "rsi_window": {
                "label": "RSI Window",
                "type": "int",
                "default": 14,
                "min": 2,
                "max": 252,
            },
            "lower": {
                "label": "Entry RSI",
                "type": "float",
                "default": 30.0,
                "min": 1.0,
                "max": 99.0,
            },
            "exit": {
                "label": "Exit RSI",
                "type": "float",
                "default": 50.0,
                "min": 1.0,
                "max": 99.0,
            },
        },
    },
    "bollinger_mean_reversion": {
        "label": "Bollinger Mean Reversion",
        "description": "Enter below the lower band and exit near the middle band.",
        "parameters": {
            "window": {
                "label": "Window",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 252,
            },
            "entry_z": {
                "label": "Entry Z",
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 6.0,
            },
            "exit_z": {
                "label": "Exit Z",
                "type": "float",
                "default": 0.0,
                "min": -3.0,
                "max": 3.0,
            },
        },
    },
    "donchian_breakout": {
        "label": "Donchian Breakout",
        "description": "Enter on upside channel breakouts and exit on downside channel breaks.",
        "parameters": {
            "lookback": {
                "label": "Entry Lookback",
                "type": "int",
                "default": 55,
                "min": 2,
                "max": 512,
            },
            "exit_lookback": {
                "label": "Exit Lookback",
                "type": "int",
                "default": 20,
                "min": 2,
                "max": 512,
            },
        },
    },
}


def list_strategy_templates() -> list[dict[str, Any]]:
    """Return UI-friendly built-in strategy template metadata."""

    return [
        {
            "id": template_id,
            "label": definition["label"],
            "description": definition["description"],
            "parameters": definition["parameters"],
        }
        for template_id, definition in TEMPLATE_DEFINITIONS.items()
    ]


def make_strategy_spec(
    template: str = "sma_crossover",
    parameters: Optional[dict[str, Any]] = None,
    name: Optional[str] = None,
    description: str = "",
    advanced_code: str = "",
) -> StrategySpec:
    """Build a validated strategy spec with template defaults filled in."""

    if template not in TEMPLATE_DEFINITIONS:
        raise ValueError(f"Unknown strategy template: {template}.")
    defaults = {
        key: value["default"]
        for key, value in TEMPLATE_DEFINITIONS[template]["parameters"].items()
        if "default" in value
    }
    return StrategySpec(
        name=name or TEMPLATE_DEFINITIONS[template]["label"],
        template=template,
        parameters={**defaults, **(parameters or {})},
        description=description,
        advanced_code=advanced_code,
    )


def strategy_from_payload(payload: Optional[dict[str, Any]]) -> StrategySpec:
    """Normalize web/API payloads into a validated StrategySpec."""

    if not payload:
        return make_strategy_spec("sma_crossover")
    if "template" in payload:
        return make_strategy_spec(
            template=str(payload.get("template", "sma_crossover")),
            parameters=dict(payload.get("parameters") or {}),
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            advanced_code=str(payload.get("advanced_code") or ""),
        )
    return StrategySpec.from_dict(payload)


def validate_strategy_parameters(template: str, parameters: dict[str, Any]) -> None:
    """Validate strategy parameter ranges and relationships."""

    definition = TEMPLATE_DEFINITIONS[template]
    for name, spec in definition["parameters"].items():
        value = parameters.get(name, spec.get("default"))
        if spec["type"] == "int":
            parsed = _coerce_int(value, name, spec["min"], spec["max"])
        else:
            parsed = _coerce_float(value, name, spec["min"], spec["max"])
        if name in {"long_window", "exit_lookback"}:
            short_name = "short_window" if name == "long_window" else "lookback"
            if short_name in parameters and parsed <= int(parameters[short_name]):
                raise ValueError(f"{name} must be greater than {short_name}.")
    if template == "rsi_mean_reversion":
        lower = float(parameters.get("lower", definition["parameters"]["lower"]["default"]))
        exit_level = float(parameters.get("exit", definition["parameters"]["exit"]["default"]))
        if exit_level <= lower:
            raise ValueError("RSI exit must be greater than entry RSI.")


def generate_strategy_signal(ohlcv: pd.DataFrame, strategy: StrategySpec) -> pd.Series:
    """Generate a long-only target signal for one asset/path OHLCV frame."""

    frame = clean_ohlcv(ohlcv)
    close = frame["Close"].astype(float)
    params = _filled_parameters(strategy)
    if strategy.template == "buy_hold":
        signal = pd.Series(1.0, index=frame.index)
        signal.iloc[0] = 0.0
        return signal
    if strategy.template == "sma_crossover":
        short = close.rolling(
            int(params["short_window"]), min_periods=int(params["short_window"])
        ).mean()
        long = close.rolling(
            int(params["long_window"]), min_periods=int(params["long_window"])
        ).mean()
        return (short > long).astype(float).fillna(0.0)
    if strategy.template == "ema_crossover":
        short = close.ewm(
            span=int(params["short_window"]),
            adjust=False,
            min_periods=int(params["short_window"]),
        ).mean()
        long = close.ewm(
            span=int(params["long_window"]),
            adjust=False,
            min_periods=int(params["long_window"]),
        ).mean()
        return (short > long).astype(float).fillna(0.0)
    if strategy.template == "rsi_mean_reversion":
        rsi = _rsi(close, int(params["rsi_window"]))
        return _stateful_signal(
            rsi < float(params["lower"]),
            rsi > float(params["exit"]),
            frame.index,
        )
    if strategy.template == "bollinger_mean_reversion":
        window = int(params["window"])
        mean = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std(ddof=0)
        zscore = (close - mean) / std.replace(0.0, np.nan)
        return _stateful_signal(
            zscore < -float(params["entry_z"]),
            zscore > float(params["exit_z"]),
            frame.index,
        )
    if strategy.template == "donchian_breakout":
        entry_high = close.shift(1).rolling(
            int(params["lookback"]), min_periods=int(params["lookback"])
        ).max()
        exit_low = close.shift(1).rolling(
            int(params["exit_lookback"]),
            min_periods=int(params["exit_lookback"]),
        ).min()
        return _stateful_signal(
            close > entry_high,
            close < exit_low,
            frame.index,
        )
    raise ValueError(f"Unsupported strategy template: {strategy.template}.")


def strategy_summary(strategy: StrategySpec) -> str:
    """Human-readable one-line description for saved runs and exports."""

    label = TEMPLATE_DEFINITIONS[strategy.template]["label"]
    params = ", ".join(
        f"{key}={value}" for key, value in _filled_parameters(strategy).items()
    )
    return f"{strategy.name} ({label}{': ' + params if params else ''})"


def _filled_parameters(strategy: StrategySpec) -> dict[str, Any]:
    defaults = {
        key: spec.get("default")
        for key, spec in TEMPLATE_DEFINITIONS[strategy.template]["parameters"].items()
    }
    return {**defaults, **strategy.parameters}


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _stateful_signal(entry: pd.Series, exit_signal: pd.Series, index: pd.Index) -> pd.Series:
    position = 0.0
    values: list[float] = []
    entries = entry.fillna(False).to_numpy()
    exits = exit_signal.fillna(False).to_numpy()
    for should_enter, should_exit in zip(entries, exits, strict=False):
        if should_exit:
            position = 0.0
        elif should_enter:
            position = 1.0
        values.append(position)
    return pd.Series(values, index=index, dtype=float)


def _coerce_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _coerce_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed
