"""Spec Compiler — compiles StrategySpec to executable Strategy + Engine.

Two modes:
  1. Direct Runtime Mode (MVP): construct Strategy object from spec and run.
  2. Generated Code Mode (future): generate strategy.py file.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from oxq.core.engine import Engine
from oxq.core.registry import _INDICATOR_REGISTRY, _PORTFOLIO_OPTIMIZER_REGISTRY, _SIGNAL_REGISTRY
from oxq.core.strategy import Strategy
from oxq.core.types import PortfolioOptimizer, Signal
from oxq.data.market import LocalMarketDataProvider
from oxq.portfolio.analytics import RunResult
from oxq.rules.constraint import RebalanceFrequencyRule
from oxq.spec.schema import StrategySpec
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage
from oxq.universe.static import StaticUniverse

FILL_PRICE_MODE_MAP: dict[str, FillPriceMode] = {
    "close": FillPriceMode.CLOSE,
    "next_open": FillPriceMode.NEXT_OPEN,
    "mid": FillPriceMode.MID,
    "next_high": FillPriceMode.NEXT_HIGH,
    "next_low": FillPriceMode.NEXT_LOW,
}

# Signals that fire on a single bar and should latch once triggered.
# NOTE: Peak is excluded because its implementation uses shift(-i)
# which introduces future-data bias. Timestamp is excluded because
# it is time-based and should re-evaluate every bar.
_EVENT_SIGNAL_TYPES = frozenset({"Crossover"})

# Frequency string → interval_days mapping.
_FREQUENCY_INTERVAL: dict[str, int] = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 21,
    "quarterly": 63,
}


def _resolve_indicator(name: str) -> type:
    """Look up an indicator class by name from the registry."""
    cls = _INDICATOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown indicator: '{name}'. Available: {sorted(_INDICATOR_REGISTRY.keys())}")
    return cls


def _resolve_signal(name: str) -> type:
    """Look up a signal class by name from the registry."""
    cls = _SIGNAL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown signal: '{name}'. Available: {sorted(_SIGNAL_REGISTRY.keys())}")
    return cls


def _resolve_portfolio_optimizer(name: str) -> type:
    """Look up a portfolio optimizer class by name from the registry."""
    cls = _PORTFOLIO_OPTIMIZER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown portfolio optimizer: '{name}'. Available: {sorted(_PORTFOLIO_OPTIMIZER_REGISTRY.keys())}"
        )
    return cls


def _build_optimizer(spec: StrategySpec) -> PortfolioOptimizer:
    """Build a portfolio optimizer from spec, using signal-filtered equal weight when appropriate."""
    opt_cls = _resolve_portfolio_optimizer(spec.portfolio.type)

    # When EqualWeight is used with signal rules, wrap it in a signal-filtered variant
    # so that only symbols with active signal (True/positive) get weight.
    if spec.portfolio.type == "EqualWeight" and spec.signal.rules:
        signal_names = list(spec.signal.rules.keys())
        signal_types = {name: defn.type for name, defn in spec.signal.rules.items()}
        return _SignalFilteredEqualWeightOptimizer(signal_names=signal_names, signal_types=signal_types)

    return opt_cls(**spec.portfolio.params)


class _SignalFilteredEqualWeightOptimizer:
    """Equal weight among symbols with active signals.

    Event-style signals (Crossover, Peak, Timestamp) latch once triggered —
    the position is held until an exit rule closes it.  Level-style signals
    (Threshold, Comparison, Formula, Composite) are re-evaluated every bar.
    """

    name = "SignalFilteredEqualWeight"

    def __init__(self, signal_names: list[str], signal_types: dict[str, str]) -> None:
        self._signal_names = signal_names
        self._signal_types = signal_types
        self._latched: dict[str, bool] = {}

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        active: list[str] = []
        for symbol, df in signals.items():
            for sig_name in self._signal_names:
                if sig_name not in df.columns:
                    continue

                val = df[sig_name].iloc[-1]
                sig_type = self._signal_types.get(sig_name, "")
                # NaN values are not a signal
                try:
                    is_true = not pd.isna(val) and bool(val)
                except Exception:
                    is_true = not pd.isna(val) and val is not None and val > 0

                if sig_type in _EVENT_SIGNAL_TYPES:
                    # Event signal — latch once triggered
                    if is_true:
                        self._latched[symbol] = True
                    if self._latched.get(symbol):
                        active.append(symbol)
                        break
                else:
                    # Level signal — re-evaluate every bar
                    if is_true:
                        active.append(symbol)
                        break

        if not active:
            return {"CASH": 1.0}

        weight = 1.0 / len(active)
        return {s: weight for s in active}


def compile_strategy(spec: StrategySpec) -> Strategy:
    """Compile a StrategySpec into an executable Strategy object.

    This is the Direct Runtime Mode — constructs Strategy, indicator instances,
    and signal instances with required_indicators wired up.
    """
    # Build signal instances with required_indicators.
    # When there are no signal rules, attach indicators to the portfolio optimizer instead.
    signals: dict[str, tuple[Signal, dict[str, Any]]] = {}
    required: dict[str, tuple[Any, dict[str, Any]]] = {}
    for ind_name, ind_def in spec.signal.indicators.items():
        ind_cls = _resolve_indicator(ind_def.type)
        ind_instance = ind_cls()
        required[ind_name] = (ind_instance, ind_def.params)

    for signal_name, signal_def in spec.signal.rules.items():
        signal_cls = _resolve_signal(signal_def.type)
        signal_instance = signal_cls() if hasattr(signal_cls, "__init__") else signal_cls()
        signal_instance.required_indicators = required
        signals[signal_name] = (signal_instance, signal_def.params)

    # Build portfolio optimizer — use signal-filtered variant when EqualWeight
    # is paired with boolean signal rules (e.g. Crossover/Threshold).
    optimizer = _build_optimizer(spec)

    # When there are no signal rules, attach indicators to the portfolio optimizer
    # so the engine still computes them (e.g. for ranking strategies like TopNRanking).
    if not spec.signal.rules and required:
        if hasattr(optimizer, "required_indicators"):
            optimizer.required_indicators = required
        else:
            optimizer.required_indicators = required

    # Build universe
    universe = StaticUniverse(tuple(spec.universe.symbols))

    return Strategy(
        name=spec.strategy_id,
        hypothesis=spec.research.hypothesis,
        benchmarks=spec.benchmark.symbols,
        universe=universe,
        signals=signals,
        portfolio=optimizer,
    )


def compile_run(
    spec: StrategySpec,
    data_dir: str | None = None,
    out_dir: str | Path = "runs/auto",
) -> tuple[RunResult, Path]:
    """Compile and run a StrategySpec, writing standardized artifacts to out_dir.

    Returns (RunResult, run_dir).
    """
    strategy = compile_strategy(spec)

    # Data provider — use spec.data.data_dir as fallback
    _data_dir = data_dir or (spec.data.data_dir or None)
    market = LocalMarketDataProvider(data_dir=Path(_data_dir)) if _data_dir else LocalMarketDataProvider()

    # Broker with fee/slippage from spec
    fee_model = PercentageFee(rate=Decimal(str(spec.cost.fee_rate)), min_fee=Decimal(str(spec.cost.fee_min)))
    slippage_model = PercentageSlippage(rate=Decimal(str(spec.cost.slippage_rate)))
    fill_mode_str = spec.execution.fill_price_mode
    fill_mode = FILL_PRICE_MODE_MAP.get(fill_mode_str)
    if fill_mode is None:
        valid = ", ".join(sorted(FILL_PRICE_MODE_MAP.keys()))
        raise ValueError(f"Unknown fill_price_mode '{fill_mode_str}'. Valid: {valid}")
    broker = SimBroker(fee_model=fee_model, slippage_model=slippage_model, fill_price_mode=fill_mode)

    # Determine date range
    train = spec.validation.train_period
    test = spec.validation.test_period
    start = train[0] if train else (test[0] if test else "2018-01-01")
    end = test[1] if test else (train[1] if train else "2025-12-31")

    # Build rules from spec
    rules: list = []
    interval_days = spec.execution.rebalance.interval_days
    # Map frequency string to interval_days when interval_days is not explicitly set
    freq = spec.execution.rebalance.frequency
    if interval_days <= 1 and freq and freq in _FREQUENCY_INTERVAL:
        interval_days = _FREQUENCY_INTERVAL[freq]
    if interval_days > 1:
        rules.append(RebalanceFrequencyRule(interval_days=interval_days))

    # Run engine
    engine = Engine()
    result = engine.run(
        strategy=strategy,
        market=market,
        broker=broker,
        start=start,
        end=end,
        initial_cash=spec.execution.initial_cash,
        lot_size=spec.execution.lot_size,
        rules=rules,
        data_start=spec.data.min_start_date or None,
    )

    # Write artifacts — include microseconds to avoid collisions on same-second runs
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    out_path = Path(out_dir)
    if out_path.name == "auto":
        run_dir = out_path.parent / f"{timestamp}_{spec.strategy_id}"
    else:
        run_dir = out_path / f"{timestamp}_{spec.strategy_id}"

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_artifacts(spec, result, run_dir, engine, effective_data_dir=_data_dir)

    return result, run_dir


def _write_artifacts(spec: StrategySpec, result: RunResult, run_dir: Path, engine: Engine, effective_data_dir: str | None = None) -> None:
    """Write all standardized backtest artifacts to run_dir."""
    run_id = run_dir.name

    # strategy_spec.yaml
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False), encoding="utf-8"
    )

    # spec_hash.txt
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n")

    # environment.json
    env = {
        "open_xquant_version": _get_version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "run_timestamp": datetime.now(UTC).isoformat(),
        "spec_hash": spec.compute_hash(),
    }
    if effective_data_dir:
        env["data_dir"] = effective_data_dir
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    # data_manifest.json
    symbols = spec.universe.symbols
    missing_ratio = _compute_missing_ratio(result.mktdata) if result.mktdata else 0.0
    manifest = {
        "provider": spec.data.provider,
        "symbols": symbols,
        "columns": spec.data.required_columns,
        "price_adjustment": spec.data.price_adjustment,
        "start": str(spec.validation.train_period[0]) if spec.validation.train_period else "",
        "end": str(spec.validation.test_period[1]) if spec.validation.test_period else "",
        "missing_ratio": missing_ratio,
    }
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # metrics.json
    metrics = _build_metrics(spec, result, run_id)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # equity_curve.csv
    equity_rows = [{"date": str(d), "value": v} for d, v in result.equity_curve]
    pd.DataFrame(equity_rows).to_csv(run_dir / "equity_curve.csv", index=False)

    # trades.csv
    trade_rows = [
        {
            "symbol": f.order.symbol,
            "side": f.order.side,
            "shares": f.order.shares,
            "filled_price": float(f.filled_price),
            "filled_at": f.filled_at,
            "fee": float(f.fee),
        }
        for f in result.trades
    ]
    pd.DataFrame(trade_rows).to_csv(run_dir / "trades.csv", index=False)

    # positions.csv — last snapshot
    pos_rows = [
        {"symbol": sym, "shares": pos.shares, "avg_cost": float(pos.avg_cost)}
        for sym, pos in result.portfolio.positions.items()
    ]
    pd.DataFrame(pos_rows).to_csv(run_dir / "positions.csv", index=False)

    # orders.csv — reconstruct from trades
    order_rows = [
        {"symbol": f.order.symbol, "side": f.order.side, "shares": f.order.shares, "order_type": f.order.order_type}
        for f in result.trades
    ]
    pd.DataFrame(order_rows).to_csv(run_dir / "orders.csv", index=False)

    # run_log.jsonl
    with open(run_dir / "run_log.jsonl", "w") as lf:
        lf.write(
            json.dumps(
                {
                    "event": "run_complete",
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "trade_count": len(result.trades),
                    "final_value": float(result.equity_curve[-1][1]) if result.equity_curve else 0.0,
                }
            )
            + "\n"
        )


def _to_timestamp(ts_val: str | object, tz: object | None = None) -> pd.Timestamp:
    """Convert a fill timestamp to pd.Timestamp, handling existing timezone."""
    ts = pd.Timestamp(ts_val)
    if ts.tz is None and tz is not None:
        ts = ts.tz_localize(tz)
    return ts


def _compute_missing_ratio(mktdata: dict[str, pd.DataFrame]) -> float:
    """Compute the fraction of missing (NaN) values across all symbol DataFrames."""
    total = 0
    missing = 0
    for df in mktdata.values():
        if df.empty:
            continue
        total += df.size
        missing += int(df.isna().sum().sum())
    return missing / total if total > 0 else 0.0


def _get_version() -> str:
    """Get open-xquant version from package metadata."""
    try:
        from importlib.metadata import version

        return version("open-xquant")
    except Exception:
        return "0.1.0"


def _build_metrics(spec: StrategySpec, result: RunResult, run_id: str) -> dict[str, Any]:
    """Build metrics dict including OOS-only metrics when test_period is defined."""
    base = {
        "strategy_id": spec.strategy_id,
        "run_id": run_id,
        "total_return": result.total_return(),
        "annualized_return": result.annualized_return(),
        "annualized_volatility": result.annualized_volatility(),
        "max_drawdown": result.max_drawdown(),
        "sharpe_ratio": result.sharpe_ratio(),
        "sortino_ratio": result.sortino_ratio(),
        "calmar_ratio": result.calmar_ratio(),
        "turnover": result.turnover() if hasattr(result, "turnover") else 0.0,
        "trade_count": len(result.trades),
        "cost_paid": float(sum(float(f.fee) for f in result.trades)),
        "slippage_paid": None,  # Not measurable without raw-vs-slipped fill price tracking
    }

    # Compute OOS-only metrics when test_period is defined
    test = spec.validation.test_period
    if test and len(test) >= 2 and len(result.equity_curve) > 1:
        # Use first equity curve date's tz to match timezone-aware timestamps
        first_dt = result.equity_curve[0][0]
        tz = getattr(pd.Timestamp(first_dt), "tz", None)
        test_start = pd.Timestamp(test[0], tz=tz)
        oos_values = [v for d, v in result.equity_curve if pd.Timestamp(d) >= test_start]
        if len(oos_values) >= 2:
            oos_returns = np.diff(np.array(oos_values, dtype=float)) / np.array(oos_values[:-1], dtype=float)
            oos_sharpe = float(np.mean(oos_returns) / np.std(oos_returns) * np.sqrt(252)) if np.std(oos_returns) > 0 else 0.0
            oos_return = (oos_values[-1] - oos_values[0]) / oos_values[0]
            peak = np.maximum.accumulate(np.array(oos_values, dtype=float))
            oos_max_dd = float(np.min((np.array(oos_values, dtype=float) - peak) / peak))
            base["oos_sharpe_ratio"] = oos_sharpe
            base["oos_total_return"] = oos_return
            base["oos_max_drawdown"] = oos_max_dd
            # Filter OOS trades
            oos_trades = [f for f in result.trades if _to_timestamp(f.filled_at, tz=tz) >= test_start]
            base["oos_trade_count"] = len(oos_trades)

    return base
