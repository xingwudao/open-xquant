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
        return _SignalFilteredEqualWeightOptimizer(signal_names=signal_names)

    return opt_cls(**spec.portfolio.params)


class _SignalFilteredEqualWeightOptimizer:
    """Equal weight among symbols whose signal is currently active (True / > 0).

    Reads the last row of each registered signal column.  Only symbols
    where at least one signal column is truthy are included.  If no
    symbol qualifies, the full portfolio goes to CASH.
    """

    name = "SignalFilteredEqualWeight"

    def __init__(self, signal_names: list[str]) -> None:
        self._signal_names = signal_names

    def optimize(
        self,
        signals: dict[str, pd.DataFrame],
        indicators: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        active: list[str] = []
        for symbol, df in signals.items():
            for sig_name in self._signal_names:
                if sig_name in df.columns:
                    val = df[sig_name].iloc[-1]
                    try:
                        if bool(val):
                            active.append(symbol)
                            break
                    except Exception:
                        if val and val > 0:
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
    # Build signal instances with required_indicators
    signals: dict[str, tuple[Signal, dict[str, Any]]] = {}
    for signal_name, signal_def in spec.signal.rules.items():
        signal_cls = _resolve_signal(signal_def.type)
        signal_instance = signal_cls() if hasattr(signal_cls, "__init__") else signal_cls()

        # Wire up required_indicators from the spec's indicator definitions
        required: dict[str, tuple[Any, dict[str, Any]]] = {}
        for ind_name, ind_def in spec.signal.indicators.items():
            ind_cls = _resolve_indicator(ind_def.type)
            ind_instance = ind_cls()
            required[ind_name] = (ind_instance, ind_def.params)

        signal_instance.required_indicators = required

        signals[signal_name] = (signal_instance, signal_def.params)

    # Build portfolio optimizer — use signal-filtered variant when EqualWeight
    # is paired with boolean signal rules (e.g. Crossover/Threshold).
    optimizer = _build_optimizer(spec)

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

    # Data provider
    market = LocalMarketDataProvider(data_dir=Path(data_dir)) if data_dir else LocalMarketDataProvider()

    # Broker with fee/slippage from spec
    fee_model = PercentageFee(rate=Decimal(str(spec.cost.fee_rate)), min_fee=Decimal(str(spec.cost.fee_min)))
    slippage_model = PercentageSlippage(rate=Decimal(str(spec.cost.slippage_rate)))
    fill_mode = FILL_PRICE_MODE_MAP.get(spec.execution.fill_price_mode, FillPriceMode.NEXT_OPEN)
    broker = SimBroker(fee_model=fee_model, slippage_model=slippage_model, fill_price_mode=fill_mode)

    # Determine date range
    train = spec.validation.train_period
    test = spec.validation.test_period
    start = train[0] if train else (test[0] if test else "2018-01-01")
    end = test[1] if test else (train[1] if train else "2025-12-31")

    # Build rules from spec
    rules: list = []
    interval_days = spec.execution.rebalance.interval_days
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
    )

    # Write artifacts
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir)
    if out_path.name == "auto":
        run_dir = out_path.parent / f"{timestamp}_{spec.strategy_id}"
    else:
        run_dir = out_path / f"{timestamp}_{spec.strategy_id}"

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_artifacts(spec, result, run_dir, engine)

    return result, run_dir


def _write_artifacts(spec: StrategySpec, result: RunResult, run_dir: Path, engine: Engine) -> None:
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
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    # data_manifest.json
    symbols = spec.universe.symbols
    manifest = {
        "provider": spec.data.provider,
        "symbols": symbols,
        "columns": spec.data.required_columns,
        "price_adjustment": spec.data.price_adjustment,
        "start": spec.validation.train_period[0] if spec.validation.train_period else "",
        "end": spec.validation.test_period[1] if spec.validation.test_period else "",
    }
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # metrics.json
    metrics = {
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
        "slippage_paid": 0.0,
    }
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


def _get_version() -> str:
    """Get open-xquant version from package metadata."""
    try:
        from importlib.metadata import version

        return version("open-xquant")
    except Exception:
        return "0.1.0"
