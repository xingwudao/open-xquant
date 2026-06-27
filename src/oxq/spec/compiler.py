"""Spec Compiler — compiles StrategySpec to executable Strategy + Engine.

The current implementation uses Direct Runtime Mode: construct Strategy objects
from the spec, run them in memory, and persist a deterministic compiled plan
artifact for audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import pprint
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
from oxq.core.types import Fill, PortfolioOptimizer, Signal
from oxq.data.loaders import resolve_data_dir
from oxq.data.market import LocalMarketDataProvider
from oxq.market_calendar import normalize_exchange_calendar
from oxq.portfolio.analytics import RunResult
from oxq.portfolio.metrics_profile import compute_equity_curve_metrics, compute_profile_metrics, metric_assumptions
from oxq.rules.constraint import RebalanceFrequencyRule
from oxq.spec.execution import derive_execution_semantics
from oxq.spec.schema import StrategySpec
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage
from oxq.universe.static import StaticUniverse

FILL_PRICE_MODE_MAP: dict[str, FillPriceMode] = {
    "close": FillPriceMode.CLOSE,
    "next_open": FillPriceMode.NEXT_OPEN,
    "next_close": FillPriceMode.NEXT_CLOSE,
    "mid": FillPriceMode.MID,
    "next_mid": FillPriceMode.NEXT_MID,
    "next_avg": FillPriceMode.NEXT_AVG,
    "next_hl2": FillPriceMode.NEXT_HL2,
}

# Signals that fire on a single bar and should latch once triggered.
# NOTE: Peak is excluded because its implementation uses shift(-i)
# which introduces future-data bias. Timestamp is excluded because
# it is time-based and should re-evaluate every bar.
_EVENT_SIGNAL_TYPES = frozenset({"Crossover"})


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
        signal_names = _terminal_signal_names(spec)
        signal_types = {name: _effective_signal_type(spec, name) for name in spec.signal.rules}
        return _SignalFilteredEqualWeightOptimizer(signal_names=signal_names, signal_types=signal_types)

    return opt_cls(**spec.portfolio.params)


def _effective_lot_size(spec: StrategySpec) -> int:
    lot_size = spec.execution.lot_size_config.default
    if isinstance(lot_size, int) and not isinstance(lot_size, bool) and lot_size > 0:
        return lot_size
    return spec.execution.lot_size


def _effective_rebalance(spec: StrategySpec) -> tuple[int, str]:
    """Return the runtime rebalance interval and the material source field."""
    execution_interval = spec.execution.rebalance.interval_days
    rebalance_rule = spec.portfolio.rules.get("rebalance")
    if rebalance_rule is None:
        return execution_interval, "execution.rebalance.interval_days"
    if rebalance_rule.type != "RebalanceFrequencyRule":
        raise ValueError("portfolio.rules.rebalance must use RebalanceFrequencyRule")
    unsupported_params = sorted(set(rebalance_rule.params) - {"interval_days"})
    if unsupported_params:
        raise ValueError(
            "portfolio.rules.rebalance.params contains unsupported keys: "
            + ", ".join(unsupported_params)
        )
    interval = rebalance_rule.params.get("interval_days")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("portfolio.rules.rebalance.params.interval_days must be a positive integer")
    if getattr(spec.execution.rebalance, "_interval_days_explicit", False) and execution_interval != interval:
        raise ValueError(
            "portfolio.rules.rebalance.params.interval_days conflicts with "
            "execution.rebalance.interval_days"
        )
    return interval, "portfolio.rules.rebalance"


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
        self._held_symbols: set[str] = set()

    def reset_symbols(self, symbols: list[str]) -> None:
        """Clear event-signal latch state for symbols that fully exited."""
        for symbol in symbols:
            self._latched.pop(symbol, None)

    def set_held_symbols(self, symbols: list[str]) -> None:
        """Tell the optimizer which latched symbols already have positions."""
        self._held_symbols = set(symbols)

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
                if isinstance(val, (bool, np.bool_)):
                    is_true = bool(val)
                else:
                    try:
                        is_true = not pd.isna(val) and float(val) > 0
                    except (TypeError, ValueError):
                        is_true = not pd.isna(val) and bool(val)

                if sig_type in _EVENT_SIGNAL_TYPES:
                    # Event signals enter only on the trigger bar. After that,
                    # keep them active only while the position is actually held.
                    if is_true:
                        self._latched[symbol] = True
                    if is_true or (self._latched.get(symbol) and symbol in self._held_symbols):
                        active.append(symbol)
                        break
                else:
                    # Level signal — re-evaluate every bar
                    if is_true:
                        active.append(symbol)
                        break

        for symbol, latched in self._latched.items():
            if latched and symbol in self._held_symbols and symbol not in active:
                active.append(symbol)

        if not active:
            return {"CASH": 1.0}

        weight = 1.0 / len(active)
        return {s: weight for s in active}


def compile_strategy(spec: StrategySpec) -> Strategy:
    """Compile a StrategySpec into an executable Strategy object.

    This is the Direct Runtime Mode — constructs Strategy, indicator instances,
    and signal instances with required_indicators wired up.
    """
    _validate_crossover_rule_count(spec)
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

    return Strategy(
        name=spec.strategy_id,
        hypothesis=spec.research.hypothesis,
        benchmarks=spec.benchmark.symbols,
        signals=signals,
        portfolio=optimizer,
        rules=_build_runtime_rules(spec),
    )


def compile_universe(spec: StrategySpec) -> StaticUniverse:
    """Compile the run universe from a StrategySpec."""
    if spec.universe.type != "static":
        raise ValueError(f"Unsupported universe.type '{spec.universe.type}'. Only 'static' is available.")
    return StaticUniverse(tuple(spec.universe.symbols))


def _build_runtime_rules(spec: StrategySpec) -> list[Any]:
    """Build runtime Rule instances declared or implied by a StrategySpec."""
    rules: list[Any] = []
    interval_days, _rebalance_source = _effective_rebalance(spec)
    if interval_days > 1:
        rules.append(RebalanceFrequencyRule(interval_days=interval_days))

    # Auto-add ExitRule for Crossover signals so positions close on reverse cross.
    for _sig_name, sig_def in spec.signal.rules.items():
        if sig_def.type == "Crossover":
            fast = sig_def.params.get("fast", "")
            slow = sig_def.params.get("slow", "")
            if fast and slow:
                from oxq.rules.exit import ExitRule

                rules.append(ExitRule(fast=fast, slow=slow))
    return rules


def compile_run(
    spec: StrategySpec,
    data_dir: str | None = None,
    out_dir: str | Path = "runs/auto",
) -> tuple[RunResult, Path]:
    """Compile and run a StrategySpec, writing standardized artifacts to out_dir.

    Returns (RunResult, run_dir).
    """
    _validate_strategy_id_for_path(spec.strategy_id)
    _validate_crossover_rule_count(spec)
    from oxq.spec.validator import validate

    validation = validate(spec)
    if validation.status == "fail":
        messages = "; ".join(error["message"] for error in validation.errors)
        raise ValueError(f"Spec validation failed: {messages}")
    strategy = compile_strategy(spec)
    universe = compile_universe(spec)

    # Data provider — use spec.data.data_dir as fallback
    _data_dir = data_dir or (spec.data.data_dir or None)
    data_path = resolve_data_dir(Path(_data_dir) if _data_dir else None).resolve()
    runtime_calendar = normalize_exchange_calendar(spec.market.calendar)
    market = LocalMarketDataProvider(data_dir=data_path, currency=spec.market.currency, calendar=runtime_calendar)

    # Broker with fee/slippage from spec
    fee_model = PercentageFee(rate=Decimal(str(spec.cost.fee_rate)), min_fee=Decimal(str(spec.cost.fee_min)))
    slippage_model = PercentageSlippage(rate=Decimal(str(spec.cost.slippage_rate)))
    fill_mode_str = derive_execution_semantics(spec.execution).fill_price_mode
    fill_mode = FILL_PRICE_MODE_MAP.get(fill_mode_str)
    if fill_mode is None:
        valid = ", ".join(sorted(FILL_PRICE_MODE_MAP.keys()))
        raise ValueError(f"Unknown fill_price_mode '{fill_mode_str}'. Valid: {valid}")
    broker = SimBroker(
        fee_model=fee_model,
        slippage_model=slippage_model,
        fill_price_mode=fill_mode,
        market_calendar=runtime_calendar,
    )

    # Determine date range
    train = spec.validation.train_period
    test = spec.validation.test_period
    start = train[0] if train else (test[0] if test else "2018-01-01")
    end = test[1] if test else (train[1] if train else "2025-12-31")

    # Run engine
    engine = Engine()
    result = engine.run(
        strategy=strategy,
        market=market,
        broker=broker,
        start=start,
        end=end,
        initial_cash=spec.execution.initial_cash,
        universe=universe,
        lot_size=_effective_lot_size(spec),
        cash_annual_return=spec.execution.cash_annual_return,
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
    _write_artifacts(spec, result, run_dir, engine, effective_data_dir=str(data_path), strategy=strategy)

    return result, run_dir


def _write_artifacts(
    spec: StrategySpec,
    result: RunResult,
    run_dir: Path,
    engine: Engine,
    effective_data_dir: str | None = None,
    strategy: Strategy | None = None,
) -> None:
    """Write all standardized backtest artifacts to run_dir."""
    run_id = run_dir.name

    # strategy_spec.yaml
    spec_path = run_dir / "strategy_spec.yaml"
    spec_path.write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False), encoding="utf-8"
    )
    serialized_spec = StrategySpec.from_yaml(spec_path)
    serialized_spec_hash = serialized_spec.compute_hash()

    # spec_hash.txt
    (run_dir / "spec_hash.txt").write_text(serialized_spec_hash + "\n")

    # environment.json
    env = {
        "open_xquant_version": _get_version(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "run_timestamp": datetime.now(UTC).isoformat(),
        "spec_hash": serialized_spec_hash,
    }
    if effective_data_dir:
        env["data_dir"] = effective_data_dir
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    # data_manifest.json
    symbols = spec.universe.symbols
    manifest_start = spec.data.min_start_date or ""
    if not manifest_start and spec.validation.train_period:
        manifest_start = str(spec.validation.train_period[0])
    if not manifest_start and spec.validation.test_period:
        manifest_start = str(spec.validation.test_period[0])
    manifest_end = str(spec.validation.test_period[1]) if spec.validation.test_period else ""
    if not manifest_end and spec.validation.train_period:
        manifest_end = str(spec.validation.train_period[1])
    analysis_start = ""
    if spec.validation.train_period:
        analysis_start = spec.validation.train_period[0]
    elif spec.validation.test_period:
        analysis_start = spec.validation.test_period[0]
    warmup_policy = "none_declared"
    if spec.data.min_start_date:
        warmup_policy = "preload_from_min_start_date"
    missing_ratio = _compute_missing_ratio(
        result.mktdata,
        spec.data.required_columns,
        spec.market.calendar,
        start=manifest_start,
        end=manifest_end,
        symbols=symbols,
    )
    symbol_ranges = _compute_symbol_ranges(result.mktdata) if result.mktdata else {}
    data_fingerprints = _compute_data_fingerprints(
        result.mktdata,
        spec.data.required_columns,
        calendar=spec.market.calendar,
        start=manifest_start,
        end=manifest_end,
        symbols=symbols,
    ) if result.mktdata else {}
    manifest = {
        "schema_version": 1,
        "provider": spec.data.provider,
        "symbols": symbols,
        "columns": spec.data.required_columns,
        "calendar": spec.market.calendar,
        "price_adjustment": spec.data.price_adjustment,
        "data_dir": effective_data_dir or spec.data.data_dir,
        "spec_data_dir": spec.data.data_dir,
        "effective_data_dir": effective_data_dir or spec.data.data_dir,
        "min_start_date": spec.data.min_start_date,
        "analysis_start": analysis_start,
        "warmup_policy": warmup_policy,
        "start": manifest_start,
        "end": manifest_end,
        "missing_ratio": missing_ratio,
        "symbol_ranges": symbol_ranges,
        "data_fingerprints": data_fingerprints,
    }
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # execution_assumptions.json
    execution_assumptions = _build_execution_assumptions(spec)
    (run_dir / "execution_assumptions.json").write_text(json.dumps(execution_assumptions, indent=2) + "\n", encoding="utf-8")

    # compiled_plan.json
    compiled_strategy = strategy or compile_strategy(serialized_spec)
    compiled_plan = _build_compiled_plan(serialized_spec, compiled_strategy, effective_data_dir=effective_data_dir)
    (run_dir / "compiled_plan.json").write_text(
        json.dumps(compiled_plan, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    compiled_plan_hash = _hash_json_file(run_dir / "compiled_plan.json")

    # strategy.py
    (run_dir / "strategy.py").write_text(
        _build_strategy_py_artifact(serialized_spec, compiled_plan, serialized_spec_hash, compiled_plan_hash),
        encoding="utf-8",
    )

    # metrics.json
    metrics = _sanitize_json(_build_metrics(spec, result, run_id))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")

    # equity_curve.csv
    equity_rows = [{"date": str(d), "value": v} for d, v in result.equity_curve]
    pd.DataFrame(equity_rows).to_csv(run_dir / "equity_curve.csv", index=False)

    benchmark_curve_path = _write_benchmark_curve_artifact(spec, result, run_dir)

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

    # orders.csv
    order_rows = _build_order_rows(result)
    order_columns = [
        "id",
        "status",
        "status_reason",
        "symbol",
        "side",
        "shares",
        "order_type",
        "limit_price",
        "stop_price",
        "trail_pct",
        "created_at",
        "due_at",
        "filled_at",
        "filled_price",
        "filled_shares",
    ]
    pd.DataFrame(order_rows, columns=order_columns).to_csv(run_dir / "orders.csv", index=False)

    # target_weights.csv
    target_weight_rows = _build_target_weight_rows(result)
    target_weight_columns = [
        "date",
        "symbol",
        "raw_target_weight",
        "adjusted_target_weight",
        "reason",
    ]
    pd.DataFrame(target_weight_rows, columns=target_weight_columns).to_csv(run_dir / "target_weights.csv", index=False)

    artifact_hashes = {
        "schema_version": 5,
        "strategy_spec.yaml": _hash_file(run_dir / "strategy_spec.yaml"),
        "environment.json": _hash_json_file(run_dir / "environment.json", exclude_keys={"run_timestamp"}),
        "data_manifest.json": _hash_json_file(run_dir / "data_manifest.json"),
        "execution_assumptions.json": _hash_json_file(run_dir / "execution_assumptions.json"),
        "compiled_plan.json": compiled_plan_hash,
        "strategy.py": _hash_file(run_dir / "strategy.py"),
        "equity_curve.csv": _hash_file(run_dir / "equity_curve.csv"),
        "trades.csv": _hash_file(run_dir / "trades.csv"),
        "positions.csv": _hash_file(run_dir / "positions.csv"),
        "orders.csv": _hash_file(run_dir / "orders.csv"),
        "target_weights.csv": _hash_file(run_dir / "target_weights.csv"),
        "metrics.json": _hash_json_file(run_dir / "metrics.json", exclude_keys={"run_id"}),
    }
    if benchmark_curve_path is not None:
        artifact_hashes["benchmark_curve.csv"] = _hash_file(benchmark_curve_path)
    (run_dir / "artifact_hashes.json").write_text(json.dumps(artifact_hashes, indent=2) + "\n", encoding="utf-8")
    _append_run_digest(run_dir, _hash_json_file(run_dir / "artifact_hashes.json"))

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


def _write_benchmark_curve_artifact(spec: StrategySpec, result: RunResult, run_dir: Path) -> Path | None:
    preferred_symbols = [item for item in spec.benchmark.symbols if item in result.benchmark_prices]
    fallback_symbols = [item for item in result.benchmark_prices if item not in preferred_symbols]
    for symbol in [*preferred_symbols, *fallback_symbols]:
        series = result.benchmark_prices[symbol]
        if series.empty:
            continue
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            continue

        rows = [{"date": str(date), "value": float(value)} for date, value in values.sort_index().items()]
        path = run_dir / "benchmark_curve.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path
    return None


def _build_execution_assumptions(spec: StrategySpec) -> dict[str, Any]:
    semantics = derive_execution_semantics(spec.execution)
    rebalance_interval, rebalance_source = _effective_rebalance(spec)
    return {
        "schema_version": 1,
        "calendar": spec.market.calendar,
        "runtime_calendar": normalize_exchange_calendar(spec.market.calendar),
        "order_timing": semantics.order_timing,
        "price_bar": semantics.price_bar,
        "price_type": semantics.price_type,
        "fill_price_mode": semantics.fill_price_mode,
        "compatibility_source": semantics.compatibility_source,
        "cash_annual_return": spec.execution.cash_annual_return,
        "lot_size": spec.execution.lot_size,
        "lot_size_config": {
            "default": spec.execution.lot_size_config.default,
            "by_symbol": dict(spec.execution.lot_size_config.by_symbol),
        },
        "rebalance": {
            "frequency": spec.execution.rebalance.frequency,
            "interval_days": rebalance_interval,
            "source": rebalance_source,
        },
    }


def _build_compiled_plan(
    spec: StrategySpec,
    strategy: Strategy | None = None,
    *,
    effective_data_dir: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic trace of how the spec maps to runtime objects."""
    strategy = strategy or compile_strategy(spec)
    universe = compile_universe(spec)
    semantics = derive_execution_semantics(spec.execution)
    rebalance_interval, rebalance_source = _effective_rebalance(spec)
    assumptions = metric_assumptions(spec.metrics)
    signal_types = {name: rule.type for name, rule in sorted(spec.signal.rules.items())}
    terminal_signals: list[str] = []
    portfolio_runtime_type = spec.portfolio.type
    if spec.portfolio.type == "EqualWeight" and spec.signal.rules:
        terminal_signals = _terminal_signal_names(spec)
        signal_types = {name: _effective_signal_type(spec, name) for name in sorted(spec.signal.rules)}
        portfolio_runtime_type = "SignalFilteredEqualWeight"

    return {
        "schema_version": 1,
        "compilation_mode": "direct_runtime",
        "spec_hash": spec.compute_hash(),
        "strategy": {
            "strategy_id": spec.strategy_id,
            "runtime_name": strategy.name,
            "display_name": spec.name,
            "hypothesis": strategy.hypothesis,
        },
        "market": {
            "asset_class": spec.market.asset_class,
            "region": spec.market.region,
            "currency": spec.market.currency,
            "calendar": spec.market.calendar,
            "runtime_calendar": normalize_exchange_calendar(spec.market.calendar),
        },
        "universe": {
            "type": spec.universe.type,
            "runtime_class": _class_ref(type(universe)),
            "symbols": list(getattr(universe, "symbols", spec.universe.symbols)),
            "point_in_time": spec.universe.point_in_time,
            "survivorship_bias_policy": spec.universe.survivorship_bias_policy,
        },
        "data": {
            "provider": spec.data.provider,
            "data_dir": effective_data_dir or spec.data.data_dir,
            "spec_data_dir": spec.data.data_dir,
            "effective_data_dir": effective_data_dir or spec.data.data_dir,
            "price_adjustment": spec.data.price_adjustment,
            "required_columns": list(spec.data.required_columns),
            "min_start_date": spec.data.min_start_date,
        },
        "signals": {
            "signal_time": spec.signal.signal_time,
            "indicators": {
                name: {
                    "type": definition.type,
                    "class": _class_ref(_resolve_indicator(definition.type)),
                    "params": dict(definition.params),
                }
                for name, definition in sorted(spec.signal.indicators.items())
            },
            "rules": {
                name: {
                    "type": definition.type,
                    "effective_type": signal_types.get(name, definition.type),
                    "class": _class_ref(type(strategy.signals[name][0])),
                    "params": dict(definition.params),
                    "output_domain": list(definition.output_domain),
                }
                for name, definition in sorted(spec.signal.rules.items())
            },
            "terminal_signals": terminal_signals,
            "event_signal_types": sorted(_EVENT_SIGNAL_TYPES),
        },
        "portfolio": {
            "type": spec.portfolio.type,
            "runtime_type": portfolio_runtime_type,
            "class": _class_ref(type(strategy.portfolio)),
            "params": dict(spec.portfolio.params),
            "rules": {
                name: {
                    "type": definition.type,
                    "params": dict(definition.params),
                }
                for name, definition in sorted(spec.portfolio.rules.items())
            },
        },
        "execution": {
            "trade_time": spec.execution.trade_time,
            "order_timing": semantics.order_timing,
            "price_bar": semantics.price_bar,
            "price_type": semantics.price_type,
            "fill_price_mode": semantics.fill_price_mode,
            "compatibility_source": semantics.compatibility_source,
            "initial_cash": spec.execution.initial_cash,
            "cash_annual_return": spec.execution.cash_annual_return,
            "lot_size": spec.execution.lot_size,
            "effective_lot_size": _effective_lot_size(spec),
            "lot_size_config": {
                "default": spec.execution.lot_size_config.default,
                "by_symbol": dict(spec.execution.lot_size_config.by_symbol),
            },
            "rebalance": {
                "frequency": spec.execution.rebalance.frequency,
                "interval_days": rebalance_interval,
                "source": rebalance_source,
            },
        },
        "cost": {
            "fee_model": "PercentageFee",
            "fee_rate": spec.cost.fee_rate,
            "fee_min": spec.cost.fee_min,
            "slippage_model": "PercentageSlippage",
            "slippage_rate": spec.cost.slippage_rate,
        },
        "runtime_rules": _build_compiled_runtime_rules(spec),
        "benchmark": {
            "symbols": list(strategy.benchmarks),
        },
        "validation": {
            "train_period": list(spec.validation.train_period),
            "test_period": list(spec.validation.test_period),
            "required_oos": spec.validation.required_oos,
        },
        "metrics": {
            "profile": spec.metrics.profile,
            "risk_free_rate": assumptions["risk_free_rate"],
            "return_type": assumptions["return_type"],
            "annualization_days": assumptions["annualization_days"],
            "calmar_denominator": assumptions["calmar_denominator"],
            "evaluation_window": assumptions["evaluation_window"],
        },
    }


def compile_plan(spec: StrategySpec, *, effective_data_dir: str | None = None) -> dict[str, Any]:
    """Return the deterministic compiled runtime plan for a strategy spec."""
    from oxq.spec.validator import validate

    validation = validate(spec)
    if validation.status == "fail":
        messages = "; ".join(f"{error['check']}: {error['message']}" for error in validation.errors)
        raise ValueError(f"spec cannot be compiled: {messages}")
    return _build_compiled_plan(spec, effective_data_dir=effective_data_dir)


def _build_strategy_py_artifact(
    spec: StrategySpec,
    compiled_plan: dict[str, Any],
    spec_hash: str,
    compiled_plan_hash: str,
) -> str:
    spec_dict = spec.to_dict()
    indicator_names = list(spec_dict.get("signal", {}).get("indicators", {}).keys())
    signal_names = list(spec_dict.get("signal", {}).get("rules", {}).keys())
    terminal_signals = compiled_plan.get("signals", {}).get("terminal_signals", [])
    runtime_rules = compiled_plan.get("runtime_rules", [])
    flow = [
        {
            "phase": "universe",
            "description": "Resolve the tradable pool for this run. The universe is a run input, not the strategy body.",
            "artifact": "UNIVERSE",
        },
        {
            "phase": "indicator",
            "description": "Load market data, then calculate indicators declared under SIGNAL['indicators'].",
            "indicator_names": indicator_names,
        },
        {
            "phase": "signal",
            "description": "Evaluate signal rules from indicator outputs and produce terminal signal columns.",
            "signal_names": signal_names,
            "terminal_signals": terminal_signals,
        },
        {
            "phase": "portfolio",
            "description": "Convert terminal signals or scores into target weights using the configured portfolio optimizer.",
            "portfolio_type": spec_dict.get("portfolio", {}).get("type", ""),
        },
        {
            "phase": "rule",
            "description": "Apply runtime rules such as rebalance frequency or risk constraints before orders are generated.",
            "runtime_rules": runtime_rules,
        },
        {
            "phase": "trade_simulation",
            "description": "Simulate order generation, fills, costs, cash, positions, equity, and result artifacts.",
            "execution": compiled_plan.get("execution", {}),
            "cost": compiled_plan.get("cost", {}),
        },
    ]
    sections = {
        "RESEARCH": spec_dict.get("research", {}),
        "MARKET": spec_dict.get("market", {}),
        "UNIVERSE": spec_dict.get("universe", {}),
        "DATA": spec_dict.get("data", {}),
        "SIGNAL": spec_dict.get("signal", {}),
        "PORTFOLIO": spec_dict.get("portfolio", {}),
        "EXECUTION": spec_dict.get("execution", {}),
        "COST": spec_dict.get("cost", {}),
        "BENCHMARK": spec_dict.get("benchmark", {}),
        "VALIDATION": spec_dict.get("validation", {}),
        "METRICS": spec_dict.get("metrics", {}),
        "ROBUSTNESS": spec_dict.get("robustness", {}),
        "DECISION_POLICY": spec_dict.get("decision_policy", {}),
    }
    lines = [
        '"""',
        "Generated open-xquant strategy artifact.",
        "",
        "Canonical source: strategy_spec.yaml",
        "Machine audit trace: compiled_plan.json",
        "",
        "This file is a human-readable Python projection of the strategy flow.",
        "It is not the backtest entrypoint. The formal runtime still reads",
        "strategy_spec.yaml and records execution semantics in compiled_plan.json.",
        "",
        "Read the functions below from top to bottom:",
        "1. define_universe()",
        "2. define_indicators()",
        "3. define_signals()",
        "4. define_portfolio()",
        "5. define_rules()",
        "6. define_execution()",
        "7. simulate_trading_flow()",
        "",
        "Edit strategy_spec.yaml, not this generated review artifact.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from copy import deepcopy",
        "from pathlib import Path",
        "",
    ]
    lines.extend(
        [
            "def define_universe() -> dict:",
            '    """Return the run universe used to evaluate this strategy."""',
            "    # The universe is deliberately modeled as a run input.",
            "    # Running the same signal, portfolio, and rule logic on another",
            "    # universe produces another portfolio without changing the strategy.",
            "    universe = deepcopy(UNIVERSE)",
            "    universe_symbols = universe.get('symbols', [])",
            "    universe['review_note'] = (",
            "        f\"This run evaluates {len(universe_symbols)} symbols; \"",
            "        'change the run universe to test the same strategy elsewhere.'",
            "    )",
            "    return universe",
            "",
            "",
            "def define_indicators() -> dict:",
            '    """Return indicator definitions used by the signal layer."""',
            "    # Indicators transform market data into numeric series.",
            "    # They should be causal: every timestamp may only use information",
            "    # available at or before that timestamp.",
            "    signal = deepcopy(SIGNAL)",
            "    return signal.get('indicators', {})",
            "",
            "",
            "def define_signals() -> dict:",
            '    """Return signal rules and terminal signal outputs."""',
            "    # Signal rules consume indicator outputs and create boolean,",
            "    # categorical, or score-like columns that the portfolio layer reads.",
            "    signal = deepcopy(SIGNAL)",
            "    return {",
            "        'rules': signal.get('rules', {}),",
            "        'terminal_signals': deepcopy(COMPILED_PLAN.get('signals', {}).get('terminal_signals', [])),",
            "    }",
            "",
            "",
            "def define_portfolio() -> dict:",
            '    """Return how signals or scores become target weights."""',
            "    # The portfolio layer is the allocation policy. Examples include",
            "    # equal weight, TopN ranking, signal-filtered allocation, or custom",
            "    # optimizers. It does not choose the run universe by itself.",
            "    return {",
            "        'spec': deepcopy(PORTFOLIO),",
            "        'compiled': deepcopy(COMPILED_PLAN.get('portfolio', {})),",
            "    }",
            "",
            "",
            "def define_rules() -> list[dict]:",
            '    """Return runtime rules applied around portfolio construction."""',
            "    # Rules constrain when or how trades happen. A rebalance rule, for",
            "    # example, must be visible here and in COMPILED_PLAN['runtime_rules']",
            "    # before the run can honestly claim it was executed.",
            "    return deepcopy(COMPILED_PLAN.get('runtime_rules', []))",
            "",
            "",
            "def define_execution() -> dict:",
            '    """Return execution, cost, benchmark, and validation assumptions."""',
            "    # Execution settings describe how target weights become simulated",
            "    # orders and fills. These assumptions materially affect performance.",
            "    return {",
            "        'execution': deepcopy(EXECUTION),",
            "        'compiled_execution': deepcopy(COMPILED_PLAN.get('execution', {})),",
            "        'cost': deepcopy(COST),",
            "        'benchmark': deepcopy(BENCHMARK),",
            "        'validation': deepcopy(VALIDATION),",
            "        'metrics': deepcopy(METRICS),",
            "    }",
            "",
            "",
            "def simulate_trading_flow() -> list[dict]:",
            '    """Return the review sequence for the simulated trading process."""',
            "    # This is a process map for human review, not a replacement for",
            "    # oxq backtest run. The formal engine still uses strategy_spec.yaml",
            "    # plus compiled_plan.json as its audited runtime contract.",
            "    return deepcopy(STRATEGY_FLOW)",
            "",
            "",
            "def describe() -> dict:",
            '    """Return all review material embedded in this generated file."""',
            "    return {",
            '        "strategy_spec_hash": STRATEGY_SPEC_HASH,',
            '        "compiled_plan_hash": COMPILED_PLAN_HASH,',
            '        "strategy_spec": deepcopy(STRATEGY_SPEC),',
            '        "compiled_plan": deepcopy(COMPILED_PLAN),',
            '        "strategy_flow": simulate_trading_flow(),',
            '        "universe": define_universe(),',
            '        "indicators": define_indicators(),',
            '        "signals": define_signals(),',
            '        "portfolio": define_portfolio(),',
            '        "rules": define_rules(),',
            '        "execution": define_execution(),',
            "    }",
            "",
            "",
            "def build_strategy():",
            '    """Build the runtime Strategy object for compatibility checks.',
            "",
            "    The human review path above intentionally shows the process without",
            "    making this file the formal backtest entrypoint. This helper remains",
            "    for import-based smoke tests and old integrations that expect a",
            "    Strategy object.",
            '    """',
            "    from oxq.core.component_manifest import load_component_manifests_from_run, scoped_component_registries",
            "    from oxq.spec.compiler import compile_strategy",
            "    from oxq.spec.schema import StrategySpec",
            "",
            "    run_dir = Path(__file__).parent",
            "    with scoped_component_registries():",
            '        if (run_dir / "component_manifests.json").exists():',
            "            load_component_manifests_from_run(run_dir)",
            '        spec_path = run_dir / "strategy_spec.yaml"',
            "        return compile_strategy(StrategySpec.from_yaml(spec_path))",
            "",
            "",
            "# -----------------------------------------------------------------------------",
            "# Audit data appendix",
            "# -----------------------------------------------------------------------------",
            "# The constants below are intentionally kept as top-level assignments so the",
            "# reproducibility audit can parse them without executing this file. They are",
            "# placed after the process functions so human readers see strategy flow first.",
            "",
            f"STRATEGY_SPEC_HASH = {spec_hash!r}",
            f"COMPILED_PLAN_HASH = {compiled_plan_hash!r}",
            "",
            f"STRATEGY_SPEC = {_format_python_literal(spec_dict)}",
            "",
            f"COMPILED_PLAN = {_format_python_literal(compiled_plan)}",
            "",
            f"STRATEGY_FLOW = {_format_python_literal(flow)}",
            "",
        ]
    )
    for name, value in sections.items():
        lines.extend([f"{name} = {_format_python_literal(value)}", ""])
    return "\n".join(lines)


def _format_python_literal(value: Any) -> str:
    payload = json.loads(json.dumps(_sanitize_json(value), default=str, allow_nan=False))
    return pprint.pformat(payload, sort_dicts=False, width=100)


def _build_compiled_runtime_rules(spec: StrategySpec) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    rebalance_interval, rebalance_source = _effective_rebalance(spec)
    if rebalance_interval > 1:
        rules.append(
            {
                "type": "RebalanceFrequencyRule",
                "class": f"{RebalanceFrequencyRule.__module__}.{RebalanceFrequencyRule.__qualname__}",
                "params": {"interval_days": rebalance_interval},
                "source": rebalance_source,
            }
        )
    for signal_name, signal_def in sorted(spec.signal.rules.items()):
        if signal_def.type != "Crossover":
            continue
        fast = signal_def.params.get("fast", "")
        slow = signal_def.params.get("slow", "")
        if fast and slow:
            rules.append(
                {
                    "type": "ExitRule",
                    "class": "oxq.rules.exit.ExitRule",
                    "params": {"fast": fast, "slow": slow},
                    "source": f"signal.rules.{signal_name}",
                }
            )
    return rules


def _class_ref(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _to_timestamp(ts_val: str | object, tz: object | None = None) -> pd.Timestamp:
    """Convert a fill timestamp to pd.Timestamp, handling existing timezone."""
    ts = pd.Timestamp(ts_val)
    if ts.tz is None and tz is not None:
        ts = ts.tz_localize(tz)
    return ts


def _compute_missing_ratio(
    mktdata: dict[str, pd.DataFrame],
    columns: list[str] | None = None,
    calendar: str = "XNYS",
    start: str = "",
    end: str = "",
    symbols: list[str] | None = None,
) -> float:
    """Compute missing data after aligning symbols to an expected business calendar."""
    symbols_to_check = sorted(set(symbols or []) | set(mktdata.keys()))
    indexes = [pd.DatetimeIndex(df.index) for df in mktdata.values() if not df.empty]
    if not indexes:
        expected_index = _expected_market_index_from_range(calendar, start, end)
        if expected_index.empty:
            return 1.0 if symbols_to_check else 0.0
    else:
        expected_index = _expected_market_index(indexes, calendar, start=start, end=end)

    total = 0
    missing = 0
    for symbol in symbols_to_check:
        df = mktdata.get(symbol, pd.DataFrame())
        aligned = _align_frame_to_session_dates(df, expected_index)
        check_columns = columns or list(df.columns) or ["__rows__"]
        for col in check_columns:
            total += len(expected_index)
            if col in aligned.columns:
                missing += int(aligned[col].isna().sum())
            else:
                missing += len(expected_index)
    return missing / total if total > 0 else 0.0


def _expected_market_index_from_range(calendar: str, start: str = "", end: str = "") -> pd.DatetimeIndex:
    if not start or not end:
        return pd.DatetimeIndex([])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    tz = start_ts.tz
    if tz is None and end_ts.tz is not None:
        end_ts = end_ts.tz_localize(None)
    elif tz is not None:
        end_ts = end_ts.tz_localize(tz) if end_ts.tz is None else end_ts.tz_convert(tz)
    sessions = _exchange_calendar_sessions(start_ts, end_ts, calendar)
    if sessions is not None:
        if tz is not None:
            return sessions.tz_localize(tz) if sessions.tz is None else sessions.tz_convert(tz)
        return sessions.tz_localize(None) if sessions.tz is not None else sessions
    raise ValueError(f"exchange_calendars is required for market calendar '{calendar}'")


def _expected_market_index(
    indexes: list[pd.DatetimeIndex],
    calendar: str,
    start: str = "",
    end: str = "",
) -> pd.DatetimeIndex:
    observed_start = min(index.min() for index in indexes)
    observed_end = max(index.max() for index in indexes)
    tz = observed_start.tz
    start_ts = _requested_timestamp(start, observed_start, tz)
    end_ts = _requested_timestamp(end, observed_end, tz)
    sessions = _exchange_calendar_sessions(start_ts, end_ts, calendar)
    if sessions is not None:
        if tz is not None:
            return sessions.tz_localize(tz) if sessions.tz is None else sessions.tz_convert(tz)
        return sessions.tz_localize(None) if sessions.tz is not None else sessions
    raise ValueError(f"exchange_calendars is required for market calendar '{calendar}'")


def _requested_timestamp(value: str, fallback: pd.Timestamp, tz: object | None) -> pd.Timestamp:
    if not value:
        return fallback
    ts = pd.Timestamp(value)
    if tz is None:
        return ts.tz_localize(None) if ts.tz is not None else ts
    return ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)


def _exchange_calendar_sessions(start: pd.Timestamp, end: pd.Timestamp, calendar: str) -> pd.DatetimeIndex | None:
    try:
        import exchange_calendars as xcals
    except ImportError:
        return None
    try:
        cal = xcals.get_calendar(normalize_exchange_calendar(calendar))
        sessions = cal.sessions_in_range(start.date(), end.date())
    except Exception:
        return None
    return pd.DatetimeIndex(sessions)


def _compute_symbol_ranges(mktdata: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    """Return actual per-symbol date ranges present in loaded market data."""
    ranges: dict[str, dict[str, str]] = {}
    for symbol, df in mktdata.items():
        if df.empty:
            ranges[symbol] = {"start": "", "end": ""}
            continue
        index = pd.DatetimeIndex(df.index)
        ranges[symbol] = {
            "start": str(index.min().date()),
            "end": str(index.max().date()),
        }
    return ranges


def _compute_data_fingerprints(
    mktdata: dict[str, pd.DataFrame],
    columns: list[str] | None = None,
    calendar: str = "XNYS",
    start: str = "",
    end: str = "",
    symbols: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return deterministic per-symbol fingerprints of loaded market data."""
    symbols_to_check = sorted(set(symbols or []) | set(mktdata.keys()))
    expected_index = _expected_market_index_from_range(calendar, start, end)
    return {
        symbol: _fingerprint_dataframe(_reindex_for_fingerprint(mktdata.get(symbol, pd.DataFrame()), expected_index), columns)
        for symbol in symbols_to_check
    }


def _reindex_for_fingerprint(df: pd.DataFrame, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    return _select_frame_for_session_fingerprint(df, expected_index)


def _align_frame_to_session_dates(df: pd.DataFrame, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty:
        return df.reindex(expected_index)
    source = df.copy()
    session_dates = pd.Index([pd.Timestamp(idx).date() for idx in source.index])
    if session_dates.has_duplicates:
        raise ValueError("market data has multiple rows for the same market session")
    expected_dates = pd.Index([pd.Timestamp(idx).date() for idx in expected_index])
    source.index = session_dates
    aligned = source.reindex(expected_dates)
    aligned.index = expected_index
    return aligned


def _select_frame_for_session_fingerprint(df: pd.DataFrame, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty:
        return df.reindex(expected_index)
    source = df.copy()
    source_index = pd.DatetimeIndex(source.index)
    missing_index = _expected_index_with_source_tz(expected_index, source_index)
    session_dates = pd.Index([pd.Timestamp(idx).date() for idx in source.index])
    if session_dates.has_duplicates:
        raise ValueError("market data has multiple rows for the same market session")
    source_by_date = dict(zip(session_dates, range(len(source)), strict=True))
    rows: list[pd.Series] = []
    index_values: list[Any] = []
    for expected_ts, missing_ts in zip(expected_index, missing_index, strict=True):
        expected_date = pd.Timestamp(expected_ts).date()
        source_pos = source_by_date.get(expected_date)
        if source_pos is None:
            rows.append(pd.Series(index=source.columns, dtype="object"))
            index_values.append(missing_ts)
            continue
        rows.append(source.iloc[source_pos])
        index_values.append(source.index[source_pos])
    aligned = pd.DataFrame(rows)
    aligned.index = pd.Index(index_values)
    return aligned


def _expected_index_with_source_tz(
    expected_index: pd.DatetimeIndex,
    source_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    if source_index.tz is not None and expected_index.tz is None:
        return expected_index.tz_localize(source_index.tz)
    if source_index.tz is None and expected_index.tz is not None:
        return expected_index.tz_localize(None)
    if source_index.tz is not None and expected_index.tz is not None:
        return expected_index.tz_convert(source_index.tz)
    return expected_index


def _fingerprint_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    if df.empty:
        return {
            "row_count": 0,
            "start": "",
            "end": "",
            "columns": columns or [],
            "content_hash": "sha256:e3b0c44298fc1c14",
        }
    frame = df.sort_index()
    check_columns = columns or list(frame.columns)
    frame = frame.reindex(columns=check_columns)
    records = []
    for idx, row in frame.iterrows():
        record = {"__index__": pd.Timestamp(idx).isoformat()}
        for col in check_columns:
            value = row[col]
            record[col] = None if pd.isna(value) else value
        records.append(record)
    payload = json.dumps({"columns": check_columns, "records": records}, sort_keys=True, default=str)
    index = pd.DatetimeIndex(frame.index)
    return {
        "row_count": int(len(frame)),
        "start": pd.Timestamp(index.min()).isoformat(),
        "end": pd.Timestamp(index.max()).isoformat(),
        "columns": check_columns,
        "content_hash": f"sha256:{hashlib.sha256(payload.encode()).hexdigest()[:16]}",
    }


def _hash_file(path: Path) -> str:
    """Compute a short content hash for an artifact file."""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def _build_order_rows(result: RunResult) -> list[dict[str, Any]]:
    orders = getattr(result, "orders", [])
    if orders:
        return [_serialize_managed_order(managed) for managed in orders]
    return [
        {
            "id": "",
            "status": "filled",
            "status_reason": "",
            "symbol": fill.order.symbol,
            "side": fill.order.side,
            "shares": fill.order.shares,
            "order_type": fill.order.order_type,
            "limit_price": _optional_decimal(fill.order.limit_price),
            "stop_price": _optional_decimal(fill.order.stop_price),
            "trail_pct": fill.order.trail_pct if fill.order.trail_pct is not None else "",
            "created_at": "",
            "due_at": "",
            "filled_at": fill.filled_at,
            "filled_price": float(fill.filled_price),
            "filled_shares": fill.order.shares,
        }
        for fill in result.trades
    ]


def _build_target_weight_rows(result: RunResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_adjusted: dict[str, float] = {}
    for snapshot in result.snapshots:
        raw_weights = snapshot.target_weights
        adjusted_weights = snapshot.adjusted_weights
        rule_reasons = snapshot.rule_reasons
        symbols = sorted(set(raw_weights) | set(adjusted_weights) | set(previous_adjusted))
        current_adjusted: dict[str, float] = {}
        for symbol in symbols:
            raw_weight = float(raw_weights.get(symbol, 0.0))
            adjusted_weight = float(adjusted_weights.get(symbol, 0.0))
            if adjusted_weight != 0.0:
                current_adjusted[symbol] = adjusted_weight
            reason = rule_reasons.get(symbol) or rule_reasons.get("__all__")
            if not reason:
                reason = (
                    "target_changed"
                    if adjusted_weight != previous_adjusted.get(symbol, 0.0)
                    else "target_unchanged"
                )
            rows.append(
                {
                    "date": str(snapshot.date),
                    "symbol": symbol,
                    "raw_target_weight": raw_weight,
                    "adjusted_target_weight": adjusted_weight,
                    "reason": reason,
                }
            )
        previous_adjusted = current_adjusted
    return rows


def _serialize_managed_order(managed: Any) -> dict[str, Any]:
    order = managed.order
    return {
        "id": managed.id,
        "status": managed.status,
        "status_reason": getattr(managed, "status_reason", ""),
        "symbol": order.symbol,
        "side": order.side,
        "shares": order.shares,
        "order_type": order.order_type,
        "limit_price": _optional_decimal(order.limit_price),
        "stop_price": _optional_decimal(order.stop_price),
        "trail_pct": order.trail_pct if order.trail_pct is not None else "",
        "created_at": managed.created_at,
        "due_at": managed.due_at or "",
        "filled_at": managed.filled_at or "",
        "filled_price": _optional_decimal(managed.filled_price),
        "filled_shares": managed.filled_shares if managed.filled_shares is not None else "",
    }


def _optional_decimal(value: Any) -> float | str:
    return float(value) if value is not None else ""


def _hash_json_file(path: Path, exclude_keys: set[str] | None = None) -> str:
    """Compute a short canonical JSON hash, optionally excluding run metadata keys."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and exclude_keys:
        data = {key: value for key, value in data.items() if key not in exclude_keys}
    canonical = json.dumps(data, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _append_run_digest(run_dir: Path, artifact_hashes_hash: str) -> None:
    digest_path = run_dir.parent / "run_digests.jsonl"
    lock_path = digest_path.with_suffix(digest_path.suffix + ".lock")
    entry = {
        "run_id": run_dir.name,
        "artifact_hashes": artifact_hashes_hash,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with _FileLock(lock_path):
        with open(digest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


class _FileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: Any = None

    def __enter__(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a+")
        import fcntl

        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._fh is None:
            return
        import fcntl

        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _validate_strategy_id_for_path(strategy_id: str) -> None:
    """Reject strategy IDs that could escape the requested run directory."""
    if "/" in strategy_id or "\\" in strategy_id or ".." in Path(strategy_id).parts:
        raise ValueError("strategy_id must not contain path separators or '..'")


def _validate_crossover_rule_count(spec: StrategySpec) -> None:
    count = sum(1 for rule in spec.signal.rules.values() if rule.type == "Crossover")
    if count > 1:
        raise ValueError("Multiple Crossover rules are not supported by the spec compiler")


def _terminal_signal_names(spec: StrategySpec) -> list[str]:
    referenced: set[str] = set()
    for rule_def in spec.signal.rules.values():
        if rule_def.type != "Composite":
            continue
        signals = rule_def.params.get("signals", [])
        if isinstance(signals, list):
            referenced.update(signal_name for signal_name in signals if isinstance(signal_name, str))
    terminal = [name for name in spec.signal.rules if name not in referenced]
    if len(terminal) != 1:
        raise ValueError(
            "Exactly one terminal signal rule is required for EqualWeight specs with signal.rules; "
            f"found {terminal or 'none'}"
        )
    return terminal


def _effective_signal_type(spec: StrategySpec, signal_name: str, seen: set[str] | None = None) -> str:
    rule_def = spec.signal.rules[signal_name]
    if rule_def.type != "Composite":
        return rule_def.type
    seen = set(seen or ())
    if signal_name in seen:
        return rule_def.type
    seen.add(signal_name)
    signals = rule_def.params.get("signals", [])
    if not isinstance(signals, list):
        return rule_def.type
    child_types: list[str] = []
    for child_name in signals:
        if not isinstance(child_name, str) or child_name not in spec.signal.rules:
            continue
        child_types.append(_effective_signal_type(spec, child_name, seen))
    has_event = any(child_type in _EVENT_SIGNAL_TYPES for child_type in child_types)
    has_level = any(child_type not in _EVENT_SIGNAL_TYPES for child_type in child_types)
    if rule_def.params.get("logic", "and") == "or" and has_event and has_level:
        raise ValueError(
            f"Composite signal '{signal_name}' with logic='or' cannot mix event and level signals"
        )
    if has_event:
        return "Crossover"
    return rule_def.type


def _get_version() -> str:
    """Get open-xquant version from package metadata."""
    try:
        from importlib.metadata import version

        return version("open-xquant")
    except Exception:
        return "0.1.0"


def _build_metrics(spec: StrategySpec, result: RunResult, run_id: str) -> dict[str, Any]:
    """Build metrics dict including OOS-only metrics when test_period is defined."""
    test = spec.validation.test_period
    metric_result = result
    metric_diagnostics: list[str] = []
    activity_trades = result.trades
    if spec.metrics.evaluation_window == "oos" and test and len(test) >= 2:
        tz = getattr(pd.Timestamp(result.equity_curve[0][0]), "tz", None) if result.equity_curve else None
        test_start = pd.Timestamp(test[0], tz=tz)
        test_end = pd.Timestamp(test[1], tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        activity_trades = _filter_trades_by_window(result.trades, start=test_start, end=test_end, tz=tz)
        metric_curve = _window_equity_curve(
            result.equity_curve,
            start=test_start,
            end=test_end,
            include_previous_baseline=True,
        )
        if len(metric_curve) >= 2:
            metric_result = _run_result_for_equity_curve(result, metric_curve)
        else:
            metric_diagnostics.append("evaluation_window=oos unavailable: OOS equity curve has fewer than 2 points")

    base = compute_profile_metrics(metric_result, spec.metrics, run_id=run_id)
    if metric_diagnostics:
        _clear_top_level_profile_metrics(base)
        base["metric_diagnostics"] = metric_diagnostics
    base.update({
        "strategy_id": spec.strategy_id,
        "turnover": result.turnover() if hasattr(result, "turnover") else 0.0,
        "trade_count": len(activity_trades),
        "cost_paid": float(sum(float(f.fee) for f in activity_trades)),
        "slippage_paid": None,  # Not measurable without raw-vs-slipped fill price tracking
    })

    # Compute OOS-only metrics when test_period is defined
    train = spec.validation.train_period
    if train and len(train) >= 2 and len(result.equity_curve) > 1:
        first_dt = result.equity_curve[0][0]
        tz = getattr(pd.Timestamp(first_dt), "tz", None)
        train_start = pd.Timestamp(train[0], tz=tz)
        train_end = pd.Timestamp(train[1], tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        is_curve = _window_equity_curve(result.equity_curve, start=train_start, end=train_end, include_previous_baseline=True)
        if len(is_curve) >= 2:
            is_metrics = compute_equity_curve_metrics(is_curve, spec.metrics)
            base["is_sharpe_ratio"] = is_metrics["sharpe_ratio"]
            base["is_total_return"] = is_metrics["total_return"]
            base["is_max_drawdown"] = is_metrics["max_drawdown"]
            base["is_annualized_return"] = is_metrics["annualized_return"]
            base["is_annualized_volatility"] = is_metrics["annualized_volatility"]
            base["is_calmar_ratio"] = is_metrics["calmar_ratio"]

    if test and len(test) >= 2 and len(result.equity_curve) > 1:
        # Use first equity curve date's tz to match timezone-aware timestamps
        first_dt = result.equity_curve[0][0]
        tz = getattr(pd.Timestamp(first_dt), "tz", None)
        test_start = pd.Timestamp(test[0], tz=tz)
        test_end = pd.Timestamp(test[1], tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        oos_curve = _window_equity_curve(
            result.equity_curve,
            start=test_start,
            end=test_end,
            include_previous_baseline=True,
        )
        if len(oos_curve) >= 2:
            oos_metrics = compute_equity_curve_metrics(oos_curve, spec.metrics)
            base["oos_sharpe_ratio"] = oos_metrics["sharpe_ratio"]
            base["oos_total_return"] = oos_metrics["total_return"]
            base["oos_max_drawdown"] = oos_metrics["max_drawdown"]
            base["oos_annualized_return"] = oos_metrics["annualized_return"]
            base["oos_annualized_volatility"] = oos_metrics["annualized_volatility"]
            base["oos_calmar_ratio"] = oos_metrics["calmar_ratio"]
            oos_trades = _filter_trades_by_window(result.trades, start=test_start, end=test_end, tz=tz)
            base["oos_trade_count"] = len(oos_trades)

    return base


def _filter_trades_by_window(
    trades: list[Fill],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz: object | None,
) -> list[Fill]:
    return [fill for fill in trades if start <= _to_timestamp(fill.filled_at, tz=tz) <= end]


def _clear_top_level_profile_metrics(metrics: dict[str, Any]) -> None:
    for key in (
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
    ):
        metrics[key] = None


def _window_equity_curve(
    equity_curve: list[tuple[object, float]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    include_previous_baseline: bool = False,
) -> list[tuple[pd.Timestamp, float]]:
    equity_points = [(pd.Timestamp(d), v) for d, v in equity_curve]
    window = [(d, v) for d, v in equity_points if d >= start and (end is None or d <= end)]
    if include_previous_baseline:
        previous = [(d, v) for d, v in equity_points if d < start]
        if previous:
            window.insert(0, (start - pd.Timedelta(nanoseconds=1), previous[-1][1]))
    return window


def _run_result_for_equity_curve(result: RunResult, equity_curve: list[tuple[object, float]]) -> RunResult:
    return RunResult(
        portfolio=result.portfolio,
        trades=result.trades,
        equity_curve=equity_curve,
        mktdata=result.mktdata,
        benchmark_prices=result.benchmark_prices,
        snapshots=result.snapshots,
        orders=result.orders,
    )
