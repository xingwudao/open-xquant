"""Observe tools — strategy monitoring, market state detection, experiment tracking."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from oxq.observe.audit import AuditRecord
from oxq.observe.detector import MarketStateDetector
from oxq.observe.experiment import ExperimentLog
from oxq.observe.monitor import StrategyMonitor
from oxq.observe.tracer import DefaultTracer
from oxq.tools import session
from oxq.tools.registry import registry


@registry.tool(
    name="observe_monitor_create",
    description="Create a StrategyMonitor for a run result to check strategy health. "
    "Returns monitor_id and initial summary.",
)
def observe_monitor_create(
    run_id: str,
    benchmark: str | None = None,
    roll_window: int = 63,
    min_bad_days: int = 20,
    gap_days: int = 5,
) -> dict[str, Any]:
    """Create a StrategyMonitor from a run result."""
    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        monitor = StrategyMonitor(
            result,
            benchmark=benchmark,
            roll_window=roll_window,
            min_bad_days=min_bad_days,
            gap_days=gap_days,
        )
    except Exception as e:
        return {"error": str(e)}

    monitor_id = f"mon_{run_id}_{int(time.time())}"
    session._monitors[monitor_id] = monitor
    session._save()

    return {
        "monitor_id": monitor_id,
        **monitor.summary(),
    }


@registry.tool(
    name="observe_monitor_summary",
    description="Get health summary and bad periods for a strategy monitor.",
)
def observe_monitor_summary(monitor_id: str) -> dict[str, Any]:
    """Get summary and bad periods from a monitor."""
    monitor = session._monitors.get(monitor_id)
    if monitor is None:
        return {"error": f"Monitor '{monitor_id}' not found"}

    bad_periods = [
        {
            "start": str(bp.start),
            "end": str(bp.end),
            "days": bp.days,
            "avg_sharpe": bp.avg_sharpe,
        }
        for bp in monitor.bad_periods
    ]

    return {
        "monitor_id": monitor_id,
        **monitor.summary(),
        "bad_periods": bad_periods,
    }


@registry.tool(
    name="observe_detect_market_state",
    description="Detect market volatility regimes (high/normal/low) from a run result's market data.",
)
def observe_detect_market_state(
    run_id: str,
    symbols: list[str] | None = None,
    vol_lookback: int = 20,
    high_vol_multiplier: float = 1.3,
    low_vol_multiplier: float = 0.7,
) -> dict[str, Any]:
    """Create a MarketStateDetector from a run result."""
    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        detector = MarketStateDetector(
            result,
            symbols=tuple(symbols) if symbols else None,
            vol_lookback=vol_lookback,
            high_vol_multiplier=high_vol_multiplier,
            low_vol_multiplier=low_vol_multiplier,
        )
    except Exception as e:
        return {"error": str(e)}

    detector_id = f"det_{run_id}_{int(time.time())}"
    session._detectors[detector_id] = detector
    session._save()

    state_counts = detector.states.value_counts().to_dict()

    return {
        "detector_id": detector_id,
        "thresholds": {
            "vol_median": detector.vol_median,
            "high_vol_line": detector.high_vol_line,
            "low_vol_line": detector.low_vol_line,
        },
        "state_counts": {
            "high": int(state_counts.get("high", 0)),
            "normal": int(state_counts.get("normal", 0)),
            "low": int(state_counts.get("low", 0)),
        },
    }


@registry.tool(
    name="observe_performance_by_state",
    description="Evaluate strategy performance grouped by market state (high/normal/low volatility).",
)
def observe_performance_by_state(
    detector_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Get performance breakdown by market state."""
    detector = session._detectors.get(detector_id)
    if detector is None:
        return {"error": f"Detector '{detector_id}' not found"}

    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        perf = detector.performance_by_state(result)
    except Exception as e:
        return {"error": str(e)}

    return {
        "detector_id": detector_id,
        "run_id": run_id,
        "performance": perf,
    }


@registry.tool(
    name="observe_experiment_create",
    description="Create an experiment log for tracking strategy iteration experiments.",
)
def observe_experiment_create(name: str = "") -> dict[str, Any]:
    """Create a new ExperimentLog."""
    log = ExperimentLog(name=name)
    log_id = f"explog_{int(time.time())}"
    session._experiment_logs[log_id] = log
    session._save()

    return {
        "log_id": log_id,
        "name": name,
    }


@registry.tool(
    name="observe_experiment_add",
    description="Add an experiment record manually to an experiment log.",
)
def observe_experiment_add(
    log_id: str,
    name: str,
    observation: str,
    hypothesis: str,
    criteria: dict[str, Any],
    result: dict[str, Any],
    conclusion: str,
    notes: str = "",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Add a manual experiment record to a log."""
    log = session._experiment_logs.get(log_id)
    if log is None:
        return {"error": f"Experiment log '{log_id}' not found"}

    log.add(
        name=name,
        observation=observation,
        hypothesis=hypothesis,
        criteria=criteria,
        result=result,
        conclusion=conclusion,
        notes=notes,
        run_id=run_id,
    )
    session._save()

    return {
        "log_id": log_id,
        "experiment_count": len(log.experiments),
    }


@registry.tool(
    name="observe_experiment_add_from_strategy",
    description="Auto-extract experiment data from a strategy and run result, add to experiment log.",
)
def observe_experiment_add_from_strategy(
    log_id: str,
    strategy: str,
    run_id: str,
    observation: str,
    conclusion: str,
    notes: str = "",
    audit_id: str | None = None,
) -> dict[str, Any]:
    """Add experiment from strategy + run result auto-extraction."""
    log = session._experiment_logs.get(log_id)
    if log is None:
        return {"error": f"Experiment log '{log_id}' not found"}

    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    run_result = session._run_results.get(run_id)
    if run_result is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        log.add_from_strategy(
            strategy=strat,
            result=run_result,
            observation=observation,
            conclusion=conclusion,
            notes=notes,
            run_id=audit_id,
        )
    except Exception as e:
        return {"error": str(e)}

    session._save()

    # Return the last added experiment data
    last_exp = log.experiments[-1]
    return {
        "log_id": log_id,
        "experiment_count": len(log.experiments),
        "experiment": asdict(last_exp),
    }


@registry.tool(
    name="observe_experiment_list",
    description="List all experiments in a log as a formatted table.",
)
def observe_experiment_list(log_id: str) -> dict[str, Any]:
    """List experiments in a log."""
    log = session._experiment_logs.get(log_id)
    if log is None:
        return {"error": f"Experiment log '{log_id}' not found"}

    experiments = [asdict(e) for e in log.experiments]

    # Try to generate markdown table
    try:
        markdown = log.to_markdown()
    except ImportError:
        markdown = ""

    return {
        "log_id": log_id,
        "name": log.name,
        "experiment_count": len(experiments),
        "experiments": experiments,
        "markdown_table": markdown,
    }


@registry.tool(
    name="observe_trace",
    description="View execution trace (TraceSpans) for a run's audit record. "
    "Shows each component's inputs, output summary, timing, and status.",
)
def observe_trace(audit_id: str) -> dict[str, Any]:
    """View execution trace from an audit record."""
    audit = session._audit_records.get(audit_id)
    if audit is None:
        return {"error": f"Audit record '{audit_id}' not found"}

    spans = [
        {
            "component": s.component,
            "inputs": s.inputs,
            "output_summary": s.output_summary,
            "duration_ms": s.duration_ms,
            "status": s.status,
            "error": s.error,
        }
        for s in audit.trace_spans
    ]

    return {
        "audit_id": audit_id,
        "run_id": audit.run_id,
        "strategy_name": audit.strategy_name,
        "span_count": len(spans),
        "spans": spans,
    }


@registry.tool(
    name="observe_audit_log",
    description="Create or view an audit record for a run. "
    "Captures strategy config, execution trace, and layered hashes for determinism verification.",
)
def observe_audit_log(
    run_id: str,
    strategy: str | None = None,
    audit_id: str | None = None,
) -> dict[str, Any]:
    """Create or retrieve an audit record.

    If audit_id is provided, retrieves an existing record.
    If strategy is provided, creates a new audit record from the run.
    """
    # Retrieve existing
    if audit_id is not None:
        audit = session._audit_records.get(audit_id)
        if audit is None:
            return {"error": f"Audit record '{audit_id}' not found"}
        return {
            "audit_id": audit_id,
            "run_id": audit.run_id,
            "strategy_name": audit.strategy_name,
            "strategy_config": audit.strategy_config,
            "start_date": audit.start_date,
            "end_date": audit.end_date,
            "initial_cash": audit.initial_cash,
            "mktdata_hash": audit.mktdata_hash,
            "trades_hash": audit.trades_hash,
            "equity_hash": audit.equity_hash,
            "result_hash": audit.result_hash,
            "span_count": len(audit.trace_spans),
            "created_at": audit.created_at,
        }

    # Create new
    if strategy is None:
        return {"error": "Provide either audit_id to retrieve, or strategy + run_id to create"}

    run_result = session._run_results.get(run_id)
    if run_result is None:
        return {"error": f"Run '{run_id}' not found"}

    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    # Build a tracer retroactively (no live trace data — record config only)
    tracer = DefaultTracer()
    strategy_config = {
        "signals": {k: v[1] for k, v in strat.signals.items()},
        "portfolio": strat.portfolio.name,
    }
    tracer.on_run_start(strat.name, strategy_config)
    tracer.on_run_end("ok")

    # Determine date range from equity curve
    start_date = str(run_result.equity_curve[0][0]) if run_result.equity_curve else ""
    end_date = str(run_result.equity_curve[-1][0]) if run_result.equity_curve else ""

    audit = AuditRecord.build(
        tracer=tracer,
        result=run_result,
        strategy_name=strat.name,
        strategy_config=strategy_config,
        start_date=start_date,
        end_date=end_date,
        initial_cash=float(strat.objectives.get("initial_cash", {}).get("value", 100000.0))
        if "initial_cash" in strat.objectives
        else 100000.0,
    )

    audit_id = audit.run_id
    session._audit_records[audit_id] = audit
    session._save()

    return {
        "audit_id": audit_id,
        "run_id": run_id,
        "strategy_name": audit.strategy_name,
        "strategy_config": audit.strategy_config,
        "mktdata_hash": audit.mktdata_hash,
        "trades_hash": audit.trades_hash,
        "equity_hash": audit.equity_hash,
        "result_hash": audit.result_hash,
        "created_at": audit.created_at,
    }


@registry.tool(
    name="observe_audit_compare",
    description="Compare two audit records to identify which layer (mktdata/trades/equity) changed. "
    "Useful for verifying determinism or understanding the impact of parameter changes.",
)
def observe_audit_compare(
    audit_id_a: str,
    audit_id_b: str,
) -> dict[str, Any]:
    """Compare layered hashes of two audit records."""
    a = session._audit_records.get(audit_id_a)
    if a is None:
        return {"error": f"Audit record '{audit_id_a}' not found"}

    b = session._audit_records.get(audit_id_b)
    if b is None:
        return {"error": f"Audit record '{audit_id_b}' not found"}

    return {
        "audit_a": audit_id_a,
        "audit_b": audit_id_b,
        "strategy_a": a.strategy_name,
        "strategy_b": b.strategy_name,
        "config_match": a.strategy_config == b.strategy_config,
        "mktdata_match": a.mktdata_hash == b.mktdata_hash,
        "trades_match": a.trades_hash == b.trades_hash,
        "equity_match": a.equity_hash == b.equity_hash,
        "result_match": a.result_hash == b.result_hash,
        "config_a": a.strategy_config,
        "config_b": b.strategy_config,
    }
