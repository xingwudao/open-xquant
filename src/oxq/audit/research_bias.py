"""Research Bias Audit — detect common backtest pitfalls.

P0 checks: execution lag, cost model, OOS requirement, benchmark presence,
survivorship bias, parameter count, trade count, concentration risk,
drawdown severity, and data quality.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from oxq.spec.execution import derive_execution_semantics
from oxq.spec.schema import StrategySpec


def audit_research(run_dir: str | Path) -> dict:
    """Run P0 research bias audit checks on a backtest run.

    Reads the strategy spec and artifacts from run_dir and checks for
    common backtesting pitfalls.

    Parameters
    ----------
    run_dir : str or Path
        Path to the run directory.

    Returns
    -------
    dict
        Audit result with 'status', 'checks', 'fatal_count', 'warning_count'.
    """
    run_path = Path(run_dir)
    checks: list[dict] = []
    try:
        from oxq.core.component_manifest import (
            load_component_manifests_from_run,
            scoped_component_registries,
            validate_component_manifest_records_from_run,
        )

        for warning in validate_component_manifest_records_from_run(run_path):
            checks.append(_finding(
                "component_manifest_record",
                "fail",
                "warning",
                f"run component manifest record could not be verified from archived artifacts: {warning}",
            ))
    except Exception as exc:
        return {
            "status": "fail",
            "checks": [{
                "id": "component_manifest_load",
                "status": "fail",
                "severity": "fatal",
                "message": f"run component manifest records are invalid: {exc}",
            }],
            "fatal_count": 1,
            "warning_count": 0,
        }

    with scoped_component_registries():
        try:
            load_component_manifests_from_run(run_path)
        except Exception as exc:
            checks.append(_finding(
                "component_manifest_load",
                "fail",
                "warning",
                f"run component manifests could not be loaded for spec validation: {exc}",
            ))

        # Load spec
        spec_path = run_path / "strategy_spec.yaml"
        if not spec_path.exists():
            return {
                "status": "fail",
                "checks": [{"id": "spec_missing", "status": "fail", "severity": "fatal", "message": "strategy_spec.yaml not found"}],
                "fatal_count": 1,
                "warning_count": 0,
            }

        try:
            spec = StrategySpec.from_yaml(str(spec_path))
        except Exception as exc:
            return {
                "status": "fail",
                "checks": [{
                    "id": "spec_parse_error",
                    "status": "fail",
                    "severity": "fatal",
                    "message": f"strategy_spec.yaml could not be parsed: {exc}",
                }],
                "fatal_count": 1,
                "warning_count": 0,
            }
        from oxq.spec.validator import validate

        validation = validate(spec)
        for error in validation.errors:
            checks.append(_finding(
                "spec_validation",
                "fail",
                "fatal" if error.get("severity") == "fatal" else "warning",
                f"{error.get('check')}: {error.get('message')}",
            ))

    # Load metrics if available
    metrics = {}
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            checks.append(_finding("metrics_json", "fail", "fatal", "metrics.json is corrupted — cannot parse"))
            metrics = {}
        if not isinstance(metrics, dict):
            checks.append(_finding("metrics_schema", "fail", "fatal", "metrics.json must contain a JSON object"))
            metrics = {}
    else:
        checks.append(_finding("metrics_json", "fail", "fatal", "metrics.json not found — cannot audit run metrics"))

    required_metrics = {"trade_count", "max_drawdown"}
    missing_metrics = sorted(required_metrics.difference(metrics))
    if missing_metrics:
        checks.append(_finding(
            "metrics_required",
            "fail",
            "fatal",
            f"metrics.json missing required metrics: {missing_metrics}",
        ))
    invalid_metrics = [
        metric for metric in sorted(required_metrics.difference(missing_metrics))
        if _finite_number(metrics.get(metric)) is None
    ]
    if invalid_metrics:
        checks.append(_finding(
            "metrics_schema",
            "fail",
            "fatal",
            f"metrics.json has invalid required metrics: {invalid_metrics}",
        ))

    # --- Execution lag ---
    signal_time = spec.signal.signal_time
    trade_time = spec.execution.trade_time
    effective_execution = None
    try:
        effective_execution = derive_execution_semantics(spec.execution)
    except ValueError:
        pass
    fill_price_mode = effective_execution.fill_price_mode if effective_execution is not None else spec.execution.fill_price_mode
    has_valid_explicit_execution = effective_execution is not None and any(
        (spec.execution.order_timing, spec.execution.price_bar, spec.execution.price_type)
    )

    if signal_time == "close_t" and trade_time == "close_t":
        checks.append(_finding(
            "execution_lag", "fail", "fatal",
            "signal_time=close_t and trade_time=close_t — signal generated and filled on same bar",
        ))
    elif signal_time == "close_t" and effective_execution is not None and (
        effective_execution.price_bar == "same_session"
        or effective_execution.order_timing.startswith("same_session")
    ):
        checks.append(_finding(
            "execution_lag", "fail", "fatal",
            f"signal_time=close_t and fill_price_mode={fill_price_mode} — "
            "filled at same-bar price",
        ))
    elif (
        signal_time == "close_t"
        and effective_execution is not None
        and effective_execution.price_bar == "next_session"
        and effective_execution.price_type in {"close", "mid", "avg", "hl2"}
    ):
        checks.append(_finding(
            "execution_conservatism", "fail", "warning",
            f"next-session {effective_execution.price_type} fills are causal but not conservative",
        ))
    else:
        checks.append(_finding("execution_lag", "pass", "info", "signal/trade timing is reasonable"))

    # --- Cost model ---
    fee_rate = spec.cost.fee_rate
    slippage_rate = spec.cost.slippage_rate
    fee_min = spec.cost.fee_min
    if fee_rate <= 0 and slippage_rate <= 0:
        if fee_rate == 0 and slippage_rate == 0 and has_valid_explicit_execution:
            checks.append(_finding(
                "cost_model",
                "fail",
                "warning",
                "fee_rate and slippage_rate are zero; acceptable only for declared replay-style validation",
            ))
        else:
            checks.append(_finding(
                "cost_model",
                "fail",
                "fatal",
                "fee_rate and slippage_rate must be positive — zero or negative costs are invalid",
            ))
    elif fee_rate <= 0:
        checks.append(_finding("cost_model", "fail", "fatal", "fee_rate must be positive"))
    elif slippage_rate <= 0:
        checks.append(_finding("cost_model", "fail", "fatal", "slippage_rate must be positive"))
    elif fee_min < 0:
        checks.append(_finding("cost_model", "fail", "fatal", "fee_min must not be negative"))
    else:
        checks.append(_finding("cost_model", "pass", "info", f"fee_rate={fee_rate}, slippage_rate={slippage_rate}"))

    # --- OOS required ---
    test_period = spec.validation.test_period
    if not test_period or len(test_period) < 2:
        checks.append(_finding("oos_required", "fail", "fatal", "No out-of-sample test period defined"))
    else:
        checks.append(_finding("oos_required", "pass", "info", f"OOS period: {test_period[0]} to {test_period[1]}"))

    # --- Benchmark present ---
    bench_symbols = spec.benchmark.symbols
    if not bench_symbols:
        checks.append(_finding("benchmark_present", "fail", "warning", "No benchmark defined — difficult to assess excess return"))
    else:
        checks.append(_finding("benchmark_present", "pass", "info", f"Benchmark: {bench_symbols}"))

    # --- Survivorship bias ---
    if spec.universe.type == "static" and spec.universe.point_in_time is not True:
        checks.append(_finding(
            "static_universe_survivorship", "fail", "warning",
            "Static universe without point-in-time may have survivorship bias",
        ))
    else:
        checks.append(_finding("static_universe_survivorship", "pass", "info", "Universe configuration is OK"))

    # --- Parameter count ---
    param_count = sum(len(ind.params) for ind in spec.signal.indicators.values())
    if param_count > 10:
        checks.append(_finding("parameter_count", "fail", "warning", f"{param_count} indicator parameters — risk of overfitting"))
    else:
        checks.append(_finding("parameter_count", "pass", "info", f"{param_count} indicator parameters"))

    # --- Trade count ---
    if test_period and len(test_period) >= 2:
        if "oos_trade_count" not in metrics:
            checks.append(_finding(
                "trade_count",
                "fail",
                "warning",
                "OOS trade count is missing — cannot assess out-of-sample statistical significance",
            ))
        else:
            oos_trade_count = _finite_number(metrics.get("oos_trade_count"))
            if oos_trade_count is None:
                checks.append(_finding("trade_count", "fail", "warning", "OOS trade count is invalid"))
            elif oos_trade_count < 10:
                checks.append(_finding(
                    "trade_count",
                    "fail",
                    "warning",
                    f"Only {oos_trade_count} OOS trades — out-of-sample statistical significance is low",
                ))
            else:
                checks.append(_finding("trade_count", "pass", "info", f"{oos_trade_count} OOS trades"))
    else:
        trade_count = _finite_number(metrics.get("trade_count", 0))
        if trade_count is None:
            checks.append(_finding("trade_count", "fail", "warning", "Trade count is invalid"))
        elif trade_count < 10:
            checks.append(_finding("trade_count", "fail", "warning", f"Only {trade_count} trades — statistical significance is low"))
        else:
            checks.append(_finding("trade_count", "pass", "info", f"{trade_count} trades"))

    # --- Concentration ---
    max_dd = _finite_number(metrics.get("max_drawdown"))
    if max_dd is None:
        checks.append(_finding("drawdown_tail", "fail", "warning", "Max drawdown is invalid"))
    elif max_dd < -0.50:
        checks.append(_finding("drawdown_tail", "fail", "warning", f"Max drawdown {max_dd:.1%} is severe"))
    else:
        checks.append(_finding("drawdown_tail", "pass", "info", f"Max drawdown: {max_dd:.1%}"))

    # --- Missing data ---
    data_manifest_path = run_path / "data_manifest.json"
    if data_manifest_path.exists():
        try:
            manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                checks.append(_finding("missing_data", "fail", "warning", "data_manifest.json must contain a JSON object"))
            else:
                missing_ratio = _finite_number(manifest.get("missing_ratio"))
                if missing_ratio is None:
                    checks.append(_finding(
                        "missing_data", "fail", "warning",
                        "data_manifest.json has no valid missing_ratio — data quality not measured",
                    ))
                elif missing_ratio > 0.05:
                    checks.append(_finding("missing_data", "fail", "warning", f"Data missing ratio {missing_ratio:.1%} is high"))
                else:
                    checks.append(_finding("missing_data", "pass", "info", "Data quality acceptable"))
        except (json.JSONDecodeError, OSError):
            checks.append(_finding("missing_data", "fail", "warning", "data_manifest.json is corrupted — cannot assess data quality"))
    else:
        checks.append(_finding("missing_data", "fail", "warning", "data_manifest.json not found — cannot assess data quality"))

    # Summarize
    fatal_count = sum(1 for c in checks if c["severity"] == "fatal" and c["status"] == "fail")
    warning_count = sum(1 for c in checks if c["severity"] == "warning" and c["status"] == "fail")
    has_fatal = any(c["severity"] == "fatal" and c["status"] == "fail" for c in checks)

    return {
        "status": "fail" if has_fatal else "pass",
        "checks": checks,
        "fatal_count": fatal_count,
        "warning_count": warning_count,
    }


def _finding(check_id: str, status: str, severity: str, message: str) -> dict:
    return {"id": check_id, "status": status, "severity": severity, "message": message}


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
