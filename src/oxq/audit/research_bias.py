"""Research Bias Audit — detect common backtest pitfalls.

P0 checks: execution lag, cost model, OOS requirement, benchmark presence,
survivorship bias, parameter count, trade count, concentration risk,
drawdown severity, and data quality.
"""

from __future__ import annotations

import json
from pathlib import Path

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

    # Load spec
    spec_path = run_path / "strategy_spec.yaml"
    if not spec_path.exists():
        return {
            "status": "fail",
            "checks": [{"id": "spec_missing", "status": "fail", "severity": "fatal", "message": "strategy_spec.yaml not found"}],
            "fatal_count": 1,
            "warning_count": 0,
        }

    spec = StrategySpec.from_yaml(str(spec_path))

    # Load metrics if available
    metrics = {}
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # --- Execution lag ---
    signal_time = spec.signal.signal_time
    trade_time = spec.execution.trade_time
    fill_price_mode = spec.execution.fill_price_mode

    if signal_time == "close_t" and trade_time == "close_t":
        checks.append(_finding(
            "execution_lag", "fail", "fatal",
            "signal_time=close_t and trade_time=close_t — signal generated and filled on same bar",
        ))
    elif signal_time == "close_t" and fill_price_mode in ("close", "mid"):
        checks.append(_finding(
            "execution_lag", "fail", "fatal",
            f"signal_time=close_t and fill_price_mode={fill_price_mode} — "
            "filled at same-bar price",
        ))
    else:
        checks.append(_finding("execution_lag", "pass", "info", "signal/trade timing is reasonable"))

    # --- Cost model ---
    fee_rate = spec.cost.fee_rate
    slippage_rate = spec.cost.slippage_rate
    if fee_rate == 0 and slippage_rate == 0:
        checks.append(_finding("cost_model", "fail", "fatal", "Both fee_rate and slippage_rate are zero — zero-cost model"))
    elif fee_rate == 0:
        checks.append(_finding("cost_model", "fail", "fatal", "fee_rate is zero"))
    elif slippage_rate == 0:
        checks.append(_finding("cost_model", "fail", "fatal", "slippage_rate is zero"))
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
    if spec.universe.type == "static" and not spec.universe.point_in_time:
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
    trade_count = metrics.get("trade_count", 0)
    if trade_count < 10:
        checks.append(_finding("trade_count", "fail", "warning", f"Only {trade_count} trades — statistical significance is low"))
    else:
        checks.append(_finding("trade_count", "pass", "info", f"{trade_count} trades"))

    # --- Concentration ---
    max_dd = metrics.get("max_drawdown", 0)
    if max_dd < -0.50:
        checks.append(_finding("drawdown_tail", "fail", "warning", f"Max drawdown {max_dd:.1%} is severe"))
    else:
        checks.append(_finding("drawdown_tail", "pass", "info", f"Max drawdown: {max_dd:.1%}"))

    # --- Missing data ---
    data_manifest_path = run_path / "data_manifest.json"
    if data_manifest_path.exists():
        manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        missing_ratio = manifest.get("missing_ratio")
        if missing_ratio is None:
            checks.append(_finding(
                "missing_data", "fail", "warning",
                "data_manifest.json has no missing_ratio — data quality not measured",
            ))
        elif missing_ratio > 0.05:
            checks.append(_finding("missing_data", "fail", "warning", f"Data missing ratio {missing_ratio:.1%} is high"))
        else:
            checks.append(_finding("missing_data", "pass", "info", "Data quality acceptable"))
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
