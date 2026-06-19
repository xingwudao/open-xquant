"""Research Report Generator — produce research_report.md from backtest artifacts."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import yaml

from oxq.audit.reproducibility import audit_reproducibility
from oxq.audit.research_bias import audit_research
from oxq.spec.execution import derive_execution_semantics
from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate


def generate_report(run_dir: str | Path) -> str:
    """Generate a research_report.md from a backtest run directory."""
    run_path = Path(run_dir)
    spec = StrategySpec.from_yaml(str(run_path / "strategy_spec.yaml"))
    spec_dict = yaml.safe_load((run_path / "strategy_spec.yaml").read_text(encoding="utf-8")) or {}
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    execution_assumptions = _load_execution_assumptions(run_path)
    robustness_result = _load_json_object(run_path / "robustness.json")
    repro_audit = audit_reproducibility(run_dir)
    bias_audit = audit_research(run_dir)
    validation_result = validate(spec)

    strategy_id = spec.strategy_id or "unknown"
    hypothesis = spec.research.hypothesis or ""
    decision = _determine_decision(bias_audit, spec_dict, metrics, repro_audit, robustness_result)

    lines: list[str] = []
    lines.append(f"# Research Report: {strategy_id}")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Run ID**: {metrics.get('run_id', run_path.name)}")
    lines.append("")

    # 1. Executive Decision
    lines.append("## 1. Executive Decision")
    lines.append("")
    lines.append(f"**{decision}**")
    lines.append("")
    lines.append(f"_{_decision_explanation()}_")
    lines.append("")

    # 2. Hypothesis
    lines.append("## 2. Hypothesis")
    lines.append("")
    lines.append(hypothesis or "(not specified)")
    lines.append("")

    # 3. Strategy Spec Summary
    lines.append("## 3. Strategy Spec Summary")
    lines.append("")
    lines.append(f"- **Universe**: {spec.universe.type} ({len(spec.universe.symbols)} symbols)")
    lines.append(f"- **Signal**: {spec.signal.signal_time} timing")
    for name, ind in spec.signal.indicators.items():
        lines.append(f"  - {name}: {ind.type} ({ind.params})")
    lines.append(f"- **Portfolio**: {spec.portfolio.type}")
    lines.append(f"- **Execution**: {spec.execution.trade_time} trade, {_effective_fill_price_mode(spec)} fill")
    lines.append("")

    # 4. Data and Execution Assumptions
    lines.append("## 4. Data and Execution Assumptions")
    lines.append("")
    lines.append(f"- **Fee**: {spec.cost.fee_rate:.3%}")
    lines.append(f"- **Slippage**: {spec.cost.slippage_rate:.3%}")
    lines.append(f"- **Initial Cash**: ${spec.execution.initial_cash:,.0f}")
    lines.append(f"- **Price Adjustment**: {spec.data.price_adjustment}")
    if execution_assumptions is not None:
        lines.append("")
        lines.append("### Execution Assumptions")
        lines.append("")
        lines.extend(_format_execution_assumption_lines(execution_assumptions))
    lines.append("")

    # 5. Backtest Metrics
    lines.append("## 5. Backtest Metrics")
    lines.append("")
    lines.append("### Metrics Profile")
    lines.append("")
    lines.extend(_format_metric_assumption_lines(metrics, spec))
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Return | {_format_percent(metrics.get('total_return'))} |")
    lines.append(f"| Annualized Return | {_format_percent(metrics.get('annualized_return'))} |")
    lines.append(f"| Annualized Volatility | {_format_percent(metrics.get('annualized_volatility'))} |")
    lines.append(f"| Max Drawdown | {_format_percent(metrics.get('max_drawdown'))} |")
    lines.append(f"| Sharpe Ratio | {_format_float(metrics.get('sharpe_ratio'))} |")
    lines.append(f"| Sortino Ratio | {_format_float(metrics.get('sortino_ratio'))} |")
    lines.append(f"| Calmar Ratio | {_format_float(metrics.get('calmar_ratio'))} |")
    lines.append(f"| Trade Count | {metrics.get('trade_count', 0)} |")
    lines.append(f"| Cost Paid | {_format_money(metrics.get('cost_paid'))} |")
    lines.append("")
    if _has_is_oos_metrics(metrics):
        lines.append("### IS/OOS Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.extend(_format_is_oos_metric_lines(metrics))
        lines.append("")

    # 6. Benchmark Comparison
    lines.append("## 6. Benchmark Comparison")
    lines.append("")
    if spec.benchmark.symbols:
        lines.append(f"Benchmark: {', '.join(spec.benchmark.symbols)}")
    else:
        lines.append("(No benchmark defined)")
    lines.append("")

    # 7. Reproducibility Audit
    lines.append("## 7. Reproducibility Audit")
    lines.append("")
    lines.append(f"**Status**: {repro_audit['status'].upper()}")
    lines.append("")
    for c in repro_audit["checks"]:
        icon = "PASS" if c["status"] == "pass" else ("INFO" if c["status"] == "info" else "FAIL")
        lines.append(f"- [{c['severity'].upper()}] {icon} **{c['id']}**: {c['message']}")
    lines.append("")

    # 8. Research Bias Audit
    lines.append("## 8. Research Bias Audit")
    lines.append("")
    lines.append(
        f"**Status**: {bias_audit['status'].upper()} "
        f"(Fatal: {bias_audit['fatal_count']}, Warnings: {bias_audit['warning_count']})"
    )
    lines.append("")
    for c in bias_audit["checks"]:
        icon = "PASS" if c["status"] == "pass" else ("INFO" if c["status"] == "info" else "FAIL")
        lines.append(f"- [{c['severity'].upper()}] {icon} **{c['id']}**: {c['message']}")
    lines.append("")
    lines.append("### Validation Classification")
    lines.append("")
    lines.extend(_format_validation_classification_lines(validation_result.to_dict()))
    lines.append("")

    # 9. Robustness Tests
    lines.append("## 9. Robustness Tests")
    lines.append("")
    if robustness_result is not None:
        lines.extend(_format_robustness_result_lines(robustness_result))
    elif spec.robustness.cost_multiplier:
        lines.append(f"- Cost multiplier scenarios: {spec.robustness.cost_multiplier}")
    if robustness_result is None and spec.robustness.parameter_perturbation:
        lines.append(f"- Parameter perturbation: {list(spec.robustness.parameter_perturbation.keys())}")
    if robustness_result is None and spec.robustness.regime_analysis:
        lines.append("- Regime analysis: enabled")
    if robustness_result is None and not spec.robustness.cost_multiplier and not spec.robustness.parameter_perturbation:
        lines.append("(No robustness tests configured)")
    lines.append("")

    # 10. Failure Modes
    lines.append("## 10. Failure Modes")
    lines.append("")
    fatal_checks = [c for c in bias_audit["checks"] if c["severity"] == "fatal" and c["status"] == "fail"]
    warning_checks = [c for c in bias_audit["checks"] if c["severity"] == "warning" and c["status"] == "fail"]
    if fatal_checks:
        lines.append("### Fatal Issues")
        for c in fatal_checks:
            lines.append(f"- **{c['id']}**: {c['message']}")
    if warning_checks:
        lines.append("### Warnings")
        for c in warning_checks:
            lines.append(f"- **{c['id']}**: {c['message']}")
    if not fatal_checks and not warning_checks:
        lines.append("No significant issues detected.")
    lines.append("")

    # 11. Next Actions
    lines.append("## 11. Next Actions")
    lines.append("")
    if decision == "REJECT":
        lines.append("- Fix fatal audit findings before reconsidering this strategy.")
    elif decision == "WATCHLIST":
        lines.append("- Address warnings before promoting to paper trading.")
        lines.append("- Run robustness tests with cost multiplier.")
    else:
        lines.append("- Proceed to paper trading with monitoring.")
        lines.append("- Set up live monitoring and drift detection.")
    lines.append("")

    return "\n".join(lines)


def _determine_decision(
    bias_audit: dict,
    spec_dict: dict,
    metrics: dict,
    repro_audit: dict | None = None,
    robustness_result: dict | None = None,
) -> str:
    """Determine the executive decision based on audit results and decision policy."""
    decision_policy = spec_dict.get("decision_policy", {})

    if repro_audit and repro_audit.get("status") == "fail":
        return "REJECT"

    if robustness_result:
        robustness_status = robustness_result.get("status")
        if robustness_status in {"error", "fragile"}:
            return "REJECT"

    if bias_audit.get("fatal_count", 0) > 0:
        return "REJECT"

    reject_if = decision_policy.get("reject_if", {})
    # OOS policy thresholds require OOS-only metrics.
    policy_oos_sharpe = _finite_metric(metrics, "oos_sharpe_ratio")
    max_dd = _finite_metric(metrics, "oos_max_drawdown")

    if "oos_sharpe_lt" in reject_if:
        threshold = _as_finite_float(reject_if["oos_sharpe_lt"])
        if threshold is None or policy_oos_sharpe is None or threshold > policy_oos_sharpe:
            return "REJECT"
    if "max_drawdown_lt" in reject_if:
        threshold = _as_finite_float(reject_if["max_drawdown_lt"])
        if threshold is None or max_dd is None or threshold > max_dd:
            return "REJECT"

    if robustness_result and robustness_result.get("status") == "warn":
        return "WATCHLIST"

    promote_if = decision_policy.get("promote_if", {})
    # Only check thresholds that are explicitly configured
    promote_checks: list[bool] = []
    if "oos_sharpe_gte" in promote_if:
        threshold = _as_finite_float(promote_if["oos_sharpe_gte"])
        if threshold is None or policy_oos_sharpe is None:
            return "WATCHLIST"
        promote_checks.append(threshold <= policy_oos_sharpe)
    if "max_drawdown_gte" in promote_if:
        threshold = _as_finite_float(promote_if["max_drawdown_gte"])
        if threshold is None or max_dd is None:
            return "WATCHLIST"
        promote_checks.append(threshold <= max_dd)
    if promote_if:
        return "PAPER TRADING CANDIDATE" if promote_checks and all(promote_checks) else "WATCHLIST"

    if bias_audit.get("warning_count", 0) > 0:
        return "WATCHLIST"

    return "PAPER TRADING CANDIDATE"


def _finite_metric(metrics: dict, primary: str, fallback: str | None = None) -> float | None:
    value = metrics[primary] if primary in metrics else metrics.get(fallback) if fallback else None
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _format_percent(value: object) -> str:
    parsed = _as_finite_float(value)
    return "N/A" if parsed is None else f"{parsed:.2%}"


def _format_float(value: object) -> str:
    parsed = _as_finite_float(value)
    return "N/A" if parsed is None else f"{parsed:.2f}"


def _format_money(value: object) -> str:
    parsed = _as_finite_float(value)
    return "N/A" if parsed is None else f"${parsed:.2f}"


def _format_metric_assumption_lines(metrics: dict, spec: StrategySpec) -> list[str]:
    del spec
    profile = metrics.get("metrics_profile") or "open_xquant_default"
    assumptions = metrics.get("metric_assumptions")
    if not isinstance(assumptions, dict):
        assumptions = {
            "return_type": "simple",
            "risk_free_rate": 0.0,
            "annualization_days": 252,
            "calmar_denominator": "max_drawdown",
            "evaluation_window": "full",
        }
    lines = [f"- **Profile**: {_format_assumption_value(profile)}"]
    for key in ("return_type", "risk_free_rate", "annualization_days", "calmar_denominator", "evaluation_window"):
        value = assumptions.get(key)
        formatted = _format_percent(value) if key == "risk_free_rate" else _format_assumption_value(value)
        lines.append(f"- **{key}**: {formatted}")
    if profile != "open_xquant_default":
        lines.append("- **Note**: Non-default metrics profile; compare results only against runs using the same assumptions.")
    return lines


def _has_is_oos_metrics(metrics: dict) -> bool:
    return any(key in metrics for key in (
        "is_total_return",
        "is_annualized_return",
        "is_annualized_volatility",
        "is_max_drawdown",
        "is_sharpe_ratio",
        "is_calmar_ratio",
        "oos_total_return",
        "oos_annualized_return",
        "oos_annualized_volatility",
        "oos_max_drawdown",
        "oos_sharpe_ratio",
        "oos_calmar_ratio",
        "oos_trade_count",
    ))


def _format_is_oos_metric_lines(metrics: dict) -> list[str]:
    rows = [
        ("IS Total Return", _format_percent(metrics.get("is_total_return"))),
        ("IS Annualized Return", _format_percent(metrics.get("is_annualized_return"))),
        ("IS Annualized Volatility", _format_percent(metrics.get("is_annualized_volatility"))),
        ("IS Max Drawdown", _format_percent(metrics.get("is_max_drawdown"))),
        ("IS Sharpe Ratio", _format_float(metrics.get("is_sharpe_ratio"))),
        ("IS Calmar Ratio", _format_float(metrics.get("is_calmar_ratio"))),
        ("OOS Total Return", _format_percent(metrics.get("oos_total_return"))),
        ("OOS Annualized Return", _format_percent(metrics.get("oos_annualized_return"))),
        ("OOS Annualized Volatility", _format_percent(metrics.get("oos_annualized_volatility"))),
        ("OOS Max Drawdown", _format_percent(metrics.get("oos_max_drawdown"))),
        ("OOS Sharpe Ratio", _format_float(metrics.get("oos_sharpe_ratio"))),
        ("OOS Calmar Ratio", _format_float(metrics.get("oos_calmar_ratio"))),
        ("OOS Trade Count", str(metrics.get("oos_trade_count", "N/A"))),
    ]
    return [f"| {name} | {value} |" for name, value in rows]


def _format_validation_classification_lines(validation_result: dict) -> list[str]:
    findings = list(validation_result.get("errors", [])) + list(validation_result.get("warnings", []))
    dimensions = ("causal", "executable", "conservative", "production_consistent")
    lines: list[str] = [f"- **status**: {validation_result.get('status', 'unknown')}"]
    unclassified = [finding for finding in findings if not finding.get("dimensions")]
    if unclassified:
        labels = ", ".join(f"{finding.get('severity', 'unknown')}:{finding.get('check', 'unknown')}" for finding in unclassified)
        lines.append(f"- **unclassified**: {labels}")
    for dimension in dimensions:
        matching = [finding for finding in findings if dimension in finding.get("dimensions", [])]
        if not matching:
            lines.append(f"- **{dimension}**: pass")
            continue
        labels = ", ".join(f"{finding.get('severity', 'unknown')}:{finding.get('check', 'unknown')}" for finding in matching)
        lines.append(f"- **{dimension}**: {labels}")
    return lines


def _load_execution_assumptions(run_path: Path) -> dict | None:
    assumptions_path = run_path / "execution_assumptions.json"
    return _load_json_object(assumptions_path)


def _load_json_object(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _format_robustness_result_lines(result: dict) -> list[str]:
    lines = [f"**Status**: {str(result.get('status', 'unknown')).upper()}"]
    baseline_sharpe = result.get("baseline_sharpe")
    if baseline_sharpe is not None:
        lines.append(f"- **Baseline Sharpe**: {_format_float(baseline_sharpe)}")
    tests = result.get("tests")
    if not isinstance(tests, list):
        lines.append("- Robustness artifact does not contain a tests list.")
        return lines
    for test in tests:
        if not isinstance(test, dict):
            continue
        name = _format_assumption_value(test.get("name"))
        status = str(test.get("status", "unknown")).upper()
        message = _format_assumption_value(test.get("message"))
        lines.append(f"- [{status}] **{name}**: {message}")
        if "baseline_sharpe" in test or "perturbed_sharpe" in test:
            lines.append(
                "- **Sharpe comparison**: "
                f"{_format_float(test.get('baseline_sharpe'))} -> {_format_float(test.get('perturbed_sharpe'))}"
            )
        if isinstance(test.get("results"), list):
            lines.append(f"- **Parameter perturbation results**: {_summarize_status_counts(test['results'])}")
        if isinstance(test.get("regimes"), dict):
            lines.append(f"- **Regimes**: {_summarize_regimes(test['regimes'])}")
    return lines


def _summarize_status_counts(results: list) -> str:
    counts: dict[str, int] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    if not counts:
        return "N/A"
    ordered = [status for status in ("pass", "warn", "fail", "error") if status in counts]
    ordered.extend(status for status in sorted(counts) if status not in ordered)
    return ", ".join(f"{status}={counts[status]}" for status in ordered)


def _summarize_regimes(regimes: dict) -> str:
    chunks: list[str] = []
    for name, bucket in regimes.items():
        if not isinstance(bucket, dict):
            continue
        chunks.append(
            f"{name} (dates={bucket.get('date_count', 'N/A')}, trades={bucket.get('trade_count', 'N/A')})"
        )
    return ", ".join(chunks) if chunks else "N/A"


def _effective_fill_price_mode(spec: StrategySpec) -> str:
    try:
        return derive_execution_semantics(spec.execution).fill_price_mode
    except ValueError:
        return spec.execution.fill_price_mode


def _format_execution_assumption_lines(assumptions: dict) -> list[str]:
    lines = [
        f"- **order_timing**: {_format_assumption_value(assumptions.get('order_timing'))}",
        f"- **price_bar**: {_format_assumption_value(assumptions.get('price_bar'))}",
        f"- **price_type**: {_format_assumption_value(assumptions.get('price_type'))}",
    ]
    if "fill_price_mode" in assumptions:
        lines.append(f"- **fill_price_mode**: {_format_assumption_value(assumptions.get('fill_price_mode'))}")
    lines.append(f"- **cash_annual_return**: {_format_percent(assumptions.get('cash_annual_return'))}")
    lines.append(f"- **default_lot_size**: {_format_assumption_value(_default_lot_size(assumptions))}")
    if "calendar" in assumptions:
        lines.append(f"- **calendar**: {_format_assumption_value(assumptions.get('calendar'))}")
    if "runtime_calendar" in assumptions:
        lines.append(f"- **runtime_calendar**: {_format_assumption_value(assumptions.get('runtime_calendar'))}")
    return lines


def _default_lot_size(assumptions: dict) -> object:
    lot_size_config = assumptions.get("lot_size_config")
    if isinstance(lot_size_config, dict) and "default" in lot_size_config:
        return lot_size_config.get("default")
    return assumptions.get("lot_size")


def _format_assumption_value(value: object) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value)


def _decision_explanation() -> str:
    return (
        "**REJECT** = Fatal audit findings, do not proceed.\n"
        "**WATCHLIST** = Needs further investigation before promotion.\n"
        "**PAPER TRADING CANDIDATE** = Passes basic audits, suitable for paper trading evaluation."
    )
