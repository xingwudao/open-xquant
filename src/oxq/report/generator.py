"""Research Report Generator — produce research_report.md from backtest artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from oxq.audit.reproducibility import audit_reproducibility
from oxq.audit.research_bias import audit_research
from oxq.spec.schema import StrategySpec


def generate_report(run_dir: str | Path) -> str:
    """Generate a research_report.md from a backtest run directory."""
    run_path = Path(run_dir)
    spec = StrategySpec.from_yaml(str(run_path / "strategy_spec.yaml"))
    spec_dict = yaml.safe_load((run_path / "strategy_spec.yaml").read_text(encoding="utf-8")) or {}
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    repro_audit = audit_reproducibility(run_dir)
    bias_audit = audit_research(run_dir)

    strategy_id = spec.strategy_id or "unknown"
    hypothesis = spec.research.hypothesis or ""
    decision = _determine_decision(bias_audit, spec_dict, metrics)

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
    lines.append(f"- **Execution**: {spec.execution.trade_time} trade, {spec.execution.fill_price_mode} fill")
    lines.append("")

    # 4. Data and Execution Assumptions
    lines.append("## 4. Data and Execution Assumptions")
    lines.append("")
    lines.append(f"- **Fee**: {spec.cost.fee_rate:.3%}")
    lines.append(f"- **Slippage**: {spec.cost.slippage_rate:.3%}")
    lines.append(f"- **Initial Cash**: ${spec.execution.initial_cash:,.0f}")
    lines.append(f"- **Price Adjustment**: {spec.data.price_adjustment}")
    lines.append("")

    # 5. Backtest Metrics
    lines.append("## 5. Backtest Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Return | {metrics.get('total_return', 0):.2%} |")
    lines.append(f"| Annualized Return | {metrics.get('annualized_return', 0):.2%} |")
    lines.append(f"| Annualized Volatility | {metrics.get('annualized_volatility', 0):.2%} |")
    lines.append(f"| Max Drawdown | {metrics.get('max_drawdown', 0):.2%} |")
    lines.append(f"| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |")
    lines.append(f"| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |")
    lines.append(f"| Calmar Ratio | {metrics.get('calmar_ratio', 0):.2f} |")
    lines.append(f"| Trade Count | {metrics.get('trade_count', 0)} |")
    lines.append(f"| Cost Paid | ${metrics.get('cost_paid', 0):.2f} |")
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

    # 9. Robustness Tests
    lines.append("## 9. Robustness Tests")
    lines.append("")
    if spec.robustness.cost_multiplier:
        lines.append(f"- Cost multiplier scenarios: {spec.robustness.cost_multiplier}")
    if spec.robustness.parameter_perturbation:
        lines.append(f"- Parameter perturbation: {list(spec.robustness.parameter_perturbation.keys())}")
    if spec.robustness.regime_analysis:
        lines.append("- Regime analysis: enabled")
    if not spec.robustness.cost_multiplier and not spec.robustness.parameter_perturbation:
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


def _determine_decision(bias_audit: dict, spec_dict: dict, metrics: dict) -> str:
    """Determine the executive decision based on audit results and decision policy."""
    decision_policy = spec_dict.get("decision_policy", {})

    if bias_audit.get("fatal_count", 0) > 0:
        reject_if = decision_policy.get("reject_if", {})
        if reject_if.get("fatal_audit_findings", True):
            return "REJECT"

    reject_if = decision_policy.get("reject_if", {})
    # Prefer OOS-only metrics for OOS decisions; fall back to aggregate
    oos_sharpe = metrics.get("oos_sharpe_ratio", metrics.get("sharpe_ratio", 0))
    max_dd = metrics.get("oos_max_drawdown", metrics.get("max_drawdown", 0))

    if "oos_sharpe_lt" in reject_if and reject_if["oos_sharpe_lt"] > oos_sharpe:
        return "REJECT"
    if "max_drawdown_lt" in reject_if and reject_if["max_drawdown_lt"] > max_dd:
        return "REJECT"

    promote_if = decision_policy.get("promote_if", {})
    # Only check thresholds that are explicitly configured
    promote_checks: list[bool] = []
    if "oos_sharpe_gte" in promote_if:
        promote_checks.append(promote_if["oos_sharpe_gte"] <= oos_sharpe)
    if "max_drawdown_gte" in promote_if:
        promote_checks.append(promote_if["max_drawdown_gte"] <= max_dd)
    if promote_checks and all(promote_checks):
        return "PAPER TRADING CANDIDATE"

    if bias_audit.get("warning_count", 0) > 0:
        return "WATCHLIST"

    return "PAPER TRADING CANDIDATE"


def _decision_explanation() -> str:
    return (
        "**REJECT** = Fatal audit findings, do not proceed.\n"
        "**WATCHLIST** = Needs further investigation before promotion.\n"
        "**PAPER TRADING CANDIDATE** = Passes basic audits, suitable for paper trading evaluation."
    )
