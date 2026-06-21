"""Research Report Generator — produce research_report.md from backtest artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from oxq.audit.reproducibility import audit_reproducibility
from oxq.audit.research_bias import audit_research
from oxq.report.assets import ReportAsset, list_report_assets
from oxq.report.i18n import messages
from oxq.spec.execution import derive_execution_semantics
from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate


@dataclass(frozen=True)
class ReportOutputs:
    markdown: Path | None = None
    html: Path | None = None


def generate_report(run_dir: str | Path, lang: str = "zh") -> str:
    """Generate a research_report.md from a backtest run directory."""
    run_path = Path(run_dir)
    msg = messages(lang)
    headings = msg["headings"]
    labels = msg["labels"]
    spec = StrategySpec.from_yaml(str(run_path / "strategy_spec.yaml"))
    spec_dict = yaml.safe_load((run_path / "strategy_spec.yaml").read_text(encoding="utf-8")) or {}
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    execution_assumptions = _load_execution_assumptions(run_path)
    repro_audit = audit_reproducibility(run_dir)
    robustness_result = _load_verified_robustness_result(run_path, repro_audit)
    bias_audit = audit_research(run_dir)
    validation_result = validate(spec)
    assets = list_report_assets(run_path)

    strategy_id = spec.strategy_id or "unknown"
    hypothesis = spec.research.hypothesis or ""
    decision = _determine_decision(bias_audit, spec_dict, metrics, repro_audit, robustness_result)

    lines: list[str] = []
    lines.append(f"# {msg['report_title'].format(strategy_id=strategy_id)}")
    lines.append("")
    lines.append(f"**{msg['generated']}**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**{msg['run_id']}**: {metrics.get('run_id', run_path.name)}")
    lines.append("")

    # 1. Executive Decision
    lines.append(f"## 1. {headings['decision']}")
    lines.append("")
    lines.append(f"**{decision}**")
    lines.append("")
    lines.append(f"_{_decision_explanation(lang)}_")
    lines.append("")

    # 2. Hypothesis
    lines.append(f"## 2. {headings['hypothesis']}")
    lines.append("")
    lines.append(hypothesis or f"({msg['not_specified']})")
    lines.append("")

    # 3. Strategy Spec Summary
    lines.append(f"## 3. {headings['strategy']}")
    lines.append("")
    lines.append(f"- **{labels['universe']}**: {spec.universe.type} ({len(spec.universe.symbols)} symbols)")
    lines.append(f"- **{labels['signal']}**: {spec.signal.signal_time} timing")
    for name, ind in spec.signal.indicators.items():
        lines.append(f"  - {name}: {ind.type} ({ind.params})")
    lines.append(f"- **{labels['portfolio']}**: {spec.portfolio.type}")
    lines.append(f"- **{labels['execution']}**: {spec.execution.trade_time} trade, {_effective_fill_price_mode(spec)} fill")
    lines.append("")

    # 4. Data and Execution Assumptions
    lines.append(f"## 4. {headings['assumptions']}")
    lines.append("")
    lines.append(f"- **{labels['fee']}**: {spec.cost.fee_rate:.3%}")
    lines.append(f"- **{labels['slippage']}**: {spec.cost.slippage_rate:.3%}")
    lines.append(f"- **{labels['initial_cash']}**: ${spec.execution.initial_cash:,.0f}")
    lines.append(f"- **{labels['price_adjustment']}**: {spec.data.price_adjustment}")
    if execution_assumptions is not None:
        lines.append("")
        lines.append("### Execution Assumptions")
        lines.append("")
        lines.extend(_format_execution_assumption_lines(execution_assumptions))
    lines.append("")

    # 5. Backtest Metrics
    lines.append(f"## 5. {headings['metrics']}")
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
    lines.append(f"## 6. {headings['benchmark']}")
    lines.append("")
    if spec.benchmark.symbols:
        lines.append(f"Benchmark: {', '.join(spec.benchmark.symbols)}")
    else:
        lines.append(f"({msg['no_benchmark']})")
    lines.append("")

    # 7. Report Assets
    lines.append(f"## 7. {headings['assets']}")
    lines.append("")
    lines.extend(_format_asset_lines(assets, lang))
    lines.append("")

    # 8. Reproducibility Audit
    lines.append(f"## 8. {headings['reproducibility']}")
    lines.append("")
    lines.append(f"**{labels['status']}**: {repro_audit['status'].upper()}")
    lines.append("")
    for c in repro_audit["checks"]:
        icon = "PASS" if c["status"] == "pass" else ("INFO" if c["status"] == "info" else "FAIL")
        lines.append(f"- [{c['severity'].upper()}] {icon} **{c['id']}**: {c['message']}")
    lines.append("")

    # 9. Research Bias Audit
    lines.append(f"## 9. {headings['research_bias']}")
    lines.append("")
    lines.append(
        f"**{labels['status']}**: {bias_audit['status'].upper()} "
        f"({labels['fatal']}: {bias_audit['fatal_count']}, {labels['warnings']}: {bias_audit['warning_count']})"
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

    # 10. Robustness Tests
    lines.append(f"## 10. {headings['robustness']}")
    lines.append("")
    if robustness_result is not None:
        lines.extend(_format_robustness_result_lines(robustness_result))
    elif spec.robustness.cost_multiplier:
        lines.append(f"- Cost multiplier scenarios: {spec.robustness.cost_multiplier}")
    if robustness_result is None and spec.robustness.parameter_perturbation:
        lines.append(f"- Parameter perturbation: {list(spec.robustness.parameter_perturbation.keys())}")
    if robustness_result is None and spec.robustness.regime_analysis:
        lines.append("- Regime analysis: enabled")
    if (
        robustness_result is None
        and not spec.robustness.cost_multiplier
        and not spec.robustness.parameter_perturbation
        and not spec.robustness.regime_analysis
    ):
        lines.append(f"({msg['no_robustness']})")
    lines.append("")

    # 11. Failure Modes
    lines.append(f"## 11. {headings['failure_modes']}")
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
        lines.append(msg["no_significant_issues"])
    lines.append("")

    # 12. Next Actions
    lines.append(f"## 12. {headings['next_actions']}")
    lines.append("")
    next_actions = msg["next_actions"]
    if decision == "REJECT":
        lines.append(next_actions["reject"])
    elif decision == "WATCHLIST":
        lines.append(next_actions["watchlist_1"])
        lines.append(next_actions["watchlist_2"])
    else:
        lines.append(next_actions["promote_1"])
        lines.append(next_actions["promote_2"])
    lines.append("")

    return "\n".join(lines)


def write_report_files(
    run_dir: str | Path,
    *,
    lang: str = "zh",
    output_format: str = "all",
    out: str | Path | None = None,
) -> ReportOutputs:
    """Write Markdown and/or HTML report files for a run directory."""
    output_format = output_format.lower()
    if output_format not in {"all", "markdown", "html"}:
        raise ValueError(f"unsupported report format: {output_format}")

    run_path = Path(run_dir)
    markdown_path: Path | None = None
    html_path: Path | None = None

    if output_format in {"all", "markdown"}:
        report_md = generate_report(run_path, lang=lang)
        markdown_path = Path(out) if out is not None else run_path / "research_report.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report_md, encoding="utf-8")

    if output_format in {"all", "html"}:
        from oxq.report.html import render_html_report

        report_html = render_html_report(run_path, lang=lang)
        if output_format == "all" and out is not None:
            html_path = Path(out).with_suffix(".html")
        elif output_format == "html" and out is not None:
            html_path = Path(out)
        else:
            html_path = run_path / "research_report.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(report_html, encoding="utf-8")

    return ReportOutputs(markdown=markdown_path, html=html_path)


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

    if _has_actionable_robustness_warning(robustness_result):
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


def _has_actionable_robustness_warning(robustness_result: dict | None) -> bool:
    if not robustness_result or robustness_result.get("status") != "warn":
        return False
    tests = robustness_result.get("tests")
    if not isinstance(tests, list):
        return True
    for test in tests:
        if not isinstance(test, dict):
            return True
        if test.get("status") not in {"warn", "fail", "error"}:
            continue
        if _is_unconfigured_robustness_warning(test):
            continue
        return True
    return False


def _is_unconfigured_robustness_warning(test: dict) -> bool:
    if test.get("status") != "warn":
        return False
    name = test.get("name")
    message = str(test.get("message", ""))
    return (
        name == "parameter_perturbation"
        and message == "No parameter perturbation targets configured in spec"
    ) or (
        name == "regime_analysis"
        and message == "Regime analysis not configured"
    )


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


def _format_asset_lines(assets: list[ReportAsset], lang: str) -> list[str]:
    msg = messages(lang)
    labels = msg["labels"]
    if not assets:
        return [msg["no_assets"]]

    lines: list[str] = []
    figure_index = 0
    for asset in assets:
        report_path = f"report_assets/{asset.path}"
        if asset.kind == "figure":
            figure_index += 1
            lines.append(f"![{asset.title}]({report_path})")
            lines.append("")
            caption = asset.caption or asset.title
            lines.append(f"{msg['figure_prefix']} {figure_index}. {caption}")
        else:
            lines.append(f"- **{msg['attachment']}**: [{asset.title}]({report_path})")
            if asset.caption:
                lines.append(f"  - {asset.caption}")
        lines.append(f"- **id**: {asset.id}")
        lines.append(f"- **{labels['kind']}**: {asset.kind}")
        lines.append(f"- **sha256**: {asset.sha256}")
        if asset.source.script:
            lines.append(f"- **{labels['source_script']}**: {asset.source.script}")
        if asset.source.input_artifacts:
            lines.append(f"- **{labels['source_artifacts']}**: {', '.join(asset.source.input_artifacts)}")
        lines.append("")
    return lines


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


def _load_verified_robustness_result(run_path: Path, repro_audit: dict) -> dict | None:
    robustness_path = run_path / "robustness.json"
    if not robustness_path.exists():
        return None

    artifact_hashes_path = run_path / "artifact_hashes.json"
    if artifact_hashes_path.exists():
        artifact_hashes = _load_json_object(artifact_hashes_path)
        if not artifact_hashes or "robustness.json" not in artifact_hashes:
            return None
        checks = repro_audit.get("checks", [])
        if not any(
            isinstance(check, dict)
            and check.get("id") == "robustness_hash"
            and check.get("status") == "pass"
            for check in checks
        ):
            return None

    return _load_json_object(robustness_path)


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


def _decision_explanation(lang: str = "en") -> str:
    return str(messages(lang)["decision_explanation"])
