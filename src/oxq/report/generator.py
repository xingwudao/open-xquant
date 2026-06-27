"""Research Report Generator — produce research_report.md from backtest artifacts."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from oxq.audit.reproducibility import audit_reproducibility
from oxq.audit.research_bias import audit_research
from oxq.report.artifacts import RunArtifacts
from oxq.report.assets import ReportAsset, list_report_assets
from oxq.report.facts import ReportFacts, build_report_facts
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
    subheadings = msg["subheadings"]
    labels = msg["labels"]
    spec = StrategySpec.from_yaml(str(run_path / "strategy_spec.yaml"))
    spec_dict = yaml.safe_load((run_path / "strategy_spec.yaml").read_text(encoding="utf-8")) or {}
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    facts = build_report_facts(RunArtifacts.load(run_path))
    execution_assumptions = _load_execution_assumptions(run_path)
    compiled_plan = _load_json_object(run_path / "compiled_plan.json") or {}
    data_manifest = _load_json_object(run_path / "data_manifest.json") or {}
    repro_audit = audit_reproducibility(run_dir)
    robustness_result = _load_verified_robustness_result(run_path, repro_audit)
    bias_audit = audit_research(run_dir)
    validation_result = validate(spec)
    assets = list_report_assets(run_path)
    benchmark_configured = bool(spec.benchmark.symbols)
    robustness_configured = _robustness_configured(spec_dict)
    benchmark_metrics = (
        _benchmark_relative_metrics(run_path)
        if benchmark_configured and _benchmark_artifact_trusted(repro_audit)
        else None
    )

    strategy_id = spec.strategy_id or "unknown"
    hypothesis = spec.research.hypothesis or ""
    decision = _determine_decision(bias_audit, spec_dict, metrics, repro_audit, robustness_result)
    decision_summary = _decision_summary(
        decision,
        metrics,
        repro_audit,
        bias_audit,
        robustness_result,
        benchmark_metrics,
        lang,
        robustness_configured=robustness_configured,
        benchmark_configured=benchmark_configured,
        oos_evidence_required=_requires_oos_trade_evidence(spec_dict),
    )

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
    lines.extend(_format_decision_summary_lines(decision_summary, lang))
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
    lines.extend(_format_date_fact_lines(facts, lang))
    if execution_assumptions is not None:
        lines.append("")
        lines.append(f"### {subheadings['execution_assumptions']}")
        lines.append("")
        lines.extend(_format_execution_assumption_lines(execution_assumptions))
    runtime_disclosure = _format_runtime_disclosure_lines(
        compiled_plan,
        data_manifest,
        lang,
        artifacts_trusted=_runtime_artifacts_trusted(
            repro_audit,
            require_compiled_plan_hash=bool(compiled_plan),
            require_data_manifest_hash=bool(data_manifest),
        ),
    )
    if runtime_disclosure:
        lines.append("")
        lines.append(f"### {subheadings['runtime_disclosure']}")
        lines.append("")
        lines.extend(runtime_disclosure)
    lines.append("")

    # 5. Backtest Metrics
    lines.append(f"## 5. {headings['metrics']}")
    lines.append("")
    lines.append(f"### {msg['key_metrics']['title']}")
    lines.append("")
    lines.extend(_format_key_metric_snapshot_lines(metrics, decision, lang))
    lines.append("")
    lines.append(f"### {subheadings['metrics_profile']}")
    lines.append("")
    lines.extend(_format_metric_assumption_lines(metrics, spec, lang))
    lines.append("")
    lines.extend(_format_main_metric_lines(metrics, lang))
    lines.append("")
    if _has_is_oos_metrics(metrics):
        lines.append(f"### {subheadings['is_oos_metrics']}")
        lines.append("")
        lines.extend(_format_is_oos_metric_lines(metrics, lang))
        lines.append("")

    # 6. Benchmark Comparison
    lines.append(f"## 6. {headings['benchmark']}")
    lines.append("")
    if spec.benchmark.symbols:
        lines.append(f"{labels['benchmark']}: {', '.join(spec.benchmark.symbols)}")
    else:
        lines.append(f"({msg['no_benchmark']})")
    if benchmark_metrics is not None:
        lines.append("")
        lines.append(f"### {msg['benchmark_metrics']['title']}")
        lines.append("")
        lines.extend(_format_benchmark_metric_lines(benchmark_metrics, lang))
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
    lines.append(f"### {subheadings['validation_classification']}")
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
        lines.append(f"### {subheadings['fatal_issues']}")
        for c in fatal_checks:
            lines.append(f"- **{c['id']}**: {c['message']}")
    if warning_checks:
        lines.append(f"### {subheadings['warnings']}")
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
    elif decision == "NO EVIDENCE":
        lines.append(next_actions["no_evidence"])
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
    report_md: str | None = None

    if output_format in {"all", "markdown"}:
        report_md = generate_report(run_path, lang=lang)
        markdown_path = _markdown_output_path(run_path, output_format, out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report_md, encoding="utf-8")

    if output_format in {"all", "html"}:
        from oxq.report.html import render_html_report, render_markdown_html_report

        if report_md is None:
            report_md = _read_existing_report_markdown(run_path)
        report_html = (
            render_markdown_html_report(report_md, lang=lang)
            if report_md is not None
            else render_html_report(run_path, lang=lang)
        )
        html_path = _html_output_path(run_path, output_format, out)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(report_html, encoding="utf-8")

    if out is not None:
        output_path = markdown_path or html_path
        if output_path is not None:
            _copy_report_asset_bundle(run_path, output_path.parent)

    return ReportOutputs(markdown=markdown_path, html=html_path)


def _read_existing_report_markdown(run_path: Path) -> str | None:
    path = run_path / "research_report.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _markdown_output_path(run_path: Path, output_format: str, out: str | Path | None) -> Path:
    if out is None:
        return run_path / "research_report.md"
    out_path = Path(out)
    if output_format == "all" and out_path.suffix.lower() == ".html":
        return out_path.with_suffix(".md")
    return out_path


def _html_output_path(run_path: Path, output_format: str, out: str | Path | None) -> Path:
    if out is None:
        return run_path / "research_report.html"
    out_path = Path(out)
    if output_format == "html":
        return out_path
    return out_path if out_path.suffix.lower() == ".html" else out_path.with_suffix(".html")


def _copy_report_asset_bundle(run_path: Path, output_dir: Path) -> None:
    source = run_path / "report_assets"
    if not source.exists():
        return
    destination = output_dir / "report_assets"
    if source.resolve() == destination.resolve():
        return
    shutil.copytree(source, destination, dirs_exist_ok=True)


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

    if _requires_oos_trade_evidence(spec_dict) and _as_finite_float(metrics.get("oos_trade_count")) is None:
        return "NO EVIDENCE"

    if _has_no_trade_evidence(metrics):
        return "NO EVIDENCE"

    if _robustness_configured(spec_dict) and robustness_result is None:
        return "WATCHLIST"

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


def _has_no_trade_evidence(metrics: dict) -> bool:
    oos_trades = metrics.get("oos_trade_count")
    if oos_trades is not None:
        return _as_finite_float(oos_trades) == 0.0
    trades = metrics.get("trade_count")
    return _as_finite_float(trades) == 0.0


def _requires_oos_trade_evidence(spec_dict: dict) -> bool:
    validation = spec_dict.get("validation", {})
    if not isinstance(validation, dict):
        return False
    test_period = validation.get("test_period")
    return isinstance(test_period, list) and len(test_period) >= 2


def _robustness_configured(spec_dict: dict) -> bool:
    robustness = spec_dict.get("robustness", {})
    if not isinstance(robustness, dict):
        return False
    return bool(
        robustness.get("cost_multiplier")
        or robustness.get("parameter_perturbation")
        or robustness.get("regime_analysis")
    )


def _decision_summary(
    decision: str,
    metrics: dict,
    repro_audit: dict,
    bias_audit: dict,
    robustness_result: dict | None,
    benchmark_metrics: dict[str, float] | None,
    lang: str,
    *,
    robustness_configured: bool = True,
    benchmark_configured: bool = True,
    oos_evidence_required: bool = False,
) -> dict[str, list[str] | str]:
    supporting: list[str] = []
    risks: list[str] = []
    actions: list[str] = []
    zh = lang == "zh"

    if repro_audit.get("status") == "pass":
        supporting.append("复现性审计通过。" if zh else "Reproducibility audit passed.")
    else:
        risks.append("复现性审计未通过。" if zh else "Reproducibility audit did not pass.")

    fatal_count = int(bias_audit.get("fatal_count", 0) or 0)
    warning_count = int(bias_audit.get("warning_count", 0) or 0)
    if fatal_count:
        risks.append(
            f"研究偏差审计存在 {fatal_count} 个致命问题。"
            if zh
            else f"Research bias audit has {fatal_count} fatal finding(s)."
        )
    else:
        supporting.append(
            f"研究偏差审计无致命问题（{warning_count} 个警告）。"
            if zh
            else f"Research bias audit has no fatal findings ({warning_count} warning(s))."
        )
    if warning_count:
        risks.append(
            f"研究偏差审计存在 {warning_count} 个需要跟进的警告。"
            if zh
            else f"Research bias audit has {warning_count} warning(s) that need follow-up."
        )
        actions.append(
            "复核研究偏差审计警告，并在升级前补齐证据。"
            if zh
            else "Review research-bias warnings and close the evidence gap before promotion."
        )

    oos_sharpe = _finite_metric(metrics, "oos_sharpe_ratio")
    if oos_sharpe is not None:
        if oos_sharpe > 0:
            supporting.append(
                f"OOS 夏普为正（{_format_float(oos_sharpe)}）。"
                if zh
                else f"OOS Sharpe is positive at {_format_float(oos_sharpe)}."
            )
        else:
            risks.append(
                f"OOS 夏普不为正（{_format_float(oos_sharpe)}）。"
                if zh
                else f"OOS Sharpe is not positive ({_format_float(oos_sharpe)})."
            )

    oos_trades = metrics.get("oos_trade_count")
    if _as_finite_float(oos_trades) == 0.0:
        risks.append(
            "OOS 交易次数为 0，因此没有可执行的样本外证据。"
            if zh
            else "OOS trade count is 0, so the experiment has no executable out-of-sample evidence."
        )
        actions.append(
            "先生成有足够 OOS 交易样本的 run，再考虑资金配置。"
            if zh
            else "Generate a run with enough OOS trades before considering capital allocation."
        )
    elif oos_trades is not None:
        supporting.append(f"OOS 交易次数为 {oos_trades}。" if zh else f"OOS trade count is {oos_trades}.")
    elif oos_evidence_required:
        risks.append(
            "缺少 OOS 交易次数，不能评估样本外交易证据。"
            if zh
            else "OOS trade count is missing, so out-of-sample trade evidence cannot be assessed."
        )
        actions.append(
            "重新生成包含 OOS 交易统计的 run，再考虑资金配置。"
            if zh
            else "Regenerate the run with OOS trade statistics before considering capital allocation."
        )

    robustness_status = str((robustness_result or {}).get("status", "missing")).upper()
    if robustness_result is None:
        if robustness_configured:
            risks.append("缺少已验证的稳健性结果。" if zh else "Verified robustness artifact is missing.")
    elif robustness_status in {"ROBUST", "PASS"}:
        supporting.append("稳健性检查通过。" if zh else "Robustness checks passed.")
    elif robustness_status == "WARN":
        risks.append("稳健性检查存在需要跟进的警告。" if zh else "Robustness checks have warnings that need follow-up.")
    elif robustness_status == "FRAGILE":
        risks.append("稳健性检查将策略判定为 fragile。" if zh else "Robustness checks classify the strategy as fragile.")
    elif robustness_status == "ERROR":
        risks.append("稳健性检查出错，不能支持升级。" if zh else "Robustness checks errored and cannot support promotion.")

    if benchmark_metrics is None:
        if not benchmark_configured:
            pass
        else:
            risks.append("无法计算相对基准收益。" if zh else "Benchmark-relative return could not be computed.")
    else:
        excess = benchmark_metrics["excess_total_return"]
        if excess > 0:
            supporting.append(
                f"总收益跑赢基准 {_format_percent(excess)}。"
                if zh
                else f"Total return exceeds benchmark by {_format_percent(excess)}."
            )
        else:
            risks.append(
                f"总收益落后基准 {_format_percent(abs(excess))}。"
                if zh
                else f"Total return trails benchmark by {_format_percent(abs(excess))}."
            )

    primary, action = _decision_primary_and_action(decision, zh)
    actions.append(action)

    return {
        "primary_reason": primary,
        "supporting": supporting or [messages(lang)["decision_fallbacks"]["no_supporting"]],
        "risks": risks or [messages(lang)["decision_fallbacks"]["no_risks"]],
        "actions": actions,
    }


def _benchmark_artifact_trusted(repro_audit: dict) -> bool:
    checks = repro_audit.get("checks", [])
    if not isinstance(checks, list):
        return True
    benchmark_guard_ids = {
        "artifact_hashes",
        "data_manifest_hash",
        "environment_hash",
        "missing_files",
        "run_digest",
        "equity_hash",
        "benchmark_hash",
        "benchmark_curve_hash",
        "benchmark_equity_hash",
        "benchmark_prices_hash",
    }
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("id") not in benchmark_guard_ids:
            continue
        if check.get("severity") == "fatal" and check.get("status") == "fail":
            return False
    return True


def _runtime_artifacts_trusted(
    repro_audit: dict,
    *,
    require_compiled_plan_hash: bool = False,
    require_data_manifest_hash: bool = False,
) -> bool:
    checks = repro_audit.get("checks", [])
    if not isinstance(checks, list):
        return not (require_compiled_plan_hash or require_data_manifest_hash)
    runtime_guard_ids = {
        "artifact_hashes",
        "compiled_plan_hash",
        "data_manifest_hash",
        "run_digest",
    }
    passed_check_ids: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("id") not in runtime_guard_ids:
            continue
        if check.get("status") == "pass":
            passed_check_ids.add(str(check.get("id")))
        if check.get("severity") == "fatal" and check.get("status") == "fail":
            return False
    if require_compiled_plan_hash and "compiled_plan_hash" not in passed_check_ids:
        return False
    if require_data_manifest_hash and "data_manifest_hash" not in passed_check_ids:
        return False
    return True


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


def _format_decision_summary_lines(summary: dict[str, list[str] | str], lang: str) -> list[str]:
    sections = messages(lang)["decision_sections"]
    lines = [
        f"### {sections['rationale']}",
        "",
        f"- **{sections['primary_reason']}**: {summary['primary_reason']}",
        "",
        f"### {sections['supporting']}",
        "",
    ]
    lines.extend(f"- {item}" for item in _as_string_list(summary.get("supporting")))
    lines.extend(["", f"### {sections['risks']}", ""])
    lines.extend(f"- {item}" for item in _as_string_list(summary.get("risks")))
    lines.extend(["", f"### {sections['actions']}", ""])
    lines.extend(f"- {item}" for item in _as_string_list(summary.get("actions")))
    return lines


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value is not None else []


def _format_key_metric_snapshot_lines(metrics: dict, decision: str, lang: str) -> list[str]:
    labels = messages(lang)["key_metrics"]
    rows = [
        (labels["decision"], decision),
        (labels["total_return"], _format_percent(metrics.get("total_return"))),
        (labels["annualized_return"], _format_percent(metrics.get("annualized_return"))),
        (labels["sharpe_ratio"], _format_float(metrics.get("sharpe_ratio"))),
        (labels["max_drawdown"], _format_percent(metrics.get("max_drawdown"))),
        (labels["oos_sharpe_ratio"], _format_float(metrics.get("oos_sharpe_ratio"))),
        (labels["oos_trade_count"], str(metrics.get("oos_trade_count", "N/A"))),
        (labels["cost_paid"], _format_money(metrics.get("cost_paid"))),
    ]
    return _format_metric_table_lines(rows, lang)


def _format_date_fact_lines(facts: ReportFacts, lang: str) -> list[str]:
    labels = messages(lang)["labels"]
    return [
        f"- **{labels['configured_end_date']}**: {facts.configured_end_date or 'N/A'}",
        f"- **{labels['effective_last_trading_day']}**: {facts.effective_last_trading_day or 'N/A'}",
    ]


def _format_main_metric_lines(metrics: dict, lang: str) -> list[str]:
    labels = messages(lang)["metric_labels"]
    rows = [
        (labels["total_return"], _format_percent(metrics.get("total_return"))),
        (labels["annualized_return"], _format_percent(metrics.get("annualized_return"))),
        (labels["annualized_volatility"], _format_percent(metrics.get("annualized_volatility"))),
        (labels["max_drawdown"], _format_percent(metrics.get("max_drawdown"))),
        (labels["sharpe_ratio"], _format_float(metrics.get("sharpe_ratio"))),
        (labels["sortino_ratio"], _format_float(metrics.get("sortino_ratio"))),
        (labels["calmar_ratio"], _format_float(metrics.get("calmar_ratio"))),
        (labels["trade_count"], str(metrics.get("trade_count", 0))),
        (labels["cost_paid"], _format_money(metrics.get("cost_paid"))),
    ]
    return _format_metric_table_lines(rows, lang)


def _format_metric_table_lines(rows: list[tuple[str, str]], lang: str) -> list[str]:
    metric_header, value_header = _metric_table_headers(lang)
    return [f"| {metric_header} | {value_header} |", "|--------|-------|", *[f"| {name} | {value} |" for name, value in rows]]


def _benchmark_relative_metrics(run_path: Path) -> dict[str, float] | None:
    equity_values = _curve_values_by_date(run_path / "equity_curve.csv")
    benchmark_values = _curve_values_by_date(run_path / "benchmark_curve.csv")
    if not equity_values or not benchmark_values:
        return None
    common_dates = sorted(set(equity_values).intersection(benchmark_values))
    if len(common_dates) < 2:
        return None
    first_date = common_dates[0]
    last_date = common_dates[-1]
    equity_start = equity_values[first_date]
    benchmark_start = benchmark_values[first_date]
    if equity_start == 0 or benchmark_start == 0:
        return None
    equity_return = equity_values[last_date] / equity_start - 1.0
    benchmark_return = benchmark_values[last_date] / benchmark_start - 1.0
    return {
        "strategy_total_return": equity_return,
        "benchmark_total_return": benchmark_return,
        "excess_total_return": equity_return - benchmark_return,
    }


def _curve_values_by_date(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    values: dict[str, float] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date = _normalize_curve_date(row.get("date"))
                value = _as_finite_float(row.get("value"))
                if date and value is not None:
                    values[date] = value
    except OSError:
        return None
    return values or None


def _normalize_curve_date(value: object) -> str:
    raw = str(value or "").strip()
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    return raw


def _format_benchmark_metric_lines(metrics: dict[str, float], lang: str) -> list[str]:
    labels = messages(lang)["benchmark_metrics"]
    rows = [
        (labels["strategy_total_return"], _format_percent(metrics.get("strategy_total_return"))),
        (labels["benchmark_total_return"], _format_percent(metrics.get("benchmark_total_return"))),
        (labels["excess_total_return"], _format_percent(metrics.get("excess_total_return"))),
    ]
    return _format_metric_table_lines(rows, lang)


def _metric_table_headers(lang: str) -> tuple[str, str]:
    return ("指标", "数值") if lang == "zh" else ("Metric", "Value")


def _decision_primary_and_action(decision: str, zh: bool) -> tuple[str, str]:
    if zh:
        if decision == "REJECT":
            return "在阻碍因素修复前，不应继续推进资金投入。", "修复失败证据项后重新生成报告。"
        if decision == "NO EVIDENCE":
            return "该 run 没有足够交易证据支撑资金决策。", "重新设计实验或扩展样本后再解读绩效。"
        if decision == "WATCHLIST":
            return "该 run 有可用证据，但仍有未解决风险，暂不适合升级。", "处理警告、重跑稳健性检查，并复核相对基准表现。"
        return "该 run 通过基础检查，可进入模拟交易评估。", "先进行带监控的模拟交易，再考虑真实资金。"
    if decision == "REJECT":
        return (
            "Do not continue toward capital deployment until blocking risks are fixed.",
            "Fix the failing evidence items and rerun the report.",
        )
    if decision == "NO EVIDENCE":
        return (
            "The run does not contain enough trade evidence to support a capital decision.",
            "Redesign the experiment or extend the sample before interpreting performance.",
        )
    if decision == "WATCHLIST":
        return (
            "The run has usable evidence, but unresolved risks prevent promotion.",
            "Resolve warnings, rerun robustness checks, and compare against benchmark-relative results.",
        )
    return (
        "The run clears baseline checks and may proceed to paper trading evaluation.",
        "Paper trade with monitoring before any live capital allocation.",
    )


def _format_percent(value: object) -> str:
    parsed = _as_finite_float(value)
    return "N/A" if parsed is None else f"{parsed:.2%}"


def _format_float(value: object) -> str:
    parsed = _as_finite_float(value)
    return "N/A" if parsed is None else f"{parsed:.2f}"


def _format_money(value: object) -> str:
    parsed = _as_finite_float(value)
    return "N/A" if parsed is None else f"${parsed:.2f}"


def _format_metric_assumption_lines(metrics: dict, spec: StrategySpec, lang: str) -> list[str]:
    del spec
    labels = messages(lang)["metric_assumptions"]
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
    lines = [f"- **{labels['profile']}**: {_format_assumption_value(profile)}"]
    for key in ("return_type", "risk_free_rate", "annualization_days", "calmar_denominator", "evaluation_window"):
        value = assumptions.get(key)
        formatted = _format_percent(value) if key == "risk_free_rate" else _format_assumption_value(value)
        lines.append(f"- **{labels[key]}**: {formatted}")
    if profile != "open_xquant_default":
        lines.append(f"- **{labels['non_default_note']}**")
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


def _format_is_oos_metric_lines(metrics: dict, lang: str) -> list[str]:
    labels = messages(lang)["is_oos_metrics"]
    rows = [
        (labels["is_total_return"], _format_percent(metrics.get("is_total_return"))),
        (labels["is_annualized_return"], _format_percent(metrics.get("is_annualized_return"))),
        (labels["is_annualized_volatility"], _format_percent(metrics.get("is_annualized_volatility"))),
        (labels["is_max_drawdown"], _format_percent(metrics.get("is_max_drawdown"))),
        (labels["is_sharpe_ratio"], _format_float(metrics.get("is_sharpe_ratio"))),
        (labels["is_calmar_ratio"], _format_float(metrics.get("is_calmar_ratio"))),
        (labels["oos_total_return"], _format_percent(metrics.get("oos_total_return"))),
        (labels["oos_annualized_return"], _format_percent(metrics.get("oos_annualized_return"))),
        (labels["oos_annualized_volatility"], _format_percent(metrics.get("oos_annualized_volatility"))),
        (labels["oos_max_drawdown"], _format_percent(metrics.get("oos_max_drawdown"))),
        (labels["oos_sharpe_ratio"], _format_float(metrics.get("oos_sharpe_ratio"))),
        (labels["oos_calmar_ratio"], _format_float(metrics.get("oos_calmar_ratio"))),
        (labels["oos_trade_count"], str(metrics.get("oos_trade_count", "N/A"))),
    ]
    return _format_metric_table_lines(rows, lang)


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
    rebalance = assumptions.get("rebalance")
    if isinstance(rebalance, dict):
        lines.append(f"- **rebalance.frequency**: {_format_assumption_value(rebalance.get('frequency'))}")
        lines.append(f"- **rebalance.interval_days**: {_format_assumption_value(rebalance.get('interval_days'))}")
        lines.append(f"- **rebalance.source**: {_format_assumption_value(rebalance.get('source'))}")
    return lines


def _format_runtime_disclosure_lines(
    compiled_plan: dict,
    data_manifest: dict,
    lang: str,
    *,
    artifacts_trusted: bool = True,
) -> list[str]:
    if not compiled_plan and not data_manifest:
        return []
    labels = messages(lang)["runtime_disclosure"]
    if not artifacts_trusted:
        return [labels["runtime_artifacts_untrusted"], labels["non_comparable"]]
    execution = compiled_plan.get("execution")
    cost = compiled_plan.get("cost")
    data = compiled_plan.get("data")
    lines = [labels["compiled_plan_source"] if compiled_plan else labels["compiled_plan_missing"]]
    if isinstance(execution, dict):
        rebalance = execution.get("rebalance")
        lines.append(f"- **runtime.fill_price_mode**: {_format_assumption_value(execution.get('fill_price_mode'))}")
        if isinstance(rebalance, dict):
            lines.append(f"- **runtime.rebalance.interval_days**: {_format_assumption_value(rebalance.get('interval_days'))}")
    if isinstance(cost, dict):
        lines.append(f"- **runtime.fee_rate**: {_format_percent(cost.get('fee_rate'))}")
        lines.append(f"- **runtime.slippage_rate**: {_format_percent(cost.get('slippage_rate'))}")
    data_source = data if isinstance(data, dict) else data_manifest
    if isinstance(data_source, dict):
        min_start = data_source.get("min_start_date") or data_manifest.get("min_start_date")
        effective_dir = data_source.get("effective_data_dir") or data_manifest.get("effective_data_dir")
        warmup_policy = data_manifest.get("warmup_policy") or ("preload_from_min_start_date" if min_start else "none_declared")
        lines.append(f"- **data.warmup_policy**: {_format_assumption_value(warmup_policy)}")
        lines.append(f"- **data.min_start_date**: {_format_assumption_value(min_start)}")
        lines.append(f"- **data.effective_data_dir**: {_format_assumption_value(effective_dir)}")
    lines.append(labels["non_comparable"])
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
