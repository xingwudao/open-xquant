from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from oxq.report.generator import (
    _benchmark_relative_metrics,
    _decision_summary,
    _determine_decision,
    _format_float,
    _format_money,
    _format_percent,
    _format_validation_classification_lines,
    generate_report,
)
from oxq.spec.compiler import _hash_file, _hash_json_file
from oxq.spec.schema import StrategySpec


def test_decision_rejects_when_reject_oos_metric_is_unavailable() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"reject_if": {"oos_sharpe_lt": 0.5}}},
        metrics={"oos_sharpe_ratio": None, "oos_max_drawdown": None},
    )

    assert decision == "REJECT"


def test_decision_rejects_fatal_audit_even_when_policy_disables_it() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 1, "warning_count": 0},
        spec_dict={"decision_policy": {"reject_if": {"fatal_audit_findings": False}}},
        metrics={"oos_sharpe_ratio": 2.0, "oos_max_drawdown": 0.0},
    )

    assert decision == "REJECT"


def test_decision_rejects_when_reject_oos_drawdown_is_missing() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"reject_if": {"max_drawdown_lt": -0.2}}},
        metrics={"max_drawdown": 0.0},
    )

    assert decision == "REJECT"


def test_decision_watchlists_when_promote_oos_metric_is_unavailable() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0}}},
        metrics={"oos_sharpe_ratio": None, "oos_max_drawdown": None},
    )

    assert decision == "WATCHLIST"


def test_decision_watchlists_when_promote_oos_drawdown_is_missing() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0, "max_drawdown_gte": -0.2}}},
        metrics={"oos_sharpe_ratio": 2.0, "max_drawdown": 0.0},
    )

    assert decision == "WATCHLIST"


def test_decision_watchlists_when_promote_oos_metric_is_below_threshold() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0}}},
        metrics={"oos_sharpe_ratio": 0.1},
    )

    assert decision == "WATCHLIST"


def test_decision_watchlists_when_configured_robustness_is_unverified() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={
            "robustness": {"cost_multiplier": [2.0]},
            "decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0}},
        },
        metrics={"trade_count": 12, "oos_trade_count": 12, "oos_sharpe_ratio": 2.0},
        repro_audit={"status": "pass"},
        robustness_result=None,
    )

    assert decision == "WATCHLIST"


def test_decision_rejects_fragile_robustness_result() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={},
        metrics={},
        repro_audit={"status": "pass"},
        robustness_result={"status": "fragile"},
    )

    assert decision == "REJECT"


def test_decision_rejects_unverified_robustness_result() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={},
        metrics={},
        repro_audit={"status": "fail", "checks": [{"id": "robustness_hash", "status": "fail"}]},
        robustness_result={"status": "robust"},
    )

    assert decision == "REJECT"


def test_decision_returns_no_evidence_when_oos_has_no_trades() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0}}},
        metrics={"trade_count": 20, "oos_trade_count": 0, "oos_sharpe_ratio": 3.0},
        repro_audit={"status": "pass"},
        robustness_result={"status": "warn", "tests": []},
    )

    assert decision == "NO EVIDENCE"


def test_decision_returns_no_evidence_when_oos_trade_count_is_missing_for_test_period() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 1},
        spec_dict={
            "validation": {
                "test_period": ["2024-01-01", "2024-12-31"],
                "required_oos": False,
            },
            "decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0}},
        },
        metrics={"trade_count": 25, "oos_sharpe_ratio": 3.0, "oos_max_drawdown": -0.05},
        repro_audit={"status": "pass"},
        robustness_result={"status": "warn", "tests": []},
    )

    assert decision == "NO EVIDENCE"


def test_decision_watchlists_warn_robustness_before_promotion() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0, "max_drawdown_gte": -0.2}}},
        metrics={"oos_sharpe_ratio": 2.0, "oos_max_drawdown": -0.05},
        repro_audit={"status": "pass"},
        robustness_result={"status": "warn"},
    )

    assert decision == "WATCHLIST"


def test_decision_promotes_when_robustness_warns_only_for_unconfigured_checks() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0, "max_drawdown_gte": -0.2}}},
        metrics={"oos_sharpe_ratio": 2.0, "oos_max_drawdown": -0.05},
        repro_audit={"status": "pass"},
        robustness_result={
            "status": "warn",
            "tests": [
                {
                    "name": "parameter_perturbation",
                    "status": "warn",
                    "message": "No parameter perturbation targets configured in spec",
                },
                {"name": "regime_analysis", "status": "warn", "message": "Regime analysis not configured"},
            ],
        },
    )

    assert decision == "PAPER TRADING CANDIDATE"


def test_decision_does_not_fallback_when_oos_metric_is_explicitly_unavailable() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0}}},
        metrics={"oos_sharpe_ratio": None, "sharpe_ratio": 99.0},
    )

    assert decision == "WATCHLIST"


def test_decision_does_not_fallback_when_oos_metric_is_missing() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": 1.0}}},
        metrics={"sharpe_ratio": 99.0},
    )

    assert decision == "WATCHLIST"


def test_decision_policy_threshold_strings_do_not_crash() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={"decision_policy": {"promote_if": {"oos_sharpe_gte": "0.5"}}},
        metrics={"oos_sharpe_ratio": 1.0},
    )

    assert decision == "PAPER TRADING CANDIDATE"


def test_metric_formatters_render_unavailable_values_as_na() -> None:
    assert _format_percent(None) == "N/A"
    assert _format_percent(float("nan")) == "N/A"
    assert _format_float(None) == "N/A"
    assert _format_money(None) == "N/A"


def test_validation_classification_reports_unclassified_failures() -> None:
    lines = _format_validation_classification_lines(
        {
            "status": "fail",
            "errors": [{"severity": "fatal", "check": "metrics_profile_unsupported", "dimensions": []}],
            "warnings": [],
        }
    )

    assert "- **status**: fail" in lines
    assert "- **unclassified**: fatal:metrics_profile_unsupported" in lines


def test_report_includes_decision_rationale_and_evidence(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    report = generate_report(run_dir, lang="en")

    assert "### Decision Rationale" in report
    assert "- **Primary reason**:" in report
    assert "### Supporting Evidence" in report
    assert "Research bias audit has no fatal findings" in report
    assert "### Blocking Risks" in report
    assert "### Recommended Next Actions" in report


def test_report_includes_benchmark_relative_metrics(monkeypatch, tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_benchmark_relative",
        hypothesis="benchmark relative results should be visible",
    )
    spec.benchmark.symbols = ["SPY"]
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-02,100\n2024-01-03,120\n",
        encoding="utf-8",
    )
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n2024-01-02,200\n2024-01-03,220\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "oxq.report.generator.audit_reproducibility",
        lambda run_dir: {"status": "pass", "checks": [], "fatal_count": 0, "warning_count": 0},
    )

    report = generate_report(run_dir, lang="en")

    assert "### Benchmark Relative Metrics" in report
    assert "| Strategy Total Return | 20.00% |" in report
    assert "| Benchmark Total Return | 10.00% |" in report
    assert "| Excess Total Return | 10.00% |" in report


def test_report_omits_benchmark_relative_metrics_when_benchmark_hash_fails(monkeypatch, tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_untrusted_benchmark",
        hypothesis="benchmark metrics should require a trusted benchmark artifact",
    )
    spec.benchmark.symbols = ["SPY"]
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-02,100\n2024-01-03,120\n",
        encoding="utf-8",
    )
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n2024-01-02,200\n2024-01-03,220\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "oxq.report.generator.audit_reproducibility",
        lambda run_dir: {
            "status": "fail",
            "checks": [
                {
                    "id": "benchmark_hash",
                    "severity": "fatal",
                    "status": "fail",
                    "message": "benchmark_curve.csv hash mismatch",
                }
            ],
            "fatal_count": 1,
            "warning_count": 0,
        },
    )

    report = generate_report(run_dir, lang="en")

    assert "### Benchmark Relative Metrics" not in report
    assert "Total return exceeds benchmark" not in report


def test_report_omits_benchmark_relative_metrics_when_artifact_hash_guard_fails(monkeypatch, tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_untrusted_benchmark_guard",
        hypothesis="benchmark metrics should require a trusted hash manifest",
    )
    spec.benchmark.symbols = ["SPY"]
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-02,100\n2024-01-03,120\n",
        encoding="utf-8",
    )
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n2024-01-02,200\n2024-01-03,220\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "oxq.report.generator.audit_reproducibility",
        lambda run_dir: {
            "status": "fail",
            "checks": [
                {
                    "id": "artifact_hashes",
                    "severity": "fatal",
                    "status": "fail",
                    "message": "artifact_hashes.json missing required keys: ['benchmark_curve.csv']",
                }
            ],
            "fatal_count": 1,
            "warning_count": 0,
        },
    )

    report = generate_report(run_dir, lang="en")

    assert "### Benchmark Relative Metrics" not in report
    assert "Total return exceeds benchmark" not in report


def test_report_omits_benchmark_relative_metrics_when_audit_files_are_missing(monkeypatch, tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_missing_hash_manifest",
        hypothesis="missing audit files should make benchmark evidence untrusted",
    )
    spec.benchmark.symbols = ["SPY"]
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-02,100\n2024-01-03,120\n",
        encoding="utf-8",
    )
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n2024-01-02,200\n2024-01-03,220\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "oxq.report.generator.audit_reproducibility",
        lambda run_dir: {
            "status": "fail",
            "checks": [
                {
                    "id": "missing_files",
                    "severity": "fatal",
                    "status": "fail",
                    "message": "Missing files: ['artifact_hashes.json']",
                }
            ],
            "fatal_count": 1,
            "warning_count": 0,
        },
    )

    report = generate_report(run_dir, lang="en")

    assert "### Benchmark Relative Metrics" not in report
    assert "Total return exceeds benchmark" not in report


def test_benchmark_relative_metrics_aligns_curve_dates(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-01,100\n2024-01-02,110\n2024-01-03,120\n",
        encoding="utf-8",
    )
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n2024-01-02,200\n2024-01-03,220\n",
        encoding="utf-8",
    )

    metrics = _benchmark_relative_metrics(run_dir)

    assert metrics is not None
    assert metrics["strategy_total_return"] == pytest.approx(120 / 110 - 1.0)
    assert metrics["benchmark_total_return"] == pytest.approx(220 / 200 - 1.0)
    assert metrics["excess_total_return"] == pytest.approx((120 / 110) - (220 / 200))


def test_benchmark_relative_metrics_requires_two_overlapping_dates(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-01,100\n2024-01-02,110\n",
        encoding="utf-8",
    )
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n2024-01-02,200\n2024-01-03,220\n",
        encoding="utf-8",
    )

    assert _benchmark_relative_metrics(run_dir) is None


def test_benchmark_relative_metrics_normalizes_timestamp_dates(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-02 00:00:00+08:00,100\n2024-01-03 00:00:00+08:00,120\n",
        encoding="utf-8",
    )
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n2024-01-02,200\n2024-01-03,220\n",
        encoding="utf-8",
    )

    metrics = _benchmark_relative_metrics(run_dir)

    assert metrics is not None
    assert metrics["strategy_total_return"] == pytest.approx(0.2)
    assert metrics["benchmark_total_return"] == pytest.approx(0.1)


def test_decision_summary_uses_chinese_empty_risk_fallback() -> None:
    summary = _decision_summary(
        "PAPER TRADING CANDIDATE",
        {
            "trade_count": 10,
            "oos_trade_count": 10,
            "oos_sharpe_ratio": 1.2,
        },
        repro_audit={"status": "pass"},
        bias_audit={"fatal_count": 0, "warning_count": 0},
        robustness_result={"status": "robust"},
        benchmark_metrics={
            "strategy_total_return": 0.2,
            "benchmark_total_return": 0.1,
            "excess_total_return": 0.1,
        },
        lang="zh",
    )

    assert summary["risks"] == ["配置检查未发现阻碍资金决策的风险。"]


def test_decision_summary_does_not_block_on_unconfigured_benchmark_or_robustness() -> None:
    summary = _decision_summary(
        "PAPER TRADING CANDIDATE",
        {
            "trade_count": 10,
            "oos_trade_count": 10,
            "oos_sharpe_ratio": 1.2,
        },
        repro_audit={"status": "pass"},
        bias_audit={"fatal_count": 0, "warning_count": 0},
        robustness_result=None,
        benchmark_metrics=None,
        lang="en",
        robustness_configured=False,
        benchmark_configured=False,
    )

    risks = "\n".join(summary["risks"])
    assert "Verified robustness artifact is missing" not in risks
    assert "Benchmark-relative return could not be computed" not in risks
    assert summary["risks"] == ["No blocking risks were detected by configured checks."]


def test_decision_summary_surfaces_research_audit_warnings_as_risks() -> None:
    summary = _decision_summary(
        "WATCHLIST",
        {
            "trade_count": 10,
            "oos_trade_count": 10,
            "oos_sharpe_ratio": 1.2,
        },
        repro_audit={"status": "pass"},
        bias_audit={"fatal_count": 0, "warning_count": 2},
        robustness_result=None,
        benchmark_metrics=None,
        lang="en",
        robustness_configured=False,
        benchmark_configured=False,
    )

    risks = "\n".join(summary["risks"])
    assert "Research bias audit has 2 warning(s)" in risks
    assert "No blocking risks were detected" not in risks


def test_report_includes_execution_assumptions_when_artifact_exists(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "execution_assumptions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calendar": "XSHE",
                "runtime_calendar": "XSHG",
                "order_timing": "next_session_open",
                "price_bar": "next_session",
                "price_type": "open",
                "fill_price_mode": "next_open",
                "cash_annual_return": 0.025,
                "lot_size": 1,
                "lot_size_config": {"default": 100, "by_symbol": {"SPY": 10}},
                "rebalance": {
                    "frequency": "daily",
                    "interval_days": 10,
                    "source": "portfolio.rules.rebalance",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "compiled_plan.json").write_text(
        json.dumps(
            {
                "execution": {
                    "fill_price_mode": "next_open",
                    "rebalance": {"interval_days": 10},
                },
                "cost": {
                    "fee_rate": 0.001,
                    "slippage_rate": 0.002,
                },
                "data": {
                    "effective_data_dir": "/tmp/oxq-data",
                    "min_start_date": "2023-12-01",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "warmup_policy": "preload_from_min_start_date",
                "min_start_date": "2023-12-01",
                "effective_data_dir": "/tmp/oxq-data",
            }
        ),
        encoding="utf-8",
    )
    _write_report_artifact_hashes(run_dir)

    report = generate_report(run_dir, lang="en")

    assert "### Execution Assumptions" in report
    assert "### Runtime Artifact Disclosure" in report
    assert "`compiled_plan.json`" in report
    assert "- **runtime.rebalance.interval_days**: 10" in report
    assert "- **runtime.fee_rate**: 0.10%" in report
    assert "- **runtime.slippage_rate**: 0.20%" in report
    assert "- **data.warmup_policy**: preload_from_min_start_date" in report
    assert "- **data.min_start_date**: 2023-12-01" in report
    assert "- **order_timing**: next_session_open" in report
    assert "- **price_bar**: next_session" in report
    assert "- **price_type**: open" in report
    assert "- **fill_price_mode**: next_open" in report
    assert "- **cash_annual_return**: 2.50%" in report
    assert "- **default_lot_size**: 100" in report
    assert "- **calendar**: XSHE" in report
    assert "- **runtime_calendar**: XSHG" in report
    assert "- **rebalance.frequency**: daily" in report
    assert "- **rebalance.interval_days**: 10" in report
    assert "- **rebalance.source**: portfolio.rules.rebalance" in report


def test_chinese_report_localizes_professional_metric_sections(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "execution_assumptions.json").write_text(
        json.dumps(
            {
                "order_timing": "next_session_open",
                "price_bar": "next_session",
                "price_type": "open",
                "fill_price_mode": "next_open",
                "cash_annual_return": 0.025,
            }
        ),
        encoding="utf-8",
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update(
        {
            "is_total_return": 0.06,
            "is_annualized_return": 0.12,
            "is_annualized_volatility": 0.10,
            "is_max_drawdown": -0.02,
            "is_sharpe_ratio": 1.45,
            "is_calmar_ratio": 6.0,
            "oos_total_return": 0.04,
            "oos_annualized_return": 0.08,
            "oos_annualized_volatility": 0.12,
            "oos_calmar_ratio": 2.67,
            "sortino_ratio": 1.4,
            "calmar_ratio": 1.6,
            "cost_paid": 3.0,
        }
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    report = generate_report(run_dir, lang="zh")

    assert "### 执行假设" in report
    assert "### 指标口径" in report
    assert "### IS/OOS 指标" in report
    assert "| 总收益 | 10.00% |" in report
    assert "| 样本内夏普比率 | 1.45 |" in report
    assert "| 样本外 Calmar 比率 | 2.67 |" in report
    assert "### Execution Assumptions" not in report
    assert "### Metrics Profile" not in report
    assert "| Metric | Value |" not in report


def test_chinese_report_localizes_runtime_disclosure(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "compiled_plan.json").write_text(
        json.dumps(
            {
                "execution": {"fill_price_mode": "next_open", "rebalance": {"interval_days": 10}},
                "cost": {"fee_rate": 0.001, "slippage_rate": 0.002},
                "data": {"effective_data_dir": "/tmp/oxq-data", "min_start_date": "2023-12-01"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "warmup_policy": "preload_from_min_start_date",
                "min_start_date": "2023-12-01",
                "effective_data_dir": "/tmp/oxq-data",
            }
        ),
        encoding="utf-8",
    )
    _write_report_artifact_hashes(run_dir)

    report = generate_report(run_dir, lang="zh")

    assert "### 运行产物披露" in report
    assert "已报告的运行执行语义来自 `compiled_plan.json`" in report
    assert "不同执行、成本或数据预热设置会导致收益结果不可直接比较" in report
    assert "### Runtime Artifact Disclosure" not in report
    assert "Reported execution semantics are taken" not in report


def test_report_generation_does_not_fail_without_execution_assumptions(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    report = generate_report(run_dir, lang="en")

    assert "# Research Report: report_execution_assumptions" in report
    assert "### Execution Assumptions" not in report


def test_report_generation_ignores_malformed_execution_assumptions(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "execution_assumptions.json").write_text("{not-json", encoding="utf-8")

    report = generate_report(run_dir, lang="en")

    assert "# Research Report: report_execution_assumptions" in report
    assert "## 5. Backtest Metrics" in report
    assert "### Execution Assumptions" not in report


def test_report_does_not_claim_compiled_plan_source_when_plan_missing(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "warmup_policy": "none_declared",
                "min_start_date": "",
                "effective_data_dir": "/tmp/oxq-data",
            }
        ),
        encoding="utf-8",
    )
    _write_report_artifact_hashes(run_dir)

    report = generate_report(run_dir, lang="en")

    assert "### Runtime Artifact Disclosure" in report
    assert "`compiled_plan.json` is missing or unreadable" in report
    assert "Reported execution semantics are taken from `compiled_plan.json`" not in report


def test_report_marks_runtime_disclosure_untrusted_when_runtime_hash_fails(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "compiled_plan.json").write_text(
        json.dumps(
            {
                "execution": {"fill_price_mode": "tampered", "rebalance": {"interval_days": 1}},
                "cost": {"fee_rate": 0.99, "slippage_rate": 0.99},
                "data": {"effective_data_dir": "/tmp/tampered", "min_start_date": "2020-01-01"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "warmup_policy": "preload_from_min_start_date",
                "min_start_date": "2020-01-01",
                "effective_data_dir": "/tmp/tampered",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "schema_version": 0,
                "compiled_plan.json": "sha256:stale-compiled-plan",
            }
        ),
        encoding="utf-8",
    )
    spec = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml")
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"open_xquant_version": "test", "spec_hash": spec.compute_hash()}),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text("date,equity\n2024-01-02,100000\n", encoding="utf-8")
    (run_dir / "trades.csv").write_text("date,symbol,quantity,price\n", encoding="utf-8")
    artifact_hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    artifact_hashes.update(
        {
            "data_manifest.json": _hash_json_file(run_dir / "data_manifest.json"),
            "equity_curve.csv": _hash_file(run_dir / "equity_curve.csv"),
            "trades.csv": _hash_file(run_dir / "trades.csv"),
            "metrics.json": _hash_json_file(run_dir / "metrics.json", exclude_keys={"run_id"}),
        }
    )
    (run_dir / "artifact_hashes.json").write_text(json.dumps(artifact_hashes), encoding="utf-8")

    report = generate_report(run_dir, lang="en")

    assert "Runtime semantics artifacts did not pass hash verification" in report
    assert "- **runtime.fill_price_mode**: tampered" not in report
    assert "- **runtime.fee_rate**: 99.00%" not in report
    assert "- **data.effective_data_dir**: /tmp/tampered" not in report


def test_report_does_not_display_compiled_plan_without_compiled_hash(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "compiled_plan.json").write_text(
        json.dumps(
            {
                "execution": {"fill_price_mode": "next_open", "rebalance": {"interval_days": 10}},
                "cost": {"fee_rate": 0.001, "slippage_rate": 0.002},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "effective_data_dir": "/tmp/oxq-data"}),
        encoding="utf-8",
    )
    spec = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml")
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"open_xquant_version": "test", "spec_hash": spec.compute_hash()}),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text("date,equity\n2024-01-02,100000\n", encoding="utf-8")
    (run_dir / "trades.csv").write_text("date,symbol,quantity,price\n", encoding="utf-8")
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "schema_version": 0,
                "data_manifest.json": _hash_json_file(run_dir / "data_manifest.json"),
                "equity_curve.csv": _hash_file(run_dir / "equity_curve.csv"),
                "trades.csv": _hash_file(run_dir / "trades.csv"),
                "metrics.json": _hash_json_file(run_dir / "metrics.json", exclude_keys={"run_id"}),
            }
        ),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "Runtime semantics artifacts did not pass hash verification" in report
    assert "- **runtime.rebalance.interval_days**: 10" not in report


def test_report_marks_runtime_disclosure_untrusted_when_run_digest_fails(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "compiled_plan.json").write_text(
        json.dumps(
            {
                "execution": {"fill_price_mode": "next_open", "rebalance": {"interval_days": 10}},
                "cost": {"fee_rate": 0.001, "slippage_rate": 0.002},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "effective_data_dir": "/tmp/oxq-data"}),
        encoding="utf-8",
    )
    _write_report_artifact_hashes(run_dir)
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps({"run_id": run_dir.name, "artifact_hashes": "sha256:stale"}) + "\n",
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "Runtime semantics artifacts did not pass hash verification" in report
    assert "- **runtime.rebalance.interval_days**: 10" not in report


def test_report_summary_uses_effective_execution_fill_mode(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_explicit_execution",
        hypothesis="report effective execution summary",
    )
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_close"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "close"
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "- **Execution**: next_open trade, next_close fill" in report


def test_report_includes_metric_assumptions_oos_and_validation_classification(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_metric_assumptions",
        hypothesis="report metric assumptions when present",
    )
    spec.metrics.profile = "xquant_production"
    spec.metrics.risk_free_rate = 0.02
    spec.metrics.return_type = "log"
    spec.metrics.annualization_days = 252
    spec.metrics.calmar_denominator = "max_drawdown"
    spec.metrics.evaluation_window = "full"
    spec.validation.train_period = ["2020-01-01", "2020-12-31"]
    spec.validation.test_period = ["2021-01-01", "2021-12-31"]
    spec.validation.required_oos = True
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "report-run",
                "metrics_profile": "xquant_production",
                "metric_assumptions": {
                    "return_type": "log",
                    "risk_free_rate": 0.02,
                    "annualization_days": 252,
                    "calmar_denominator": "max_drawdown",
                    "evaluation_window": "full",
                },
                "trade_count": 12,
                "oos_trade_count": 4,
                "is_total_return": 0.06,
                "is_annualized_return": 0.12,
                "is_annualized_volatility": 0.10,
                "is_max_drawdown": -0.02,
                "is_sharpe_ratio": 1.45,
                "is_calmar_ratio": 6.0,
                "max_drawdown": -0.05,
                "oos_max_drawdown": -0.03,
                "oos_total_return": 0.04,
                "oos_annualized_return": 0.08,
                "oos_annualized_volatility": 0.12,
                "oos_sharpe_ratio": 1.23,
                "oos_calmar_ratio": 2.67,
                "total_return": 0.1,
                "annualized_return": 0.08,
                "annualized_volatility": 0.12,
                "sharpe_ratio": 1.1,
                "sortino_ratio": 1.4,
                "calmar_ratio": 1.6,
                "cost_paid": 3.0,
            }
        ),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "### Metrics Profile" in report
    assert "- **Profile**: xquant_production" in report
    assert "- **return_type**: log" in report
    assert "- **risk_free_rate**: 2.00%" in report
    assert "- **annualization_days**: 252" in report
    assert "- **calmar_denominator**: max_drawdown" in report
    assert "- **evaluation_window**: full" in report
    assert "Non-default metrics profile" in report
    assert "### IS/OOS Metrics" in report
    assert "| IS Sharpe Ratio | 1.45 |" in report
    assert "| IS Calmar Ratio | 6.00 |" in report
    assert "| OOS Sharpe Ratio | 1.23 |" in report
    assert "| OOS Calmar Ratio | 2.67 |" in report
    assert "### Validation Classification" in report
    assert "- **conservative**:" in report


def test_report_missing_metric_assumptions_uses_legacy_defaults(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_legacy_metrics",
        hypothesis="legacy metrics artifacts should not inherit ignored spec metrics",
    )
    spec.metrics.profile = "xquant_production"
    spec.metrics.return_type = "log"
    spec.metrics.risk_free_rate = 0.02
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "- **Profile**: open_xquant_default" in report
    assert "- **return_type**: simple" in report
    assert "- **risk_free_rate**: 0.00%" in report
    assert "Non-default metrics profile" not in report


def test_report_includes_robustness_artifact_summary(tmp_path, monkeypatch) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps({"robustness.json": "sha256:unused"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "oxq.report.generator.audit_reproducibility",
        lambda *_args, **_kwargs: {
            "status": "pass",
            "checks": [
                {
                    "id": "robustness_hash",
                    "severity": "fatal",
                    "status": "pass",
                    "message": "robustness.json hash: OK",
                }
            ],
            "fatal_count": 0,
            "warning_count": 0,
        },
    )
    (run_dir / "robustness.json").write_text(
        json.dumps(
            {
                "status": "warn",
                "baseline_sharpe": 1.1,
                "tests": [
                    {
                        "name": "cost_x2",
                        "status": "pass",
                        "baseline_sharpe": 1.1,
                        "perturbed_sharpe": 0.9,
                        "message": "costs are stable",
                    },
                    {
                        "name": "parameter_perturbation",
                        "status": "warn",
                        "results": [
                            {"target": "mom.period", "status": "pass"},
                            {"target": "missing.period", "status": "error"},
                        ],
                        "message": "Ran 2 one-at-a-time parameter perturbations",
                    },
                    {
                        "name": "regime_analysis",
                        "status": "pass",
                        "regimes": {
                            "uptrend": {"date_count": 3, "trade_count": 1},
                            "downtrend": {"date_count": 2, "trade_count": 1},
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "**Status**: WARN" in report
    assert "- [PASS] **cost_x2**: costs are stable" in report
    assert "- **Parameter perturbation results**: pass=1, error=1" in report
    assert "- **Regimes**: uptrend (dates=3, trades=1), downtrend (dates=2, trades=1)" in report


def test_report_rejects_fragile_robustness_result(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_fragile_robustness",
        hypothesis="fragile robustness should block promotion",
    )
    spec.universe.point_in_time = True
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001
    spec.validation.train_period = ["2023-01-01", "2023-12-31"]
    spec.validation.test_period = ["2024-01-01", "2024-12-31"]
    spec.validation.required_oos = True
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "report-run",
                "trade_count": 12,
                "oos_trade_count": 12,
                "max_drawdown": -0.05,
                "oos_max_drawdown": -0.05,
                "oos_sharpe_ratio": 1.2,
                "total_return": 0.1,
                "annualized_return": 0.08,
                "annualized_volatility": 0.12,
                "sharpe_ratio": 1.1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "robustness.json").write_text(
        json.dumps({"status": "fragile", "baseline_sharpe": 1.1, "tests": []}),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "## 1. Executive Decision\n\n**REJECT**" in report


def test_report_does_not_trust_unhashed_robustness_artifact(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "artifact_hashes.json").write_text(json.dumps({"metrics.json": "sha256:unused"}), encoding="utf-8")
    (run_dir / "robustness.json").write_text(
        json.dumps({"status": "robust", "baseline_sharpe": 1.1, "tests": []}),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "## 1. Executive Decision\n\n**REJECT**" in report
    assert "**Status**: ROBUST" not in report


def test_report_does_not_trust_robustness_without_artifact_hashes(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "robustness.json").write_text(
        json.dumps({"status": "robust", "baseline_sharpe": 1.1, "tests": []}),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "**Status**: ROBUST" not in report
    assert "Robustness checks passed" not in report


def test_report_regime_only_config_does_not_claim_no_robustness_tests(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    spec = StrategySpec.template(
        strategy_id="report_regime_only_robustness",
        hypothesis="regime-only robustness config should be described consistently",
    )
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    spec.robustness.regime_analysis = True
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "- Regime analysis: enabled" in report
    assert "(No robustness tests configured)" not in report


def test_generated_report_discloses_configured_and_effective_dates(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-02,100\n2024-01-03,101\n",
        encoding="utf-8",
    )

    report = generate_report(run_dir, lang="en")

    assert "**Configured end date**: 2024-01-03" in report
    assert "**Effective last trading day**: 2024-01-03" in report


def _write_report_run(tmp_path):
    spec = StrategySpec.template(
        strategy_id="report_execution_assumptions",
        hypothesis="report execution assumptions when present",
    )
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "report-run",
                "trade_count": 12,
                "oos_trade_count": 12,
                "max_drawdown": -0.05,
                "oos_max_drawdown": -0.05,
                "oos_sharpe_ratio": 1.2,
                "total_return": 0.1,
                "annualized_return": 0.08,
                "annualized_volatility": 0.12,
                "sharpe_ratio": 1.1,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_report_artifact_hashes(run_dir: Path) -> None:
    spec = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml")
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"open_xquant_version": "test", "spec_hash": spec.compute_hash()}),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text("date,equity\n2024-01-02,100000\n", encoding="utf-8")
    (run_dir / "trades.csv").write_text("date,symbol,quantity,price\n", encoding="utf-8")
    artifact_hashes = {
        "schema_version": 0,
        "data_manifest.json": _hash_json_file(run_dir / "data_manifest.json"),
        "equity_curve.csv": _hash_file(run_dir / "equity_curve.csv"),
        "trades.csv": _hash_file(run_dir / "trades.csv"),
        "metrics.json": _hash_json_file(run_dir / "metrics.json", exclude_keys={"run_id"}),
    }
    if (run_dir / "compiled_plan.json").exists():
        artifact_hashes["compiled_plan.json"] = _hash_json_file(run_dir / "compiled_plan.json")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(artifact_hashes), encoding="utf-8")
