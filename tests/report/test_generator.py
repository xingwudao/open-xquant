from __future__ import annotations

import json

import yaml

from oxq.report.generator import (
    _determine_decision,
    _format_float,
    _format_money,
    _format_percent,
    _format_validation_classification_lines,
    generate_report,
)
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


def test_decision_rejects_fragile_robustness_result() -> None:
    decision = _determine_decision(
        bias_audit={"fatal_count": 0, "warning_count": 0},
        spec_dict={},
        metrics={},
        repro_audit={"status": "pass"},
        robustness_result={"status": "fragile"},
    )

    assert decision == "REJECT"


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
            }
        ),
        encoding="utf-8",
    )

    report = generate_report(run_dir)

    assert "### Execution Assumptions" in report
    assert "- **order_timing**: next_session_open" in report
    assert "- **price_bar**: next_session" in report
    assert "- **price_type**: open" in report
    assert "- **fill_price_mode**: next_open" in report
    assert "- **cash_annual_return**: 2.50%" in report
    assert "- **default_lot_size**: 100" in report
    assert "- **calendar**: XSHE" in report
    assert "- **runtime_calendar**: XSHG" in report


def test_report_generation_does_not_fail_without_execution_assumptions(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)

    report = generate_report(run_dir)

    assert "# Research Report: report_execution_assumptions" in report
    assert "### Execution Assumptions" not in report


def test_report_generation_ignores_malformed_execution_assumptions(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    (run_dir / "execution_assumptions.json").write_text("{not-json", encoding="utf-8")

    report = generate_report(run_dir)

    assert "# Research Report: report_execution_assumptions" in report
    assert "## 5. Backtest Metrics" in report
    assert "### Execution Assumptions" not in report


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

    report = generate_report(run_dir)

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

    report = generate_report(run_dir)

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

    report = generate_report(run_dir)

    assert "- **Profile**: open_xquant_default" in report
    assert "- **return_type**: simple" in report
    assert "- **risk_free_rate**: 0.00%" in report
    assert "Non-default metrics profile" not in report


def test_report_includes_robustness_artifact_summary(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
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

    report = generate_report(run_dir)

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

    report = generate_report(run_dir)

    assert "## 1. Executive Decision\n\n**REJECT**" in report


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
