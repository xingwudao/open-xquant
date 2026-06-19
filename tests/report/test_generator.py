from __future__ import annotations

import json

import yaml

from oxq.report.generator import (
    _determine_decision,
    _format_float,
    _format_money,
    _format_percent,
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
