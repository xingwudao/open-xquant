from __future__ import annotations

from oxq.report.generator import _determine_decision, _format_float, _format_money, _format_percent


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
