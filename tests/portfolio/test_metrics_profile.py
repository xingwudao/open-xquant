"""Tests for profile-aware portfolio metrics."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.portfolio.metrics_profile import compute_profile_metrics
from oxq.spec.schema import MetricsSection


def _make_result(values: list[float]) -> RunResult:
    dates = pd.bdate_range("2024-01-01", periods=len(values))
    return RunResult(
        portfolio=Portfolio(cash=Decimal(str(values[-1])) if values else Decimal("0")),
        trades=[],
        equity_curve=[(date, value) for date, value in zip(dates, values)],
        mktdata={},
    )


def test_open_xquant_default_profile_matches_existing_run_result_metrics() -> None:
    result = _make_result([100.0, 102.0, 99.0, 103.0, 97.0, 105.0])
    metrics = compute_profile_metrics(result, MetricsSection(), run_id="run-1")

    assert metrics["metrics_profile"] == "open_xquant_default"
    assert metrics["metric_assumptions"] == {
        "return_type": "simple",
        "risk_free_rate": 0.0,
        "annualization_days": 252,
        "calmar_denominator": "max_drawdown",
        "evaluation_window": "full",
    }
    assert metrics["total_return"] == pytest.approx(result.total_return())
    assert metrics["annualized_return"] == pytest.approx(result.annualized_return())
    assert metrics["annualized_volatility"] == pytest.approx(result.annualized_volatility())
    assert metrics["max_drawdown"] == pytest.approx(result.max_drawdown())
    assert metrics["sharpe_ratio"] == pytest.approx(result.sharpe_ratio())
    assert metrics["sortino_ratio"] == pytest.approx(result.sortino_ratio())
    assert metrics["calmar_ratio"] == pytest.approx(result.calmar_ratio())


@pytest.mark.parametrize("values", [[], [100.0]])
def test_open_xquant_default_profile_preserves_short_run_zero_metrics(values: list[float]) -> None:
    result = _make_result(values)

    metrics = compute_profile_metrics(result, MetricsSection(), run_id="run-1")

    assert metrics["metrics_profile"] == "open_xquant_default"
    assert metrics["total_return"] == 0.0
    assert metrics["annualized_return"] == 0.0
    assert metrics["annualized_volatility"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["sortino_ratio"] == 0.0
    assert metrics["calmar_ratio"] == 0.0


def test_xquant_production_profile_uses_log_returns_and_risk_free_rate() -> None:
    values = np.array([100.0, 102.0, 101.0, 106.0, 104.0, 109.0])
    result = _make_result(values.tolist())
    config = MetricsSection(
        profile="xquant_production",
        risk_free_rate=0.02,
        return_type="log",
        annualization_days=252,
        calmar_denominator="max_drawdown",
        evaluation_window="full",
    )

    metrics = compute_profile_metrics(result, config, run_id="run-1")

    log_returns = np.diff(np.log(values))
    expected_annualized_return = float(np.mean(log_returns) * 252)
    expected_sharpe = float((np.mean(log_returns) - 0.02 / 252) / np.std(log_returns) * np.sqrt(252))

    assert metrics["metrics_profile"] == "xquant_production"
    assert metrics["annualized_return"] == pytest.approx(expected_annualized_return)
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert metrics["sharpe_ratio"] != pytest.approx(result.sharpe_ratio())


def test_log_profile_keeps_invalid_equity_metrics_unavailable() -> None:
    result = _make_result([100.0, 0.0, 101.0])
    config = MetricsSection(profile="xquant_production", return_type="log", risk_free_rate=0.02)

    metrics = compute_profile_metrics(result, config, run_id="run-1")

    assert metrics["total_return"] is None
    assert metrics["annualized_return"] is None
    assert metrics["annualized_volatility"] is None
    assert metrics["max_drawdown"] is None
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["calmar_ratio"] is None


def test_default_profile_preserves_stable_metrics_when_equity_touches_zero() -> None:
    result = _make_result([100.0, 0.0, 101.0])

    metrics = compute_profile_metrics(result, MetricsSection(), run_id="run-1")

    assert metrics["total_return"] == pytest.approx(result.total_return())
    assert metrics["annualized_return"] == pytest.approx(result.annualized_return())
    assert metrics["annualized_volatility"] is None
    assert metrics["max_drawdown"] == pytest.approx(result.max_drawdown())
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["calmar_ratio"] == pytest.approx(result.calmar_ratio())


def test_default_profile_preserves_complete_loss_simple_metrics() -> None:
    result = _make_result([100.0, 0.0])

    metrics = compute_profile_metrics(result, MetricsSection(), run_id="run-1")

    assert metrics["total_return"] == pytest.approx(-1.0)
    assert metrics["annualized_return"] == pytest.approx(-1.0)
    assert metrics["annualized_volatility"] is None
    assert metrics["max_drawdown"] == pytest.approx(-1.0)
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["calmar_ratio"] == pytest.approx(-1.0)


def test_simple_profile_keeps_negative_endpoint_slice_unavailable() -> None:
    result = _make_result([100.0, -1.0])

    metrics = compute_profile_metrics(result, MetricsSection(), run_id="run-1")

    assert metrics["annualized_return"] is None
    assert metrics["sharpe_ratio"] is None


def test_in_memory_xquant_profile_applies_profile_defaults() -> None:
    values = np.array([100.0, 102.0, 101.0, 106.0, 104.0, 109.0])
    result = _make_result(values.tolist())
    config = MetricsSection()
    config.profile = "xquant_production"

    metrics = compute_profile_metrics(result, config, run_id="run-1")

    log_returns = np.diff(np.log(values))
    expected_sharpe = float((np.mean(log_returns) - 0.02 / 252) / np.std(log_returns) * np.sqrt(252))
    assert metrics["metric_assumptions"]["return_type"] == "log"
    assert metrics["metric_assumptions"]["risk_free_rate"] == 0.02
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe)


def test_in_memory_xquant_profile_preserves_programmatic_overrides() -> None:
    values = np.array([100.0, 102.0, 101.0, 106.0, 104.0, 109.0])
    result = _make_result(values.tolist())
    config = MetricsSection()
    config.profile = "xquant_production"
    config.return_type = "simple"
    config.risk_free_rate = 0.0

    metrics = compute_profile_metrics(result, config, run_id="run-1")

    assert metrics["metric_assumptions"]["return_type"] == "simple"
    assert metrics["metric_assumptions"]["risk_free_rate"] == 0.0
    assert metrics["sharpe_ratio"] == pytest.approx(result.sharpe_ratio())


def test_risk_free_rate_affects_simple_profile_sharpe() -> None:
    values = np.array([100.0, 102.0, 101.0, 106.0, 104.0, 109.0])
    result = _make_result(values.tolist())
    no_risk_free = compute_profile_metrics(result, MetricsSection(risk_free_rate=0.0), run_id="run-1")
    with_risk_free = compute_profile_metrics(result, MetricsSection(risk_free_rate=0.02), run_id="run-1")

    simple_returns = np.diff(values) / values[:-1]
    expected = float((np.mean(simple_returns) - 0.02 / 252) / np.std(simple_returns) * np.sqrt(252))

    assert no_risk_free["sharpe_ratio"] == pytest.approx(result.sharpe_ratio())
    assert with_risk_free["sharpe_ratio"] == pytest.approx(expected)
    assert with_risk_free["sharpe_ratio"] != pytest.approx(no_risk_free["sharpe_ratio"])


def test_annualization_days_affects_profile_metrics() -> None:
    result = _make_result([100.0, 102.0, 101.0, 106.0, 104.0, 109.0])
    standard = compute_profile_metrics(result, MetricsSection(annualization_days=252), run_id="run-1")
    short_year = compute_profile_metrics(result, MetricsSection(annualization_days=126), run_id="run-1")

    assert short_year["annualized_return"] != pytest.approx(standard["annualized_return"])
    assert short_year["annualized_volatility"] == pytest.approx(
        standard["annualized_volatility"] * np.sqrt(126 / 252)
    )
    assert short_year["metric_assumptions"]["annualization_days"] == 126
