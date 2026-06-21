"""Artifact-derived facts for human-written research reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from oxq.report.artifacts import RunArtifacts


@dataclass(frozen=True)
class ReportFacts:
    configured_end_date: str | None
    effective_last_trading_day: str | None
    monthly_returns: list[dict[str, Any]]
    positive_month_count: int
    negative_month_count: int
    oos_trade_count: int | None
    trade_contribution: list[dict[str, Any]]
    known_numbers: list[dict[str, float | str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured_end_date": self.configured_end_date,
            "effective_last_trading_day": self.effective_last_trading_day,
            "monthly_returns": self.monthly_returns,
            "positive_month_count": self.positive_month_count,
            "negative_month_count": self.negative_month_count,
            "oos_trade_count": self.oos_trade_count,
            "trade_contribution": self.trade_contribution,
            "known_numbers": self.known_numbers,
        }


def build_report_facts(artifacts: RunArtifacts) -> ReportFacts:
    """Compute report facts from run artifacts, never from report prose."""
    configured_end_date = _configured_end_date(artifacts)
    effective_last_trading_day = _effective_last_trading_day(artifacts.equity_curve)
    monthly_returns = _monthly_returns(artifacts.equity_curve)
    positive_month_count = sum(1 for item in monthly_returns if _finite_float(item.get("return")) is not None and item["return"] > 0)
    negative_month_count = sum(1 for item in monthly_returns if _finite_float(item.get("return")) is not None and item["return"] < 0)
    oos_trade_count = _oos_trade_count(artifacts)
    known_numbers = _known_numbers(
        artifacts,
        monthly_returns=monthly_returns,
        positive_month_count=positive_month_count,
        negative_month_count=negative_month_count,
        oos_trade_count=oos_trade_count,
    )
    return ReportFacts(
        configured_end_date=configured_end_date,
        effective_last_trading_day=effective_last_trading_day,
        monthly_returns=monthly_returns,
        positive_month_count=positive_month_count,
        negative_month_count=negative_month_count,
        oos_trade_count=oos_trade_count,
        trade_contribution=[],
        known_numbers=known_numbers,
    )


def _configured_end_date(artifacts: RunArtifacts) -> str | None:
    validation = artifacts.strategy_spec.get("validation")
    if isinstance(validation, dict):
        test_period = validation.get("test_period")
        if isinstance(test_period, list) and len(test_period) >= 2:
            return _normalize_date_string(test_period[1])
    if artifacts.data_manifest and isinstance(artifacts.data_manifest.get("end"), str):
        return _normalize_date_string(artifacts.data_manifest["end"])
    return None


def _effective_last_trading_day(equity_curve: pd.DataFrame) -> str | None:
    if equity_curve.empty or "date" not in equity_curve.columns or "value" not in equity_curve.columns:
        return None
    frame = equity_curve.loc[:, ["date", "value"]].copy()
    frame["date"] = _parse_dates(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date")
    if frame.empty:
        return None
    last_date = frame["date"].iloc[-1]
    return last_date.isoformat() if isinstance(last_date, date) else None


def _monthly_returns(equity_curve: pd.DataFrame) -> list[dict[str, Any]]:
    if equity_curve.empty or "date" not in equity_curve.columns or "value" not in equity_curve.columns:
        return []
    frame = equity_curve.loc[:, ["date", "value"]].copy()
    frame["date"] = _parse_dates(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date")
    if frame.empty:
        return []
    frame["month"] = frame["date"].map(lambda value: value.strftime("%Y-%m"))
    monthly: list[dict[str, Any]] = []
    previous_close: float | None = None
    for month, group in frame.groupby("month", sort=True):
        first = _finite_float(group["value"].iloc[0])
        last = _finite_float(group["value"].iloc[-1])
        base = previous_close if previous_close is not None else first
        value = None if base is None or base == 0.0 or last is None else (last / base) - 1.0
        monthly.append({"month": str(month), "return": value})
        previous_close = last
    return monthly


def _oos_trade_count(artifacts: RunArtifacts) -> int | None:
    metric_value = _finite_float(artifacts.metrics.get("oos_trade_count"))
    if metric_value is not None:
        return int(metric_value)

    validation = artifacts.strategy_spec.get("validation")
    if not isinstance(validation, dict):
        return None
    test_period = validation.get("test_period")
    if not (isinstance(test_period, list) and len(test_period) >= 2):
        return None
    start = _parse_one_date(test_period[0])
    end = _parse_one_date(test_period[1])
    if start is None or end is None or "filled_at" not in artifacts.trades.columns:
        return None
    dates = _parse_dates(artifacts.trades["filled_at"]).dropna()
    return int(((dates >= start) & (dates <= end)).sum())


def _known_numbers(
    artifacts: RunArtifacts,
    *,
    monthly_returns: list[dict[str, Any]],
    positive_month_count: int,
    negative_month_count: int,
    oos_trade_count: int | None,
) -> list[dict[str, float | str]]:
    numbers: list[dict[str, float | str]] = []
    metrics = _metrics_with_default_assumptions(artifacts.metrics)
    _append_nested_numbers(numbers, "metric", metrics)
    _append_audit_numbers(numbers, artifacts)
    for item in monthly_returns:
        parsed = _finite_float(item.get("return"))
        if parsed is not None:
            numbers.append({"name": f"fact.monthly_return.{item['month']}", "value": parsed})
    numbers.extend(_benchmark_return_numbers(artifacts))
    numbers.extend(_strategy_spec_numbers(artifacts.strategy_spec))
    numbers.append({"name": "fact.positive_month_count", "value": float(positive_month_count)})
    numbers.append({"name": "fact.negative_month_count", "value": float(negative_month_count)})
    if oos_trade_count is not None:
        numbers.append({"name": "fact.oos_trade_count", "value": float(oos_trade_count)})
    return numbers


def _metrics_with_default_assumptions(metrics: dict[str, Any]) -> dict[str, Any]:
    copied = dict(metrics)
    if not isinstance(copied.get("metric_assumptions"), dict):
        copied["metric_assumptions"] = {
            "return_type": "simple",
            "risk_free_rate": 0.0,
            "annualization_days": 252,
            "calmar_denominator": "max_drawdown",
            "evaluation_window": "full",
        }
    return copied


def _append_audit_numbers(numbers: list[dict[str, float | str]], artifacts: RunArtifacts) -> None:
    if artifacts.research_bias_audit is not None:
        _append_nested_numbers(numbers, "audit.research_bias", artifacts.research_bias_audit)
    if artifacts.reproducibility_audit is not None:
        _append_nested_numbers(numbers, "audit.reproducibility", artifacts.reproducibility_audit)


def _benchmark_return_numbers(artifacts: RunArtifacts) -> list[dict[str, float | str]]:
    equity_values = _curve_values_by_date(artifacts.equity_curve)
    benchmark_values = _curve_values_by_date(artifacts.benchmark_curve)
    common_dates = sorted(set(equity_values).intersection(benchmark_values))
    if len(common_dates) < 2:
        return []
    first_date = common_dates[0]
    last_date = common_dates[-1]
    equity_start = equity_values[first_date]
    benchmark_start = benchmark_values[first_date]
    if equity_start == 0 or benchmark_start == 0:
        return []
    strategy_return = equity_values[last_date] / equity_start - 1.0
    benchmark_return = benchmark_values[last_date] / benchmark_start - 1.0
    return [
        {"name": "fact.strategy_total_return", "value": strategy_return},
        {"name": "fact.benchmark_total_return", "value": benchmark_return},
        {"name": "fact.excess_total_return", "value": strategy_return - benchmark_return},
    ]


def _curve_values_by_date(frame: pd.DataFrame) -> dict[date, float]:
    if frame.empty or "date" not in frame.columns or "value" not in frame.columns:
        return {}
    values: dict[date, float] = {}
    dates = _parse_dates(frame["date"])
    for parsed_date, raw_value in zip(dates, frame["value"], strict=False):
        parsed_value = _finite_float(raw_value)
        if parsed_date is not None and parsed_value is not None:
            values[parsed_date] = parsed_value
    return values


def _strategy_spec_numbers(strategy_spec: dict[str, Any]) -> list[dict[str, float | str]]:
    numbers: list[dict[str, float | str]] = []
    _append_nested_numbers(numbers, "spec", strategy_spec)
    execution = strategy_spec.get("execution")
    if not (isinstance(execution, dict) and "initial_cash" in execution):
        _append_number(numbers, "spec.execution.initial_cash", 100_000.0)
    return numbers


def _append_nested_numbers(numbers: list[dict[str, float | str]], name: str, value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key:
                _append_nested_numbers(numbers, f"{name}.{key}", child)
        return
    if isinstance(value, list):
        _append_number(numbers, f"{name}.count", len(value))
        for index, child in enumerate(value):
            _append_nested_numbers(numbers, f"{name}.{index}", child)
        return
    _append_number(numbers, name, value)


def _append_number(numbers: list[dict[str, float | str]], name: str, value: Any) -> None:
    parsed = _finite_float(value)
    if parsed is not None:
        if any(item.get("name") == name for item in numbers):
            return
        numbers.append({"name": name, "value": parsed})


def _parse_dates(values: pd.Series) -> pd.Series:
    return values.map(_parse_one_date)


def _parse_one_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    parsed_date = parsed.date()
    return parsed_date if isinstance(parsed_date, date) else None


def _normalize_date_string(value: Any) -> str | None:
    parsed = _parse_one_date(value)
    return parsed.isoformat() if parsed is not None else None


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
