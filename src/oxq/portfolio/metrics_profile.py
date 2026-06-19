"""Profile-aware portfolio metric calculations."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import numpy as np

from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.schema import MetricsSection


def compute_profile_metrics(result: RunResult, config: MetricsSection, *, run_id: str) -> dict[str, Any]:
    """Compute backtest metrics using the requested metrics profile assumptions."""
    assumptions = metric_assumptions(config)
    metrics = {
        "run_id": run_id,
        "metrics_profile": config.profile,
        "metric_assumptions": assumptions,
        "total_return": result.total_return(),
        "max_drawdown": result.max_drawdown(),
    }
    if config.return_type == "log":
        metrics.update(_log_return_metrics(result, config))
    else:
        days = config.annualization_days
        metrics.update(
            {
                "annualized_return": result.annualized_return(days),
                "annualized_volatility": result.annualized_volatility(days),
                "sharpe_ratio": result.sharpe_ratio(days),
                "sortino_ratio": result.sortino_ratio(config.risk_free_rate, days),
                "calmar_ratio": result.calmar_ratio(days),
            }
        )
    return metrics


def metric_assumptions(config: MetricsSection) -> dict[str, Any]:
    """Return serializable metric assumptions for artifacts and reports."""
    return {
        "return_type": config.return_type,
        "risk_free_rate": config.risk_free_rate,
        "annualization_days": config.annualization_days,
        "calmar_denominator": config.calmar_denominator,
        "evaluation_window": config.evaluation_window,
    }


def compute_equity_curve_metrics(
    equity_curve: list[tuple[object, float]],
    config: MetricsSection,
) -> dict[str, float | None]:
    """Compute profile-aware metrics for a standalone equity-curve slice."""
    if config.return_type == "log":
        return _log_curve_metrics(equity_curve, config)
    return _simple_curve_metrics(equity_curve, config)


def _simple_curve_metrics(equity_curve: list[tuple[object, float]], config: MetricsSection) -> dict[str, float | None]:
    values = _values(equity_curve)
    if len(values) < 2 or values[0] <= 0 or np.any(values[:-1] <= 0):
        return {
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
            "calmar_ratio": None,
        }
    result = RunResult(
        portfolio=Portfolio(cash=Decimal(str(values[-1]))),
        trades=[],
        equity_curve=equity_curve,
        mktdata={},
    )
    return {
        "total_return": result.total_return(),
        "annualized_return": result.annualized_return(config.annualization_days),
        "annualized_volatility": result.annualized_volatility(config.annualization_days),
        "max_drawdown": result.max_drawdown(),
        "sharpe_ratio": result.sharpe_ratio(config.annualization_days),
        "calmar_ratio": result.calmar_ratio(config.annualization_days),
    }


def _log_return_metrics(result: RunResult, config: MetricsSection) -> dict[str, float]:
    curve_metrics = _log_curve_metrics(result.equity_curve, config)
    return {
        "annualized_return": _none_to_zero(curve_metrics["annualized_return"]),
        "annualized_volatility": _none_to_zero(curve_metrics["annualized_volatility"]),
        "sharpe_ratio": _none_to_zero(curve_metrics["sharpe_ratio"]),
        "sortino_ratio": _log_sortino_ratio(result, config),
        "calmar_ratio": _none_to_zero(curve_metrics["calmar_ratio"]),
    }


def _log_curve_metrics(equity_curve: list[tuple[object, float]], config: MetricsSection) -> dict[str, float | None]:
    values = _values(equity_curve)
    if len(values) < 2 or values[0] <= 0 or np.any(values <= 0):
        return {
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
            "calmar_ratio": None,
        }
    returns = np.diff(np.log(values))
    max_drawdown = _max_drawdown(values)
    annualized_return = float(np.mean(returns) * config.annualization_days)
    annualized_volatility = _annualized_volatility(returns, config.annualization_days)
    sharpe = _sharpe_ratio(returns, config.risk_free_rate, config.annualization_days)
    calmar = 0.0 if max_drawdown == 0.0 else annualized_return / abs(max_drawdown)
    return {
        "total_return": float((values[-1] - values[0]) / values[0]),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "calmar_ratio": float(calmar),
    }


def _log_sortino_ratio(result: RunResult, config: MetricsSection) -> float:
    values = _values(result.equity_curve)
    if len(values) < 2 or np.any(values <= 0):
        return 0.0
    returns = np.diff(np.log(values))
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_dev = float(np.sqrt(np.mean(downside**2)) * np.sqrt(config.annualization_days))
    if downside_dev == 0.0:
        return 0.0
    ann_ret = float(np.mean(returns) * config.annualization_days)
    return float((ann_ret - config.risk_free_rate) / downside_dev)


def _values(equity_curve: list[tuple[object, float]]) -> np.ndarray:
    return np.array([value for _, value in equity_curve], dtype=float)


def _annualized_volatility(returns: np.ndarray, annualization_days: int) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(annualization_days))


def _sharpe_ratio(returns: np.ndarray, risk_free_rate: float, annualization_days: int) -> float:
    if len(returns) == 0:
        return 0.0
    std = float(np.std(returns))
    if std == 0.0:
        return 0.0
    excess = np.mean(returns) - risk_free_rate / annualization_days
    return float(excess / std * np.sqrt(annualization_days))


def _max_drawdown(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return float(np.min(drawdown))


def _none_to_zero(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return float(value)
