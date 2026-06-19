"""Profile-aware portfolio metric calculations."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np

from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult

if TYPE_CHECKING:
    from oxq.spec.schema import MetricsSection


def compute_profile_metrics(result: RunResult, config: MetricsSection, *, run_id: str) -> dict[str, Any]:
    """Compute backtest metrics using the requested metrics profile assumptions."""
    config = _normalize_profile_defaults(config)
    assumptions = metric_assumptions(config)
    metrics = {
        "run_id": run_id,
        "metrics_profile": config.profile,
        "metric_assumptions": assumptions,
    }
    if config.return_type == "log":
        metrics.update(_log_return_metrics(result, config))
    else:
        metrics.update(_simple_return_metrics(result, config))
    return metrics


def metric_assumptions(config: MetricsSection) -> dict[str, Any]:
    """Return serializable metric assumptions for artifacts and reports."""
    config = _normalize_profile_defaults(config)
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
    config = _normalize_profile_defaults(config)
    if config.return_type == "log":
        return _log_curve_metrics(equity_curve, config)
    return _simple_curve_metrics(equity_curve, config)


def _normalize_profile_defaults(config: MetricsSection) -> MetricsSection:
    defaults = _profile_defaults(config.profile)
    if not defaults:
        return config
    explicit_fields = set(getattr(config, "_explicit_fields", set()))
    values = {
        "risk_free_rate": config.risk_free_rate,
        "return_type": config.return_type,
        "annualization_days": config.annualization_days,
        "calmar_denominator": config.calmar_denominator,
        "evaluation_window": config.evaluation_window,
    }
    changed = False
    for field_name, default_value in defaults.items():
        if field_name not in explicit_fields and values[field_name] != default_value:
            values[field_name] = default_value
            changed = True
    if not changed:
        return config
    return type(config)(profile=config.profile, _explicit_fields=explicit_fields, **values)


def _profile_defaults(profile: str) -> dict[str, object]:
    if profile == "xquant_production":
        return {
            "risk_free_rate": 0.02,
            "return_type": "log",
            "annualization_days": 252,
            "calmar_denominator": "max_drawdown",
            "evaluation_window": "full",
        }
    return {}


def _unavailable_curve_metrics() -> dict[str, None]:
    return {
        "total_return": None,
        "annualized_return": None,
        "annualized_volatility": None,
        "max_drawdown": None,
        "sharpe_ratio": None,
        "calmar_ratio": None,
    }


def _simple_return_metrics(result: RunResult, config: MetricsSection) -> dict[str, float | None]:
    values = _values(result.equity_curve)
    if len(values) < 2 and config.profile == "open_xquant_default":
        days = config.annualization_days
        return {
            "total_return": result.total_return(),
            "annualized_return": result.annualized_return(days),
            "annualized_volatility": result.annualized_volatility(days),
            "max_drawdown": result.max_drawdown(),
            "sharpe_ratio": result.sharpe_ratio(days),
            "sortino_ratio": result.sortino_ratio(config.risk_free_rate, days),
            "calmar_ratio": result.calmar_ratio(days),
        }
    if len(values) < 2 or np.any(values <= 0):
        metrics = _stable_simple_curve_metrics(result.equity_curve, config)
        if metrics is None:
            metrics = _unavailable_curve_metrics()
        metrics["sortino_ratio"] = None
        return metrics
    days = config.annualization_days
    metrics = _simple_curve_metrics(result.equity_curve, config)
    metrics["sortino_ratio"] = result.sortino_ratio(config.risk_free_rate, days)
    return metrics


def _simple_curve_metrics(equity_curve: list[tuple[object, float]], config: MetricsSection) -> dict[str, float | None]:
    values = _values(equity_curve)
    if len(values) < 2 or np.any(values <= 0):
        metrics = _stable_simple_curve_metrics(equity_curve, config)
        if metrics is not None:
            return metrics
        return _unavailable_curve_metrics()
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
        "sharpe_ratio": _simple_sharpe_ratio(values, config.risk_free_rate, config.annualization_days),
        "calmar_ratio": result.calmar_ratio(config.annualization_days),
    }


def _stable_simple_curve_metrics(
    equity_curve: list[tuple[object, float]],
    config: MetricsSection,
) -> dict[str, float | None] | None:
    values = _values(equity_curve)
    if len(values) < 2 or values[0] <= 0 or values[-1] < 0 or np.all(values > 0):
        return None
    result = RunResult(
        portfolio=Portfolio(cash=Decimal(str(values[-1]))),
        trades=[],
        equity_curve=equity_curve,
        mktdata={},
    )
    metrics = _unavailable_curve_metrics()
    metrics.update({
        "total_return": result.total_return(),
        "annualized_return": result.annualized_return(config.annualization_days),
        "max_drawdown": result.max_drawdown(),
        "calmar_ratio": result.calmar_ratio(config.annualization_days),
    })
    return metrics


def _log_return_metrics(result: RunResult, config: MetricsSection) -> dict[str, float | None]:
    curve_metrics = _log_curve_metrics(result.equity_curve, config)
    if curve_metrics["annualized_return"] is None:
        metrics = _unavailable_curve_metrics()
        metrics["sortino_ratio"] = None
        return metrics
    return {
        "total_return": curve_metrics["total_return"],
        "annualized_return": curve_metrics["annualized_return"],
        "annualized_volatility": curve_metrics["annualized_volatility"],
        "max_drawdown": curve_metrics["max_drawdown"],
        "sharpe_ratio": curve_metrics["sharpe_ratio"],
        "sortino_ratio": _log_sortino_ratio(result, config),
        "calmar_ratio": curve_metrics["calmar_ratio"],
    }


def _log_curve_metrics(equity_curve: list[tuple[object, float]], config: MetricsSection) -> dict[str, float | None]:
    values = _values(equity_curve)
    if len(values) < 2 or values[0] <= 0 or np.any(values <= 0):
        return _unavailable_curve_metrics()
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


def _simple_sharpe_ratio(values: np.ndarray, risk_free_rate: float, annualization_days: int) -> float | None:
    if len(values) < 2 or np.any(values <= 0):
        return None
    returns = np.diff(values) / values[:-1]
    return _sharpe_ratio(returns, risk_free_rate, annualization_days)


def _max_drawdown(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return float(np.min(drawdown))
