"""Optimize tools — parameter search, walk-forward, and cross-validation."""

from __future__ import annotations

import time
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from oxq.data.loaders import resolve_data_dir
from oxq.data.market import LocalMarketDataProvider
from oxq.optimize.paramset import ParameterSet
from oxq.optimize.search import GridSearch, _extract_metric
from oxq.optimize.validation import TimeSeriesCV
from oxq.optimize.walk_forward import WalkForward
from oxq.portfolio.analytics import RunResult
from oxq.tools import session
from oxq.tools.registry import registry
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import SimBroker
from oxq.trade.slippage import PercentageSlippage
from oxq.universe.static import StaticUniverse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_paramset(
    name: str,
    params: list[dict[str, Any]],
    constraints: list[str] | None = None,
) -> ParameterSet:
    """Build a ParameterSet from tool-friendly dicts.

    Each item in *params*: ``{"component": "sma_fast", "param": "period", "values": [5, 10, 20]}``
    """
    ps = ParameterSet(name)
    for p in params:
        ps.add(p["component"], p["param"], values=p["values"])
    for expr in constraints or []:
        ps.add_constraint(expr)
    return ps


def _broker_factory(
    fee_rate: float | None,
    fee_min: float | None,
    slippage_rate: float | None,
) -> Callable[[], SimBroker]:
    """Return a factory that creates fresh SimBroker instances."""
    def factory() -> SimBroker:
        fee_model = None
        if fee_rate is not None:
            fee_model = PercentageFee(
                rate=Decimal(str(fee_rate)),
                min_fee=Decimal(str(fee_min)) if fee_min is not None else Decimal("0"),
            )
        slippage_model = None
        if slippage_rate is not None:
            slippage_model = PercentageSlippage(rate=Decimal(str(slippage_rate)))
        return SimBroker(fee_model=fee_model, slippage_model=slippage_model)
    return factory


def _metrics_dict(rr: RunResult) -> dict[str, float]:
    """Extract standard metrics from a RunResult."""
    return {
        "total_return": float(rr.total_return()),
        "annualized_return": float(rr.annualized_return()),
        "annualized_volatility": float(rr.annualized_volatility()),
        "max_drawdown": float(rr.max_drawdown()),
        "sharpe_ratio": float(rr.sharpe_ratio()),
        "calmar_ratio": float(rr.calmar_ratio()),
        "sortino_ratio": float(rr.sortino_ratio()),
        "num_trades": len(rr.trades),
    }


# ---------------------------------------------------------------------------
# paramset_create
# ---------------------------------------------------------------------------


@registry.tool(
    name="paramset_create",
    description=(
        "Create a parameter search space for optimization. "
        "IMPORTANT: 'component' must exactly match names from strategy_inspect — "
        "indicator keys from signals[x].indicators, portfolio.type, or rule names."
    ),
)
def paramset_create(
    name: str,
    params: list[dict[str, Any]],
    strategy: str | None = None,
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    """Create a ParameterSet and store it in the session.

    Parameters
    ----------
    name : str
        Name for this parameter set.
    params : list[dict]
        Each dict: {"component": "sma_fast", "param": "period", "values": [5, 10, 20]}.
        component must match: indicator key in signal.indicators, portfolio.type
        (e.g. "TopNRanking"), or rule.name (e.g. "StopLossRule").
    strategy : str or None
        Optional strategy name to validate component names against.
    constraints : list[str] or None
        Constraint expressions, e.g. ["sma_fast.period < sma_slow.period"].
    """
    # Validate component names against strategy if provided
    if strategy:
        strat = session._strategies.get(strategy)
        if strat is not None:
            known: set[str] = set()
            for _sig_name, (sig, _p) in strat.signals.items():
                known.add(_sig_name)
                for ind_name in getattr(sig, "required_indicators", {}):
                    known.add(ind_name)
            portfolio_name = getattr(strat.portfolio, "name", None)
            if portfolio_name:
                known.add(portfolio_name)
            for rule in getattr(strat, "rules", getattr(strat, "_pending_rules", [])):
                rule_name = getattr(rule, "name", None)
                if rule_name:
                    known.add(rule_name)

            unmatched = [
                p["component"] for p in params if p["component"] not in known
            ]
            if unmatched:
                return {
                    "error": (
                        f"Component names {unmatched} do not match any indicator, "
                        f"portfolio, or rule in strategy '{strategy}'. "
                        f"Known names: {sorted(known)}. "
                        f"Use strategy_inspect to check exact names."
                    ),
                }

    try:
        ps = _build_paramset(name, params, constraints)
    except (ValueError, KeyError) as e:
        return {"error": str(e)}

    session._paramsets[name] = ps
    session._save()

    grid = ps.grid()
    return {
        "name": name,
        "total_combinations": ps.total_combinations,
        "valid_combinations": len(grid),
        "distributions": len(ps.distributions),
        "constraints": len(ps.constraints),
    }


# ---------------------------------------------------------------------------
# paramset_list / paramset_inspect
# ---------------------------------------------------------------------------


@registry.tool(
    name="paramset_list",
    description="List all parameter sets created in the current session",
)
def paramset_list() -> dict[str, Any]:
    """Return names and summaries of all parameter sets in session state."""
    items = []
    for name, ps in sorted(session._paramsets.items()):
        items.append({
            "name": name,
            "distributions": len(ps.distributions),
            "constraints": len(ps.constraints),
            "total_combinations": ps.total_combinations(),
        })
    return {"paramsets": items}


@registry.tool(
    name="paramset_inspect",
    description="Inspect a parameter set: distributions, constraints, and sample combinations",
)
def paramset_inspect(name: str) -> dict[str, Any]:
    """Return details of a stored ParameterSet."""
    ps = session._paramsets.get(name)
    if ps is None:
        return {"error": f"ParameterSet '{name}' not found"}

    grid = ps.grid()
    return {
        "name": ps.name,
        "total_combinations": ps.total_combinations,
        "valid_combinations": len(grid),
        "distributions": [
            {"component": d.component, "param": d.param, "values": list(d.values)}
            for d in ps.distributions
        ],
        "constraints": [c.expr for c in ps.constraints],
        "sample_combinations": grid[:5],
    }


# ---------------------------------------------------------------------------
# grid_search
# ---------------------------------------------------------------------------


@registry.tool(
    name="grid_search",
    description="Run exhaustive grid search over all parameter combinations and return ranked results",
)
def grid_search(
    strategy: str,
    paramset: str,
    start: str,
    end: str,
    symbols: list[str] | None = None,
    metric: str = "sharpe_ratio",
    metric_direction: str | None = None,
    initial_cash: float = 100_000.0,
    data_dir: str | None = None,
    fee_rate: float | None = None,
    fee_min: float | None = None,
    slippage_rate: float | None = None,
    lot_size: int = 1,
    cash_annual_return: float = 0.0,
    data_start: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Run GridSearch and return top results."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    ps = session._paramsets.get(paramset)
    if ps is None:
        return {"error": f"ParameterSet '{paramset}' not found"}

    run_universe = StaticUniverse(tuple(symbols)) if symbols else session._strategy_universes.get(strategy)
    if run_universe is None:
        run_universe = getattr(strat, "_legacy_universe", None)
    if run_universe is None or not getattr(run_universe, "symbols", ()):
        return {"error": "No universe set for this run. Use strategy_set_universe first, or pass symbols parameter."}

    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    market = LocalMarketDataProvider(path)
    factory = _broker_factory(fee_rate, fee_min, slippage_rate)

    # Collect pending rules from strategy
    rules = getattr(strat, "rules", getattr(strat, "_pending_rules", [])) or []

    try:
        result = GridSearch(ps).run(
            strategy=strat,
            market=market,
            broker_factory=factory,
            start=start,
            end=end,
            metric=metric,
            metric_direction=metric_direction,
            initial_cash=initial_cash,
            rules=rules,
            universe=run_universe,
            lot_size=lot_size,
            cash_annual_return=cash_annual_return,
            data_start=data_start,
        )
    except Exception as e:
        return {"error": str(e)}

    search_id = f"gs_{strategy}_{int(time.time())}"
    session._search_results[search_id] = result
    session._save()

    top = result.top_n(top_n)
    ranked = []
    for i, trial in enumerate(top, 1):
        ranked.append({
            "rank": i,
            "params": trial.params,
            "metric_value": round(trial.metric_value, 6),
            "metrics": _metrics_dict(trial.run_result),
        })

    return {
        "search_id": search_id,
        "total_trials": len(result.all_results),
        "metric": result.metric,
        "metric_direction": result.metric_direction,
        "top_results": ranked,
    }


# ---------------------------------------------------------------------------
# walk_forward
# ---------------------------------------------------------------------------


@registry.tool(
    name="walk_forward",
    description="Run walk-forward analysis with rolling or anchored windows for out-of-sample validation",
)
def walk_forward(
    strategy: str,
    paramset: str,
    symbols: list[str],
    start: str,
    end: str,
    train_period: str,
    test_period: str,
    anchored: bool = False,
    step: str | None = None,
    metric: str = "sharpe_ratio",
    metric_direction: str | None = None,
    initial_cash: float = 100_000.0,
    data_dir: str | None = None,
    fee_rate: float | None = None,
    fee_min: float | None = None,
    slippage_rate: float | None = None,
) -> dict[str, Any]:
    """Run walk-forward analysis and return per-window + aggregate results."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    ps = session._paramsets.get(paramset)
    if ps is None:
        return {"error": f"ParameterSet '{paramset}' not found"}

    run_universe = StaticUniverse(tuple(symbols))
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    market = LocalMarketDataProvider(path)
    factory = _broker_factory(fee_rate, fee_min, slippage_rate)

    try:
        wf = WalkForward(ps, train_period, test_period, step=step, anchored=anchored)
        result = wf.run(
            strategy=strat,
            market=market,
            broker_factory=factory,
            start=start,
            end=end,
            metric=metric,
            metric_direction=metric_direction,
            initial_cash=initial_cash,
            universe=run_universe,
        )
    except Exception as e:
        return {"error": str(e)}

    wf_id = f"wf_{strategy}_{int(time.time())}"
    session._wf_results[wf_id] = result
    session._save()

    windows = []
    for i, w in enumerate(result.windows, 1):
        windows.append({
            "window": i,
            "train": f"{w.train_start} ~ {w.train_end}",
            "test": f"{w.test_start} ~ {w.test_end}",
            "best_params": w.best_params,
            "in_sample_metric": round(w.in_sample_metric, 6),
            "oos_metrics": _metrics_dict(w.oos_result),
        })

    return {
        "wf_id": wf_id,
        "mode": "anchored" if anchored else "rolling",
        "num_windows": len(result.windows),
        "metric": result.metric,
        "oos_summary": {
            "total_return": round(result.oos_total_return(), 6),
            "sharpe_ratio": round(result.oos_sharpe_ratio(), 6),
            "max_drawdown": round(result.oos_max_drawdown(), 6),
        },
        "windows": windows,
    }


# ---------------------------------------------------------------------------
# cross_validate
# ---------------------------------------------------------------------------


@registry.tool(
    name="cross_validate",
    description="Run time-series cross-validation with expanding or sliding windows and optional optimization",
)
def cross_validate(
    strategy: str,
    symbols: list[str],
    start: str,
    end: str,
    n_splits: int = 5,
    expanding: bool = True,
    embargo_days: int = 0,
    paramset: str | None = None,
    metric: str = "sharpe_ratio",
    metric_direction: str | None = None,
    initial_cash: float = 100_000.0,
    data_dir: str | None = None,
    fee_rate: float | None = None,
    fee_min: float | None = None,
    slippage_rate: float | None = None,
) -> dict[str, Any]:
    """Run time-series cross-validation and return per-fold results."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    ps = None
    if paramset is not None:
        ps = session._paramsets.get(paramset)
        if ps is None:
            return {"error": f"ParameterSet '{paramset}' not found"}

    run_universe = StaticUniverse(tuple(symbols))
    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    market = LocalMarketDataProvider(path)
    factory = _broker_factory(fee_rate, fee_min, slippage_rate)

    try:
        cv = TimeSeriesCV(n_splits=n_splits, embargo_days=embargo_days, expanding=expanding)
        result = cv.cross_validate(
            strategy=strat,
            market=market,
            broker_factory=factory,
            start=start,
            end=end,
            paramset=ps,
            metric=metric,
            metric_direction=metric_direction,
            initial_cash=initial_cash,
            universe=run_universe,
        )
    except Exception as e:
        return {"error": str(e)}

    cv_id = f"cv_{strategy}_{int(time.time())}"
    session._cv_results[cv_id] = result
    session._save()

    folds = []
    for i, sr in enumerate(result.splits, 1):
        fold: dict[str, Any] = {
            "fold": i,
            "train": f"{sr.split.train_start} ~ {sr.split.train_end}",
            "test": f"{sr.split.test_start} ~ {sr.split.test_end}",
            "oos_metrics": _metrics_dict(sr.oos_result),
        }
        if sr.best_params is not None:
            fold["best_params"] = sr.best_params
            fold["in_sample_metric"] = round(sr.in_sample_metric, 6)
        folds.append(fold)

    return {
        "cv_id": cv_id,
        "mode": "expanding" if expanding else "sliding",
        "n_splits": len(result.splits),
        "embargo_days": embargo_days,
        "metric": result.metric,
        "mean_oos_metric": round(result.mean_oos_metric(), 6),
        "std_oos_metric": round(result.std_oos_metric(), 6),
        "folds": folds,
    }


# ---------------------------------------------------------------------------
# overfit_analysis
# ---------------------------------------------------------------------------


@registry.tool(
    name="overfit_analysis",
    description="Analyze overfitting by comparing in-sample GridSearch results with walk-forward OOS results",
)
def overfit_analysis(
    search_id: str,
    wf_id: str | None = None,
    cv_id: str | None = None,
) -> dict[str, Any]:
    """Compare IS vs OOS performance to quantify overfitting risk.

    Requires a grid_search result. Optionally compares with walk_forward
    and/or cross_validate results for a comprehensive overfitting report.
    """
    gs_result = session._search_results.get(search_id)
    if gs_result is None:
        return {"error": f"GridSearch result '{search_id}' not found"}

    best = gs_result.best
    report: dict[str, Any] = {
        "in_sample": {
            "best_params": best.params,
            "metric": gs_result.metric,
            "metric_value": round(best.metric_value, 6),
            "metrics": _metrics_dict(best.run_result),
        },
        "search_stats": {
            "total_trials": len(gs_result.all_results),
            "metric_direction": gs_result.metric_direction,
        },
    }

    # Parameter stability: check if top 5 params are similar
    top5 = gs_result.top_n(5)
    if len(top5) >= 2:
        # Collect all param values for each component.param
        param_ranges: dict[str, list] = {}
        for trial in top5:
            for comp, params in trial.params.items():
                for pname, pval in params.items():
                    key = f"{comp}.{pname}"
                    param_ranges.setdefault(key, []).append(pval)

        stability: dict[str, Any] = {}
        for key, values in param_ranges.items():
            try:
                nums = [float(v) for v in values]
                spread = max(nums) - min(nums)
                stability[key] = {
                    "min": min(nums),
                    "max": max(nums),
                    "spread": round(spread, 6),
                }
            except (TypeError, ValueError):
                stability[key] = {"values": values}

        report["parameter_stability"] = stability

    # Walk-Forward comparison
    if wf_id is not None:
        wf_result = session._wf_results.get(wf_id)
        if wf_result is None:
            return {"error": f"WalkForward result '{wf_id}' not found"}

        det = wf_result.deterioration()
        report["walk_forward"] = {
            "num_windows": len(wf_result.windows),
            "oos_summary": {
                "total_return": round(wf_result.oos_total_return(), 6),
                "sharpe_ratio": round(wf_result.oos_sharpe_ratio(), 6),
                "max_drawdown": round(wf_result.oos_max_drawdown(), 6),
            },
            "deterioration": {k: round(v, 4) for k, v in det.items()},
        }

    # CV comparison
    if cv_id is not None:
        cv_result = session._cv_results.get(cv_id)
        if cv_result is None:
            return {"error": f"CV result '{cv_id}' not found"}

        oos_metrics = [
            _extract_metric(sr.oos_result, cv_result.metric)
            for sr in cv_result.splits
        ]
        report["cross_validation"] = {
            "n_splits": len(cv_result.splits),
            "mean_oos_metric": round(cv_result.mean_oos_metric(), 6),
            "std_oos_metric": round(cv_result.std_oos_metric(), 6),
            "oos_per_fold": [round(v, 6) for v in oos_metrics],
        }

    # Overfitting verdict
    verdicts = []
    is_metric = best.metric_value

    if wf_id and "walk_forward" in report:
        wf_oos = report["walk_forward"]["oos_summary"].get("sharpe_ratio", 0)
        if gs_result.metric == "sharpe_ratio" and abs(is_metric) > 1e-6:
            decay = (wf_oos - is_metric) / abs(is_metric)
            if decay < -0.5:
                verdicts.append(f"HIGH overfitting risk: WF OOS Sharpe decays {decay:.0%} vs IS")
            elif decay < -0.2:
                verdicts.append(f"MODERATE overfitting risk: WF OOS Sharpe decays {decay:.0%} vs IS")
            else:
                verdicts.append(f"LOW overfitting risk: WF OOS Sharpe decays {decay:.0%} vs IS")

    if cv_id and "cross_validation" in report:
        cv_mean = report["cross_validation"]["mean_oos_metric"]
        cv_std = report["cross_validation"]["std_oos_metric"]
        if cv_std > 0 and abs(cv_mean) > 1e-6:
            cv_coeff_var = cv_std / abs(cv_mean)
            if cv_coeff_var > 1.0:
                verdicts.append(f"UNSTABLE: CV metric varies widely (CoV={cv_coeff_var:.2f})")
            elif cv_coeff_var > 0.5:
                verdicts.append(f"MODERATE stability: CV metric CoV={cv_coeff_var:.2f}")
            else:
                verdicts.append(f"STABLE: CV metric CoV={cv_coeff_var:.2f}")

    if not verdicts:
        verdicts.append("Run walk_forward or cross_validate for overfitting assessment")

    report["verdicts"] = verdicts

    return report
