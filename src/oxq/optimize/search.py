"""Grid search parameter optimization."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from oxq.core.engine import Engine
from oxq.core.strategy import Strategy
from oxq.core.types import Broker
from oxq.data.providers import MarketDataProvider
from oxq.optimize.paramset import ParameterSet
from oxq.portfolio.analytics import RunResult
from oxq.universe.base import UniverseProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric direction registry
# ---------------------------------------------------------------------------

METRIC_DIRECTIONS: dict[str, str] = {
    "total_return": "maximize",
    "sharpe_ratio": "maximize",
    "annualized_return": "maximize",
    "calmar_ratio": "maximize",
    "sortino_ratio": "maximize",
    "max_drawdown": "maximize",  # negative value; maximize = least bad
    "annualized_volatility": "minimize",
}


def _extract_metric(
    result: RunResult,
    metric: str | Callable[[RunResult], float],
) -> float:
    """Extract a metric value from a RunResult."""
    if callable(metric) and not isinstance(metric, str):
        return metric(result)
    fn = getattr(result, metric, None)
    if fn is None or not callable(fn):
        msg = (
            f"Unknown metric {metric!r}. "
            f"Available: {list(METRIC_DIRECTIONS.keys())}"
        )
        raise ValueError(msg)
    return fn()


def _resolve_direction(
    metric: str | Callable[[RunResult], float],
    metric_direction: str | None,
) -> str:
    """Resolve the optimization direction for a metric."""
    if metric_direction is not None:
        if metric_direction not in ("maximize", "minimize"):
            msg = f"metric_direction must be 'maximize' or 'minimize', got {metric_direction!r}"
            raise ValueError(msg)
        return metric_direction
    if isinstance(metric, str) and metric in METRIC_DIRECTIONS:
        return METRIC_DIRECTIONS[metric]
    return "maximize"


def _apply_rule_params(
    rules: list[Any],
    params: dict[str, dict[str, Any]],
) -> list[Any]:
    """Deep-copy a list of rules, overriding attributes for matched names.

    Rules are matched by their ``name`` attribute. For each matched rule,
    a deep copy is made and the specified attributes are set via setattr.
    Unmatched rules are deep-copied unchanged.
    """
    result = []
    for rule in rules:
        rule_copy = copy.deepcopy(rule)
        rule_name = getattr(rule, "name", None)
        if rule_name and rule_name in params:
            for attr, value in params[rule_name].items():
                setattr(rule_copy, attr, value)
        result.append(rule_copy)
    return result


def _apply_params(
    strategy: Strategy,
    params: dict[str, dict[str, Any]],
) -> Strategy:
    """Create a copy of strategy with modified component params.

    Supports signals and signal-level required_indicators. Components
    are matched by name — for signals the dict key is the name.

    Only modifies parameters for components that appear in ``params``.
    Other components keep their original parameters.
    """
    new_signals = {}
    for name, (signal, orig_params) in strategy.signals.items():
        if name in params:
            merged = {**orig_params, **params[name]}
            new_signals[name] = (signal, merged)
        else:
            new_signals[name] = (signal, dict(orig_params))

        # Also apply params to required_indicators on the signal
        req_inds = getattr(signal, "required_indicators", {})
        if req_inds:
            new_req = {}
            for ind_name, (ind, ind_params) in req_inds.items():
                if ind_name in params:
                    merged_ind = {**ind_params, **params[ind_name]}
                    new_req[ind_name] = (ind, merged_ind)
                else:
                    new_req[ind_name] = (ind, dict(ind_params))
            # Set updated indicators on the signal copy
            new_signal = copy.deepcopy(new_signals[name][0])
            new_signal.required_indicators = new_req
            new_signals[name] = (new_signal, new_signals[name][1])

    # Apply params to portfolio optimizer
    new_portfolio = copy.deepcopy(strategy.portfolio)
    portfolio_name = getattr(new_portfolio, "name", None)
    if portfolio_name and portfolio_name in params:
        for attr, value in params[portfolio_name].items():
            setattr(new_portfolio, attr, value)

    # Warn about unmatched component names
    known_names: set[str] = set()
    for _name, (sig, _p) in strategy.signals.items():
        known_names.add(_name)
        for ind_name in getattr(sig, "required_indicators", {}):
            known_names.add(ind_name)
    if portfolio_name:
        known_names.add(portfolio_name)
    for rule in strategy.rules:
        rule_name = getattr(rule, "name", None)
        if rule_name:
            known_names.add(rule_name)

    for comp in params:
        if comp not in known_names:
            logger.warning(
                "Parameter component '%s' does not match any signal, "
                "indicator, or portfolio name in strategy. "
                "Known names: %s. This parameter will have NO effect.",
                comp,
                sorted(known_names),
            )

    return Strategy(
        name=strategy.name,
        signals=new_signals,
        portfolio=new_portfolio,
        rules=_apply_rule_params(strategy.rules, params),
        hypothesis=strategy.hypothesis,
        objectives=dict(strategy.objectives),
        benchmarks=list(strategy.benchmarks),
        universe=getattr(strategy, "_legacy_universe", None),
    )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Result from a single parameter combination.

    Attributes
    ----------
    params : dict[str, dict[str, Any]]
        Parameter combination, e.g. ``{"sma_fast": {"period": 10}}``.
    metric_value : float
        Raw metric value (not direction-adjusted).
    run_result : RunResult
        Full backtest result.
    """

    params: dict[str, dict[str, Any]]
    metric_value: float
    run_result: RunResult


@dataclass
class SearchResult:
    """Complete grid search results.

    Attributes
    ----------
    all_results : list[TrialResult]
        Every parameter combination tested.
    paramset : ParameterSet
        The parameter set used.
    metric : str
        Which metric was optimized.
    metric_direction : str
        ``"maximize"`` or ``"minimize"``.
    """

    all_results: list[TrialResult]
    paramset: ParameterSet
    metric: str
    metric_direction: str

    @property
    def best(self) -> TrialResult:
        """Best trial by metric and direction."""
        return self.top_n(1)[0]

    def top_n(self, n: int = 10) -> list[TrialResult]:
        """Top N trials sorted by metric (respecting direction)."""
        reverse = self.metric_direction == "maximize"
        sorted_results = sorted(
            self.all_results,
            key=lambda t: t.metric_value,
            reverse=reverse,
        )
        return sorted_results[:n]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten all trials into a DataFrame for analysis.

        Columns include one column per ``component.param`` plus
        ``metric_value`` and all standard RunResult metrics.
        """
        rows = []
        for trial in self.all_results:
            row: dict[str, Any] = {}
            # Flatten params: "sma_fast.period" -> value
            for comp, params in trial.params.items():
                for param_name, param_val in params.items():
                    row[f"{comp}.{param_name}"] = param_val
            row["metric_value"] = trial.metric_value
            # Add all standard metrics
            rr = trial.run_result
            row["total_return"] = rr.total_return()
            row["sharpe_ratio"] = rr.sharpe_ratio()
            row["max_drawdown"] = rr.max_drawdown()
            row["annualized_return"] = rr.annualized_return()
            row["annualized_volatility"] = rr.annualized_volatility()
            row["calmar_ratio"] = rr.calmar_ratio()
            row["sortino_ratio"] = rr.sortino_ratio()
            row["num_trades"] = len(rr.trades)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GridSearch
# ---------------------------------------------------------------------------


class GridSearch:
    """Exhaustive parameter grid search.

    Runs a backtest for every valid parameter combination in the given
    :class:`ParameterSet` and collects results.

    Example::

        paramset = ParameterSet("sma_tuning")
        paramset.add("sma_fast", "period", values=range(5, 30, 5))
        paramset.add("sma_slow", "period", values=range(20, 100, 10))
        paramset.add_constraint("sma_fast.period < sma_slow.period")

        result = GridSearch(paramset).run(
            strategy=strategy,
            market=LocalMarketDataProvider(),
            broker_factory=lambda: SimBroker(),
            start="2020-01-01",
            end="2024-12-31",
            metric="sharpe_ratio",
        )

        print(result.best.params)
        print(result.to_dataframe())
    """

    def __init__(self, paramset: ParameterSet) -> None:
        self.paramset = paramset

    def run(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        broker_factory: Callable[[], Broker],
        start: str,
        end: str,
        metric: str | Callable[[RunResult], float] = "sharpe_ratio",
        metric_direction: str | None = None,
        initial_cash: float = 100_000.0,
        rules: list[Any] | None = None,
        universe: UniverseProvider | None = None,
        lot_size: int = 1,
        cash_annual_return: float = 0.0,
        data_start: str | None = None,
    ) -> SearchResult:
        """Run grid search over all valid parameter combinations.

        Parameters
        ----------
        strategy : Strategy
            Base strategy. Parameters will be overridden per trial.
        market : MarketDataProvider
            Data provider (shared across all trials).
        broker_factory : Callable[[], Broker]
            Factory that returns a fresh Broker for each trial.
        start, end : str
            Date range for backtesting.
        metric : str or callable
            Metric to optimize. Either a string name matching a
            :class:`RunResult` method, or a callable ``(RunResult) -> float``.
        metric_direction : str or None
            ``"maximize"`` or ``"minimize"``. If ``None``, uses the
            default direction from :data:`METRIC_DIRECTIONS`.
        initial_cash : float
            Starting capital for each trial.

        Returns
        -------
        SearchResult
            Contains all trial results, sorted access, and DataFrame export.
        """
        direction = _resolve_direction(metric, metric_direction)
        metric_name = metric if isinstance(metric, str) else "<custom>"

        combos = self.paramset.grid()
        logger.info(
            "GridSearch '%s': %d valid combinations (metric=%s, direction=%s)",
            self.paramset.name,
            len(combos),
            metric_name,
            direction,
        )

        engine = Engine()
        all_results: list[TrialResult] = []

        for i, combo in enumerate(combos):
            modified_strategy = _apply_params(strategy, combo)
            broker = broker_factory()

            # Deep-copy rules per trial so stateful rules reset
            trial_rules = _apply_rule_params(rules, combo) if rules is not None else None

            run_result = engine.run(
                modified_strategy,
                market=market,
                broker=broker,
                start=start,
                end=end,
                initial_cash=initial_cash,
                rules=trial_rules,
                universe=universe,
                lot_size=lot_size,
                cash_annual_return=cash_annual_return,
                data_start=data_start,
            )

            metric_value = _extract_metric(run_result, metric)
            all_results.append(
                TrialResult(
                    params=combo,
                    metric_value=metric_value,
                    run_result=run_result,
                )
            )

            if (i + 1) % 10 == 0 or (i + 1) == len(combos):
                logger.info(
                    "  ... completed %d / %d trials", i + 1, len(combos),
                )

        return SearchResult(
            all_results=all_results,
            paramset=self.paramset,
            metric=metric_name,
            metric_direction=direction,
        )
