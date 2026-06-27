"""Strategy tools — create, compose, and inspect strategies."""

from __future__ import annotations

import inspect
from typing import Any

from oxq.core.registry import (
    list_indicators,
    list_portfolio_optimizers,
    list_rules,
    list_signals,
)
from oxq.core.strategy import Strategy
from oxq.portfolio.optimizers import EqualWeightOptimizer
from oxq.tools import session
from oxq.tools.registry import registry
from oxq.universe.static import StaticUniverse

# ---------------------------------------------------------------------------
# Type registries — delegated to core.registry
# ---------------------------------------------------------------------------
#
# INTENTIONAL: every consumer below calls list_indicators() / list_signals()
# / list_rules() / list_portfolio_optimizers() PER CALL, not via a module-
# level snapshot. Downstream consumers (notably xquant-studio's
# component_create) register new components at runtime; a module-level
# snapshot taken at import time would freeze the available set to the
# built-ins and silently hide everything registered afterwards.
#
# Do not "optimize" this by reintroducing INDICATOR_TYPES / SIGNAL_TYPES /
# RULE_TYPES / PORTFOLIO_TYPES module-level dicts. The cost of one dict()
# copy per tool call is negligible (~50 entries); the cost of a stale
# snapshot is a multi-hour debugging session — see
# xquant-studio/docs/plans/2026-04-07-027-impl-component-create-registration-visibility.md.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_required_indicators(
    indicators: dict[str, dict[str, Any]],
) -> dict[str, tuple] | str:
    """Convert a tool-level indicators dict to required_indicators format.

    Returns a dict of (instance, params) tuples, or an error string.
    """
    from oxq.tools._coerce import coerce_compute_params

    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    indicator_types = list_indicators()
    result: dict[str, tuple] = {}
    for ind_name, ind_def in indicators.items():
        ind_type = ind_def.get("type")
        if not ind_type:
            return f"Indicator '{ind_name}' missing 'type' field"
        cls = indicator_types.get(ind_type)
        if cls is None:
            return f"Unknown indicator type '{ind_type}'. Available: {sorted(indicator_types)}"
        instance = cls()
        params = coerce_compute_params(instance, ind_def.get("params", {}))
        result[ind_name] = (instance, params)
    return result


def _universe_symbols(universe: object | None) -> list[str]:
    if universe is None or not hasattr(universe, "symbols"):
        return []
    return list(getattr(universe, "symbols"))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="strategy_create",
    description="Create a new strategy with hypothesis and objectives",
)
def strategy_create(
    name: str,
    hypothesis: str,
    objectives: dict[str, dict[str, float]] | None = None,
    benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new empty strategy and store it in the session."""
    if not hypothesis:
        return {"error": "hypothesis must not be empty"}
    if not objectives:
        return {"error": "objectives must not be empty"}

    strategy = Strategy(
        name=name,
        hypothesis=hypothesis,
        objectives=objectives,
        benchmarks=benchmarks or [],
        signals={},
        portfolio=EqualWeightOptimizer(),
        rules=[],
    )
    session._strategies[name] = strategy
    session._strategy_universes.pop(name, None)
    session._save()
    return {
        "name": name,
        "hypothesis": hypothesis,
        "objectives": objectives,
        "benchmarks": benchmarks or [],
    }


@registry.tool(
    name="strategy_list",
    description="List all strategies created in the current session",
)
def strategy_list() -> dict[str, Any]:
    """Return names and summaries of all strategies in session state."""
    return {
        "strategies": [
            {
                "name": name,
                "hypothesis": strat.hypothesis,
                "signals": len(strat.signals),
                "portfolio": strat.portfolio.name,
            }
            for name, strat in sorted(session._strategies.items())
        ],
    }


@registry.tool(
    name="strategy_add_signal",
    description=(
        "Add a signal (e.g. Crossover) to a strategy, "
        "along with the indicators it depends on"
    ),
)
def strategy_add_signal(
    strategy: str,
    name: str,
    type: str,
    params: dict[str, Any] | None = None,
    indicators: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add a signal with its required indicators to an existing strategy."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    signal_types = list_signals()
    cls = signal_types.get(type)
    if cls is None:
        return {"error": f"Unknown signal type '{type}'. Available: {sorted(signal_types)}"}

    signal = cls()

    if indicators:
        built = _build_required_indicators(indicators)
        if isinstance(built, str):
            return {"error": built}
        signal.required_indicators = built

    strat.signals[name] = (signal, params or {})
    session._save()
    return {
        "strategy": strategy,
        "signal": name,
        "type": type,
        "params": params or {},
        "indicators": list((indicators or {}).keys()),
    }


@registry.tool(
    name="strategy_add_rule",
    description=(
        "Add a rule (exit, order, constraint, or risk) to a strategy, "
        "along with the indicators it depends on"
    ),
)
def strategy_add_rule(
    strategy: str,
    name: str,
    type: str,
    params: dict[str, Any] | None = None,
    indicators: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add a rule with its required indicators to an existing strategy."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    rule_types = list_rules()
    cls = rule_types.get(type)
    if cls is None:
        return {"error": f"Unknown rule type '{type}'. Available: {sorted(rule_types)}"}

    rule_params = params or {}
    try:
        rule = cls(**rule_params)
    except TypeError as e:
        return {"error": f"Invalid params for {type}: {e}"}

    if indicators:
        built = _build_required_indicators(indicators)
        if isinstance(built, str):
            return {"error": built}
        rule.required_indicators = built

    strat.rules.append(rule)

    session._save()
    return {
        "strategy": strategy,
        "rule": name,
        "type": type,
        "params": rule_params,
    }


@registry.tool(
    name="strategy_inspect",
    description="Inspect a strategy definition (indicators, signals, rules)",
)
def strategy_inspect(strategy: str) -> dict[str, Any]:
    """Return the full strategy definition as a dict."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    pending_rules = getattr(strat, "rules", getattr(strat, "_pending_rules", []))

    # Collect indicators from all components
    def _fmt_indicators(obj: object) -> dict[str, Any]:
        return {
            k: {"type": v[0].__class__.__name__, "params": v[1]}
            for k, v in getattr(obj, "required_indicators", {}).items()
        }

    signals_info = {}
    for k, (sig, params) in strat.signals.items():
        info: dict[str, Any] = {"type": sig.__class__.__name__, "params": params}
        ind = _fmt_indicators(sig)
        if ind:
            info["indicators"] = ind
        signals_info[k] = info

    portfolio_info: dict[str, Any] = {"type": strat.portfolio.name}
    # Show portfolio constructor params. Skip required_indicators here:
    # the cleaned dict is surfaced separately under "indicators" via
    # _fmt_indicators below, and the raw `(instance, params)` tuple
    # format would crash any JSON consumer downstream
    # (xquant-studio plan 035 wall #8).
    port_params = {
        k: v for k, v in vars(strat.portfolio).items()
        if not k.startswith("_") and k != "name" and k != "required_indicators"
    }
    if port_params:
        portfolio_info["params"] = port_params
    port_ind = _fmt_indicators(strat.portfolio)
    if port_ind:
        portfolio_info["indicators"] = port_ind

    rules_info = []
    for r in pending_rules:
        rule_item: dict[str, Any] = {"type": r.__class__.__name__, "name": r.name}
        # Same skip as portfolio above — the raw tuple format would
        # leak live Indicator instances into JSON consumers.
        rule_params = {
            k: v for k, v in vars(r).items()
            if not k.startswith("_") and k != "name" and k != "required_indicators"
        }
        if rule_params:
            rule_item["params"] = rule_params
        r_ind = _fmt_indicators(r)
        if r_ind:
            rule_item["indicators"] = r_ind
        rules_info.append(rule_item)

    return {
        "name": strat.name,
        "hypothesis": strat.hypothesis,
        "objectives": strat.objectives,
        "benchmarks": strat.benchmarks,
        "universe": _universe_symbols(
            session._strategy_universes.get(strategy, getattr(strat, "_legacy_universe", None))
        ),
        "universe_binding": (
            "run_default"
            if strategy in session._strategy_universes
            else "legacy_strategy"
            if getattr(strat, "_legacy_universe", None) is not None
            else "none"
        ),
        "signals": signals_info,
        "portfolio": portfolio_info,
        "rules": rules_info,
    }


# ---------------------------------------------------------------------------
# Indicator metadata tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="indicator_describe",
    description="Describe an indicator: show its LaTeX formula, parameters, category, and dependencies",
)
def indicator_describe(type: str) -> dict[str, Any]:
    """Return indicator metadata including LaTeX formula."""
    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    indicator_types = list_indicators()
    cls = indicator_types.get(type)
    if cls is None:
        return {"error": f"Unknown indicator '{type}'. Available: {sorted(indicator_types)}"}

    sig = inspect.signature(cls().compute)
    params = {
        k: str(v.default) if v.default is not v.empty else "(required)"
        for k, v in sig.parameters.items()
        if k not in ("self", "mktdata")
    }

    return {
        "name": getattr(cls, "name", type),
        "formula": getattr(cls, "formula", ""),
        "description": (cls.__doc__ or "").strip(),
        "params": params,
        "depends_on": list(getattr(cls, "depends_on", ())),
    }


@registry.tool(
    name="indicator_list",
    description="List all available indicator types with their formulas",
)
def indicator_list() -> dict[str, Any]:
    """Return all indicator types with names, formulas, and descriptions."""
    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    return {
        "indicators": [
            {
                "name": name,
                "formula": getattr(cls, "formula", ""),
                "description": (cls.__doc__ or "").split("\n")[0].strip(),
            }
            for name, cls in sorted(list_indicators().items())
        ],
    }


# ---------------------------------------------------------------------------
# Signal metadata tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="signal_describe",
    description="Describe a signal type: show its parameters and usage",
)
def signal_describe(type: str) -> dict[str, Any]:
    """Return signal metadata including parameters."""
    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    signal_types = list_signals()
    cls = signal_types.get(type)
    if cls is None:
        return {"error": f"Unknown signal '{type}'. Available: {sorted(signal_types)}"}

    sig = inspect.signature(cls().compute)
    params = {
        k: str(v.default) if v.default is not v.empty else "(required)"
        for k, v in sig.parameters.items()
        if k not in ("self", "mktdata")
    }

    return {
        "name": getattr(cls, "name", type),
        "description": (cls.__doc__ or "").strip(),
        "params": params,
    }


@registry.tool(
    name="signal_list",
    description="List all available signal types with descriptions",
)
def signal_list() -> dict[str, Any]:
    """Return all signal types with names and descriptions."""
    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    return {
        "signals": [
            {
                "name": name,
                "description": (cls.__doc__ or "").split("\n")[0].strip(),
            }
            for name, cls in sorted(list_signals().items())
        ],
    }


# ---------------------------------------------------------------------------
# Rule metadata tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="rule_describe",
    description="Describe a rule type: show its parameters and usage",
)
def rule_describe(type: str) -> dict[str, Any]:
    """Return rule metadata including constructor parameters."""
    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    rule_types = list_rules()
    cls = rule_types.get(type)
    if cls is None:
        return {"error": f"Unknown rule '{type}'. Available: {sorted(rule_types)}"}

    sig = inspect.signature(cls.__init__)
    params = {
        k: str(v.default) if v.default is not v.empty else "(required)"
        for k, v in sig.parameters.items()
        if k != "self"
    }

    return {
        "name": getattr(cls, "name", type),
        "description": (cls.__doc__ or "").strip(),
        "params": params,
    }


@registry.tool(
    name="rule_list",
    description="List all available rule types with descriptions",
)
def rule_list() -> dict[str, Any]:
    """Return all rule types with names and descriptions."""
    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    return {
        "rules": [
            {
                "name": name,
                "description": (cls.__doc__ or "").split("\n")[0].strip(),
            }
            for name, cls in sorted(list_rules().items())
        ],
    }


# ---------------------------------------------------------------------------
# Universe tools (strategy-level)
# ---------------------------------------------------------------------------


@registry.tool(
    name="strategy_set_universe",
    description="Set the universe (symbol pool) on a strategy. "
    "type='static' for a fixed list, type='filter' for condition-based screening.",
)
def strategy_set_universe(
    strategy: str,
    type: str,
    symbols: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    name: str = "",
) -> dict[str, Any]:
    """Set a default run universe for an existing strategy."""
    from oxq.universe import Filter, FilterUniverse

    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    if type == "static":
        if not symbols:
            return {"error": "type='static' requires 'symbols' list."}
        session._strategy_universes[strategy] = StaticUniverse(symbols=tuple(symbols), name=name)
        session._save()
        return {
            "strategy": strategy,
            "universe_type": "static",
            "symbols": symbols,
        }

    if type == "filter":
        if not symbols:
            return {"error": "type='filter' requires 'symbols' (base pool) list."}
        if not filters:
            return {"error": "type='filter' requires 'filters' list."}
        filter_objs = tuple(
            Filter(column=f["column"], op=f["op"], value=f["value"]) for f in filters
        )
        session._strategy_universes[strategy] = FilterUniverse(
            base=tuple(symbols), filters=filter_objs, mktdata={}, name=name,
        )
        session._save()
        return {
            "strategy": strategy,
            "universe_type": "filter",
            "base_symbols": symbols,
            "filters": filters,
        }

    return {"error": f"Unknown type '{type}'. Use 'static' or 'filter'."}


# ---------------------------------------------------------------------------
# Portfolio optimizer tools
# ---------------------------------------------------------------------------


@registry.tool(
    name="strategy_set_portfolio",
    description="Set the portfolio optimizer on a strategy (e.g. EqualWeight, RiskParity, Kelly, TopNRanking, PctEquity)",
)
def strategy_set_portfolio(
    strategy: str,
    type: str,
    params: dict[str, Any] | None = None,
    indicators: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Set the portfolio optimizer with its required indicators."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    portfolio_types = list_portfolio_optimizers()
    cls = portfolio_types.get(type)
    if cls is None:
        return {"error": f"Unknown portfolio type '{type}'. Available: {sorted(portfolio_types)}"}

    try:
        optimizer = cls(**(params or {}))
    except TypeError as e:
        return {"error": f"Invalid params for {type}: {e}"}

    if indicators:
        built = _build_required_indicators(indicators)
        if isinstance(built, str):
            return {"error": built}
        optimizer.required_indicators = built

    strat.portfolio = optimizer
    session._save()
    return {
        "strategy": strategy,
        "portfolio": type,
        "params": params or {},
        "indicators": list((indicators or {}).keys()),
    }


@registry.tool(
    name="portfolio_list",
    description="List all available portfolio optimizer types",
)
def portfolio_list() -> dict[str, Any]:
    """Return all portfolio optimizer types with descriptions."""
    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    return {
        "optimizers": [
            {
                "name": name,
                "description": (cls.__doc__ or "").split("\n")[0].strip(),
            }
            for name, cls in sorted(list_portfolio_optimizers().items())
        ],
    }


@registry.tool(
    name="portfolio_describe",
    description="Describe a portfolio optimizer: show its parameters and usage",
)
def portfolio_describe(type: str) -> dict[str, Any]:
    """Return portfolio optimizer metadata including constructor parameters."""
    # Live registry per call — see module-top INTENTIONAL note (plan 027).
    portfolio_types = list_portfolio_optimizers()
    cls = portfolio_types.get(type)
    if cls is None:
        return {"error": f"Unknown portfolio optimizer '{type}'. Available: {sorted(portfolio_types)}"}

    sig = inspect.signature(cls.__init__)
    params = {
        k: str(v.default) if v.default is not v.empty else "(required)"
        for k, v in sig.parameters.items()
        if k != "self"
    }

    return {
        "name": getattr(cls, "name", type),
        "description": (cls.__doc__ or "").strip(),
        "params": params,
    }
