"""Strategy definition — declarative composition of pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oxq.core.types import PortfolioOptimizer, Rule, Signal
from oxq.universe.base import UniverseProvider


def _looks_like_universe_provider(value: object) -> bool:
    return value is not None and callable(getattr(value, "get_universe", None))


@dataclass(init=False)
class Strategy:
    """Reusable strategy logic independent of the tradable universe.

    Composes Signal, PortfolioOptimizer, and Rule components into a declarative
    pipeline that can be run against different universes. The same strategy
    logic applied to different symbol pools produces different portfolios.

    Strategy is purely functional: given the same market data, it always
    produces the same target portfolio. State and side effects live in
    the Engine/Trade layer.
    """

    name: str
    signals: dict[str, tuple[Signal, dict[str, Any]]]
    portfolio: PortfolioOptimizer
    rules: list[Rule]
    # Architecture metadata
    hypothesis: str = ""
    objectives: dict[str, dict[str, float]] = field(default_factory=dict)
    benchmarks: list[str] = field(default_factory=list)

    def __init__(
        self,
        name: str,
        signals: dict[str, tuple[Signal, dict[str, Any]]] | UniverseProvider | None = None,
        portfolio: PortfolioOptimizer | None = None,
        rules: list[Rule] | PortfolioOptimizer | None = None,
        *legacy_args: Any,
        hypothesis: str = "",
        objectives: dict[str, dict[str, float]] | None = None,
        benchmarks: list[str] | None = None,
        universe: UniverseProvider | None = None,
    ) -> None:
        if _looks_like_universe_provider(signals):
            universe = signals
            signals = portfolio if isinstance(portfolio, dict) else None
            portfolio = rules if rules is not None and not isinstance(rules, list) else None
            rules = None
            if legacy_args:
                hypothesis = legacy_args[0]
            if len(legacy_args) > 1:
                objectives = legacy_args[1]
            if len(legacy_args) > 2:
                benchmarks = legacy_args[2]
            if len(legacy_args) > 3:
                msg = "too many positional arguments for legacy Strategy constructor"
                raise TypeError(msg)
        elif legacy_args:
            msg = "unexpected positional arguments for Strategy constructor"
            raise TypeError(msg)
        self.name = name
        self.signals = signals or {}
        if portfolio is None:
            msg = "Strategy requires a portfolio optimizer"
            raise TypeError(msg)
        self.portfolio = portfolio
        self.rules = list(rules or [])
        self.hypothesis = hypothesis
        self.objectives = objectives or {}
        self.benchmarks = list(benchmarks or [])
        # Backward-compatibility only. New SDK code should pass a universe to
        # Engine.run/setup or store a run default outside the Strategy object.
        self._legacy_universe = universe

    @property
    def universe(self) -> UniverseProvider:
        """Legacy universe accessor.

        New code should pass ``universe=...`` to ``Engine.run``. This property
        keeps older SDK code working during the migration.
        """
        if self._legacy_universe is None:
            msg = "Strategy has no universe; pass universe=... when running it"
            raise AttributeError(msg)
        return self._legacy_universe

    @universe.setter
    def universe(self, value: UniverseProvider) -> None:
        """Legacy universe setter for older tool/session code."""
        self._legacy_universe = value

    @property
    def _pending_rules(self) -> list[Rule]:
        """Compatibility alias for older tool code."""
        return self.rules

    @_pending_rules.setter
    def _pending_rules(self, value: list[Rule]) -> None:
        self.rules = list(value)
