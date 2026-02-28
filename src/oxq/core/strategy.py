"""Strategy definition — declarative composition of pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oxq.core.types import Indicator, Rule, Signal
from oxq.universe.base import UniverseProvider


@dataclass
class Strategy:
    """A complete strategy definition.

    Composes Universe, Indicators, Signals, and Rules into a declarative
    pipeline that the engine can execute.
    """

    name: str
    universe: UniverseProvider
    indicators: dict[str, tuple[Indicator, dict[str, Any]]]
    signals: dict[str, tuple[Signal, dict[str, Any]]]
    entry_rules: list[Rule]
    exit_rules: list[Rule]
    # Architecture metadata (Section 4.1)
    hypothesis: str = ""
    objectives: dict[str, dict[str, float]] = field(default_factory=dict)
    benchmarks: list[str] = field(default_factory=list)
