"""FormulaExitRule: close a held position when a boolean expression is True."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, RuleResult


class FormulaExitRule:
    """Exit a held symbol when ``expr`` evaluates to True on the current bar.

    The expression is evaluated against the bar row (OHLCV, indicator and
    signal columns). Symbols that are not held are skipped before evaluation
    (the engine calls rules for the whole universe every bar), and NaN
    comparisons evaluate to False, so warm-up bars never exit. Stateless, so
    the engine's two calls per bar are harmless.
    """

    name = "FormulaExitRule"

    # expr is required (no default), like ExitRule(fast, slow): the registry then
    # validates structurally instead of instantiating with no arguments.
    def __init__(self, expr: str) -> None:
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("FormulaExitRule requires a non-empty expr")
        self.expr = expr

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        if symbol not in portfolio.positions:
            return RuleResult()
        frame = pd.DataFrame([row.to_dict()])
        if bool(frame.eval(self.expr).astype(bool).iloc[0]):
            return RuleResult(
                target_positions={symbol: 0.0},
                reason=f"{self.expr} on {row.name}, exit {symbol}",
            )
        return RuleResult()
