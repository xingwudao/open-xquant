"""FormulaEvent signal: boolean expression with event (entry-trigger) lifecycle."""

from __future__ import annotations

import pandas as pd


class FormulaEvent:
    """True where ``expr`` holds; treated as an entry event, not a level.

    The value is identical to ``Formula``. The lifecycle differs: with an
    ``EqualWeight`` portfolio a True bar opens a position that is then held
    until an exit rule closes it, instead of being re-evaluated every bar.
    Unlike ``Crossover`` no symmetric exit is implied; declare one explicitly
    (e.g. ``FormulaExitRule``).
    """

    name = "FormulaEvent"

    def compute(self, mktdata: pd.DataFrame, expr: str = "") -> pd.Series:
        """Return the boolean value of ``expr`` on every bar."""
        return mktdata.eval(expr).astype(bool)
