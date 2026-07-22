"""Momentum factor — (ln(P_t) - ln(P_{t-N})) / N (eQuant-backed via eclassic)."""

from __future__ import annotations

import pandas as pd

from oxq.adapters.equant import to_panel, from_panel


class Momentum:
    """N-day momentum (average daily log return over N days).

    Momentum_N = (ln(P_t) - ln(P_{t-N})) / N
    """

    name = "Momentum"
    formula = r"Mom_N = \frac{\ln P_t - \ln P_{t-N}}{N}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        """Return N-day momentum (first ``period`` values will be NaN)."""
        import eclassic
        panel = to_panel(mktdata)
        result = eclassic.momentum(panel, close_col=column, n=period,
                                   type="log", new_col="mom", append=True)
        return from_panel(result, f"mom_{period}", mktdata.index)
