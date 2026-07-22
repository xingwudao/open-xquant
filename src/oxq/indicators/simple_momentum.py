"""SimpleMomentum — simple return normalized by period (eQuant-backed via eclassic)."""

from __future__ import annotations

import pandas as pd

from oxq.adapters.equant import to_panel, from_panel


class SimpleMomentum:
    """N-day momentum using simple returns.

    SimpleMomentum_N = (P_t / P_{t-N} - 1) / N
    """

    name = "SimpleMomentum"
    formula = r"Mom_N = \frac{P_t / P_{t-N} - 1}{N}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        """Return N-day simple momentum (first ``period`` values will be NaN)."""
        import eclassic
        panel = to_panel(mktdata)
        result = eclassic.momentum(panel, close_col=column, n=period,
                                   type="continuous", new_col="mom", append=True)
        return from_panel(result, f"mom_{period}", mktdata.index)
