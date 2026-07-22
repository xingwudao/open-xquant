"""N-day Return — log return over N days (eQuant-backed via eclassic)."""

from __future__ import annotations

import pandas as pd

from oxq.adapters.equant import to_panel, from_panel


class NdayReturn:
    """N-day cumulative log return: R_N = ln(P_t) - ln(P_{t-N})."""

    name = "NdayReturn"
    formula = r"R_N = \ln P_t - \ln P_{t-N}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        """Return N-day log returns (first ``period`` values will be NaN)."""
        import eclassic
        panel = to_panel(mktdata)
        result = eclassic.return_(panel, close_col=column, n=period,
                                  type="log", new_col="ret", append=True)
        return from_panel(result, f"ret_{period}", mktdata.index)
