"""Log Return — single-period log return (eQuant-backed via eclassic)."""

from __future__ import annotations

import pandas as pd

from oxq.adapters.equant import to_panel, from_panel


class LogReturn:
    """Daily log return: r_t = ln(P_t) - ln(P_{t-1})."""

    name = "LogReturn"
    formula = r"r_t = \ln P_t - \ln P_{t-1}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close",
    ) -> pd.Series:
        """Return log returns (first value will be NaN)."""
        import eclassic
        panel = to_panel(mktdata)
        result = eclassic.return_(panel, close_col=column, n=1,
                                  type="log", new_col="ret", append=True)
        return from_panel(result, "ret_1", mktdata.index)
