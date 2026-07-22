"""AnnualizedVolatility — annualized population stddev (eQuant-backed via eclassic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oxq.adapters.equant import to_panel, from_panel


class AnnualizedVolatility:
    """Rolling annualized volatility of simple returns.

    vol = pstdev(simple_returns, period) * sqrt(252)

    Uses population standard deviation (ddof=0) to match xquant reference.
    """

    name = "AnnualizedVolatility"
    formula = r"\sigma = \text{pstdev}(r_{\text{simple}}) \times \sqrt{252}"

    def compute(
        self,
        mktdata: pd.DataFrame,
        column: str = "close",
        period: int = 20,
    ) -> pd.Series:
        """Return rolling annualized volatility (population stddev)."""
        import eclassic
        panel = to_panel(mktdata)
        result = eclassic.volatility(panel, close_col=column, n=period,
                                     type="sd", trading_days=252,
                                     new_col="vol", append=True)
        return from_panel(result, f"vol_{period}", mktdata.index)
