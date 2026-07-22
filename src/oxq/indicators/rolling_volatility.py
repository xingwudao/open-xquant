"""Rolling Volatility — std of log returns (eQuant-backed via eclassic)."""

from __future__ import annotations

import pandas as pd

from oxq.adapters.equant import to_panel, from_panel


class RollingVolatility:
    """N-day rolling volatility of log returns.

    sigma_{N,t} = sqrt(1/(N-1) * sum(r_{t-i} - r_bar_N)^2)
    """

    name = "RollingVolatility"
    formula = r"\sigma_N = \sqrt{\frac{1}{N-1}\sum_{i=0}^{N-1}(r_{t-i}-\bar{r}_N)^2}"

    def compute(
        self, mktdata: pd.DataFrame, column: str = "close", period: int = 20,
    ) -> pd.Series:
        """Return rolling std of log returns (uses ddof=1)."""
        import eclassic
        panel = to_panel(mktdata)
        result = eclassic.volatility(panel, close_col=column, n=period,
                                     type="sd", new_col="vol",
                                     append=True)
        return from_panel(result, f"vol_{period}", mktdata.index)
