"""Parity of spec-built Zettaranc indicators against an independent reference.

The reference mirrors SebastienZh/StockTradebyZ ``pipeline/Selector.py``:
KDJ seeds K = D = 50 and uses ``rolling(min_periods=1)`` with a 1e-9 guard,
so K/D/J only converge after warm-up; white/yellow lines must match exactly.
"""

import numpy as np
import pandas as pd

from oxq.indicators import EMA, SMA, LinearCombination, StochK, TdxSMA


def _bars(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    spread = np.abs(rng.normal(0.0, 0.01, n)) * close
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close + spread, "low": close - spread, "close": close, "volume": 1e6},
        index=idx,
    )


def _reference_kdj(df: pd.DataFrame, n: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_n = df["low"].rolling(n, min_periods=1).min()
    high_n = df["high"].rolling(n, min_periods=1).max()
    rsv = ((df["close"] - low_n) / (high_n - low_n + 1e-9) * 100).to_numpy()
    k = np.empty(len(rsv))
    d = np.empty(len(rsv))
    k[0] = d[0] = 50.0
    for i in range(1, len(rsv)):
        k[i] = 2.0 / 3.0 * k[i - 1] + 1.0 / 3.0 * rsv[i]
        d[i] = 2.0 / 3.0 * d[i - 1] + 1.0 / 3.0 * k[i]
    return k, d, 3 * k - 2 * d


def _spec_chain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema10"] = EMA().compute(out, column="close", period=10)
    out["white"] = EMA().compute(out, column="ema10", period=10)
    for p in (14, 28, 57, 114):
        out[f"ma{p}"] = SMA().compute(out, column="close", period=p)
    out["yellow"] = LinearCombination().compute(out, columns=["ma14", "ma28", "ma57", "ma114"], weights=[0.25] * 4)
    out["rsv"] = StochK().compute(out, period=9)
    out["k"] = TdxSMA().compute(out, column="rsv", n=3, m=1)
    out["d"] = TdxSMA().compute(out, column="k", n=3, m=1)
    out["j"] = LinearCombination().compute(out, columns=["k", "d"], weights=[3.0, -2.0])
    return out


def test_white_yellow_match_reference_exactly() -> None:
    df = _bars()
    out = _spec_chain(df)
    close = df["close"]
    ref_white = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    ref_yellow = sum(close.rolling(p, min_periods=p).mean() for p in (14, 28, 57, 114)) / 4.0
    np.testing.assert_allclose(out["white"], ref_white, atol=1e-9)
    np.testing.assert_allclose(out["yellow"], ref_yellow, atol=1e-9)


def test_kdj_converges_to_reference_after_warmup() -> None:
    df = _bars()
    out = _spec_chain(df)
    k, d, j = _reference_kdj(df)
    np.testing.assert_allclose(out["k"].to_numpy()[60:], k[60:], atol=1e-6)
    np.testing.assert_allclose(out["d"].to_numpy()[60:], d[60:], atol=1e-6)
    np.testing.assert_allclose(out["j"].to_numpy()[60:], j[60:], atol=1e-6)
