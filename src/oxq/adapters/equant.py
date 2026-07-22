"""EQuantAdapter — bridge between oxq Protocol components and eQuant-Py functions.

Provides:

* ``to_panel`` / ``from_panel`` — convert between oxq per-symbol DataFrames
  and eQuant long-format panel DataFrames.
* ``EQuantAdapter`` — resolves ``"ettr::sma"``-style function references and
  executes them with automatic panel conversion.
* ``compute_panel_batch`` — batched indicator computation across all symbols
  in a single eQuant call (used by Engine for efficiency).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Panel conversion utilities
# ---------------------------------------------------------------------------


def to_panel(mktdata: pd.DataFrame, code: str = "_oxq_") -> pd.DataFrame:
    """Convert an oxq per-symbol mktdata DataFrame to an eQuant mini-panel.

    Parameters
    ----------
    mktdata : pd.DataFrame
        Per-symbol OHLCV data with DateTimeIndex.
    code : str
        Symbol identifier to populate the ``code`` column.

    Returns
    -------
    pd.DataFrame
        Long-format panel with ``date`` and ``code`` columns.
    """
    df = mktdata.copy()
    # Reset DateTimeIndex to make "date" a column
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df["date"] = dates
    elif "date" not in df.columns:
        df["date"] = df.index
    df["code"] = code
    return df.reset_index(drop=True)


def from_panel(
    result: pd.DataFrame,
    col: str,
    original_index: pd.Index,
    *,
    code: str = "_oxq_",
) -> pd.Series:
    """Extract a single column from an eQuant result and align to *original_index*.

    Parameters
    ----------
    result : pd.DataFrame
        Output from an eQuant function call (long-format panel).
    col : str
        Name of the column to extract.
    original_index : pd.Index
        The index of the original per-symbol DataFrame passed to
        :func:`to_panel`.
    code : str
        The code value used in :func:`to_panel`.

    Returns
    -------
    pd.Series
        Values for *col* aligned to *original_index* (NaN where no match).
    """
    mask = result.get("code") if "code" in result.columns else None
    if mask is not None:
        subset = result.loc[result["code"] == code, ["date", col]]
    else:
        subset = result[["date", col]]

    # Build a Series indexed by date
    if "date" in subset.columns:
        dates = pd.to_datetime(subset["date"])
        if original_index.tz is not None and dates.dt.tz is None:
            dates = dates.dt.tz_localize(original_index.tz)
        s = pd.Series(subset[col].values, index=dates, name=col)
    else:
        s = pd.Series(subset[col].values, name=col)
        s.index = original_index[: len(s)] if len(s) <= len(original_index) else original_index

    # Reindex to match the original, filling gaps with NaN
    s = s.reindex(original_index)
    return s


# ---------------------------------------------------------------------------
# EQuantAdapter
# ---------------------------------------------------------------------------


class EQuantAdapter:
    """Bridge between oxq function references and eQuant-Py packages.

    Resolves ``"ettr::sma"``-style dotted references to eQuant callables
    and executes them with automatic panel conversion.

    Usage::

        adapter = EQuantAdapter()
        result = adapter.execute("ettr::sma", mktdata, n=20)

    or resolve a function reference for later use::

        fn = EQuantAdapter.resolve("ettr::sma")
        result = adapter.to_panel_call(fn, mktdata, n=20)
    """

    PACKAGE_MAP: dict[str, str] = {
        "ettr": "ettr",
        "indicator": "ettr",
        "eclassic": "eclassic",
        "classic": "eclassic",
        "factor": "eclassic",
        "ealpha101": "ealpha101",
        "alpha": "ealpha101",
        "efactorcraft": "efactorcraft",
        "engineering": "efactorcraft",
        "ebacktestcraft": "ebacktestcraft",
        "backtest": "ebacktestcraft",
        "ecandlesticks": "ecandlesticks",
        "candlestick": "ecandlesticks",
        "edatatools": "edatatools",
        "data": "edatatools",
    }

    @staticmethod
    def resolve(func_ref: str):
        """Parse ``"pkg::func"`` into a callable.

        Examples
        --------
        >>> fn = EQuantAdapter.resolve("ettr::sma")   # doctest: +SKIP
        >>> fn = EQuantAdapter.resolve("eclassic::momentum")  # doctest: +SKIP
        """
        if "::" in func_ref:
            pkg_key, _, name = func_ref.partition("::")
        else:
            # Try to auto-resolve: import from ettr first, then eclassic
            pkg_key = "ettr"
            name = func_ref

        pkg_name = EQuantAdapter.PACKAGE_MAP.get(pkg_key, pkg_key)
        mod = __import__(pkg_name, fromlist=[name])
        return getattr(mod, name)

    @classmethod
    def execute(
        cls,
        func_ref: str,
        df: pd.DataFrame,
        **params: Any,
    ) -> pd.DataFrame:
        """Resolve *func_ref* and call it with *df* and *params*."""
        fn = cls.resolve(func_ref)
        return fn(df, **params)

    @staticmethod
    def to_panel_call(
        fn,
        mktdata: pd.DataFrame,
        col: str,
        *,
        code: str = "_oxq_",
        **params: Any,
    ) -> pd.Series:
        """Call an eQuant function with per-symbol mktdata and return a Series.

        Handles to_panel → call → from_panel conversion automatically.

        Parameters
        ----------
        fn : callable
            An eQuant function (e.g., ``ettr.sma``).
        mktdata : pd.DataFrame
            Per-symbol data (oxq convention).
        col : str
            Column to extract from the result.
        code : str
            Symbol identifier.
        **params
            Forwarded to *fn*.
        """
        panel = to_panel(mktdata, code=code)
        result = fn(panel, **params)
        return from_panel(result, col, mktdata.index, code=code)


# ---------------------------------------------------------------------------
# Panel-batched computation (for Engine optimization)
# ---------------------------------------------------------------------------


def compute_panel_batch(
    fn,
    mktdata_dict: dict[str, pd.DataFrame],
    col: str,
    **params: Any,
) -> dict[str, pd.Series]:
    """Call an eQuant function once across all symbols.

    Stacks all per-symbol DataFrames into a single long-format panel,
    calls *fn* once, and splits the result column back to per-symbol
    Series aligned to each original index.

    Parameters
    ----------
    fn : callable
        An eQuant function (e.g., ``ettr.sma``).
    mktdata_dict : dict[str, pd.DataFrame]
        Per-symbol market data (oxq convention).
    col : str
        Column name to extract from the result.
    **params
        Forwarded to *fn*.

    Returns
    -------
    dict[str, pd.Series]
        Symbol -> result Series aligned to each original index.
    """
    if not mktdata_dict:
        return {}

    # Stack all symbols into a single panel
    panels = []
    for symbol, df in mktdata_dict.items():
        p = to_panel(df, code=symbol)
        panels.append(p)
    panel = pd.concat(panels, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    # Single eQuant call
    result = fn(panel, **params)

    # Split results back per symbol
    outputs: dict[str, pd.Series] = {}
    for symbol, df in mktdata_dict.items():
        mask = result["code"] == symbol
        symbol_result = result.loc[mask, ["date", col]]
        outputs[symbol] = from_panel(
            symbol_result.rename(columns={col: col}),
            col, df.index, code=symbol,
        )
    return outputs


def _is_equant_indicator(indicator: object) -> bool:
    """Heuristic: return True if *indicator* appears to be eQuant-backed.

    Checks whether the indicator's module is a known wrapper module.
    """
    mod = getattr(type(indicator), "compute", None)
    if mod is None:
        return False
    mod_name = getattr(mod, "__module__", "")
    return "oxq.indicators" in mod_name and "oxq.adapters" not in mod_name
