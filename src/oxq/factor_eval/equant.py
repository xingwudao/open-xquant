"""eFactorCraft-backed factor evaluation convenience functions.

These provide an alternative implementation path using eFactorCraft's
long-format panel functions. They accept oxq FactorBundle objects and
internally convert to/from eFactorCraft's expected format.

Usage::

    from oxq.factor_eval.equant import compute_ic_equant, factor_preprocess

    # IC analysis backed by eFactorCraft
    result = compute_ic_equant(bundle, forward_col="forward_20")

    # Preprocessing: winsorize + standardize using eFactorCraft
    df = factor_preprocess(df, factor_col="mom_20")
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from oxq.factor_eval.bundle import FactorBundle


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _bundle_to_panel(bundle: FactorBundle) -> pd.DataFrame:
    """Convert a FactorBundle to an eFactorCraft-compatible long-format panel.

    Returns a DataFrame with columns: date, code, factor_value, forward_N, ...
    """
    fv = bundle.factor_values
    if isinstance(fv, pd.Series):
        # MultiIndex(date, asset) -> columns
        df = fv.reset_index()
        df.columns = ["date", "code", "factor_value"]
    else:
        df = fv.copy()
        if "date" not in df.columns and df.index.name == "date":
            df = df.reset_index()
        # Ensure code column
        if "code" not in df.columns:
            asset_col = [c for c in df.columns if c.lower() in ("asset", "symbol", "code")]
            if asset_col:
                df = df.rename(columns={asset_col[0]: "code"})

    # Merge in prices (for forward returns)
    if bundle.prices is not None:
        prices_long = bundle.prices.stack().reset_index()
        prices_long.columns = ["date", "code", "price"]
        df["date"] = pd.to_datetime(df["date"])
        prices_long["date"] = pd.to_datetime(prices_long["date"])
        df = df.merge(prices_long, on=["date", "code"], how="left")

    return df


# ---------------------------------------------------------------------------
# IC Analysis
# ---------------------------------------------------------------------------


def compute_ic_equant(
    bundle: FactorBundle,
    factor_cols: Optional[Sequence[str]] = None,
    forward_col: str = "forward_20",
    method: str = "pearson",
) -> dict:
    """Compute IC analysis using eFactorCraft.

    Parameters
    ----------
    bundle : FactorBundle
        Factor values and price data.
    factor_cols : list of str, optional
        Factor column names. If None, uses "factor_value".
    forward_col : str
        Forward return column name.
    method : str
        "pearson" or "spearman".

    Returns
    -------
    dict
        IC time series and summary statistics per factor.
    """
    import efactorcraft

    df = _bundle_to_panel(bundle)
    if factor_cols is None:
        factor_cols = ["factor_value"]
        if "factor_value" not in df.columns:
            # Try to find any non-identifier column
            skip = {"date", "code", "price"}
            factor_cols = [c for c in df.columns if c not in skip][:1]

    return efactorcraft.ic_analysis(df, factor_cols=list(factor_cols),
                                    forward_col=forward_col, method=method)


def add_next_return_equant(
    df: pd.DataFrame,
    close_col: str = "close",
    periods: Sequence[int] = (1, 5, 20),
) -> pd.DataFrame:
    """Add forward return columns using eFactorCraft.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with OHLCV data.
    close_col : str
        Price column for return calculation.
    periods : sequence of int
        Forward return periods.

    Returns
    -------
    pd.DataFrame
        Panel with ``forward_N`` columns appended.
    """
    import efactorcraft
    return efactorcraft.add_next_return(df, close_col=close_col, periods=list(periods))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def factor_preprocess(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: Optional[str] = None,
    size_col: Optional[str] = None,
    probs: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """One-stop factor preprocessing using eFactorCraft.

    Pipeline: winsorize → standardize → (optional) industry_neutralize
    → (optional) size_neutralize.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with factor and price columns.
    factor_col : str
        Column name of the raw factor.
    industry_col : str, optional
        Industry classification column for neutralization.
    size_col : str, optional
        Market cap column for size neutralization.
    probs : tuple of float
        Winsorization quantile bounds.

    Returns
    -------
    pd.DataFrame
        Panel with preprocessed factor columns appended.
    """
    import efactorcraft

    df = efactorcraft.winsorize(df, factor_col=factor_col, probs=probs, append=True)
    win_col = f"win_{factor_col}"
    df = efactorcraft.standardize(df, factor_col=win_col, append=True)
    std_col = f"std_win_{factor_col}"

    if industry_col is not None:
        df = efactorcraft.industry_neutralize(
            df, factor_col=std_col, industry_col=industry_col, append=True,
        )
        std_col = f"ind_neu_{std_col}"

    if size_col is not None:
        df = efactorcraft.size_neutralize(
            df, factor_col=std_col, size_col=size_col, append=True,
        )

    return df


# ---------------------------------------------------------------------------
# Factor Synthesis
# ---------------------------------------------------------------------------


def synthesize_factors(
    df: pd.DataFrame,
    factor_cols: list[str],
    method: str = "equal_weight",
    forward_col: str = "forward_20",
) -> pd.DataFrame:
    """Combine multiple factors into a composite using eFactorCraft.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with factor columns.
    factor_cols : list of str
        Factor column names to combine.
    method : str
        One of: "equal_weight", "ic_weighted", "icir_weighted",
        "rank_weighted", "pca", "max_decay".
    forward_col : str
        Forward return column (required for IC-based methods).

    Returns
    -------
    pd.DataFrame
        Panel with composite factor column appended.
    """
    import efactorcraft

    method_map = {
        "equal_weight": efactorcraft.equal_weighted_composite,
        "ic_weighted": efactorcraft.ic_weighted_composite,
        "icir_weighted": efactorcraft.icir_weighted_composite,
        "rank_weighted": efactorcraft.rank_weighted_composite,
        "pca": efactorcraft.pca_composite,
        "max_decay": efactorcraft.max_decay_composite,
    }

    fn = method_map.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown synthesis method: {method}. "
            f"Choose from: {list(method_map)}"
        )

    kwargs = {"df": df, "factor_cols": factor_cols, "append": True}
    if method in ("ic_weighted", "icir_weighted", "max_decay"):
        kwargs["forward_col"] = forward_col

    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Factor Selection
# ---------------------------------------------------------------------------


def screen_factors(
    df: pd.DataFrame,
    factor_cols: list[str],
    forward_col: str = "forward_20",
    min_abs_ic: float = 0.02,
    min_ir: float = 0.3,
    max_corr: float = 0.7,
    min_positive_ic_pct: float = 0.55,
) -> dict:
    """Run a comprehensive factor screening using eFactorCraft.

    Returns a report with IC screen, correlation screen, and stability screen
    results.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with factor and return columns.
    factor_cols : list of str
        Candidate factor columns.
    forward_col : str
        Forward return column.
    min_abs_ic, min_ir, max_corr, min_positive_ic_pct :
        Screening thresholds.

    Returns
    -------
    dict with keys: ic_passed, corr_passed, stability_passed, report
    """
    import efactorcraft

    ic_result = efactorcraft.ic_screen(
        df, factor_cols=factor_cols, forward_col=forward_col,
        min_abs_ic=min_abs_ic, min_ir=min_ir,
    )

    passed_cols = _extract_passed(ic_result, factor_cols)
    corr_result = efactorcraft.correlation_screen(
        df, factor_cols=passed_cols, max_corr=max_corr,
    )
    corr_passed = _extract_passed(corr_result, passed_cols)

    stability_result = efactorcraft.stability_screen(
        df, factor_cols=corr_passed, forward_col=forward_col,
        min_positive_ic_pct=min_positive_ic_pct,
    )

    report = efactorcraft.factor_report(
        df, factor_cols=factor_cols, forward_col=forward_col,
    )

    return {
        "ic_passed": ic_result,
        "corr_passed": corr_result,
        "stability_passed": stability_result,
        "report": report,
    }


def _extract_passed(result: object, fallback: list[str]) -> list[str]:
    """Extract passed factor column names from a screen result."""
    if isinstance(result, dict):
        return list(result.keys())
    if isinstance(result, pd.DataFrame):
        return list(result.columns)
    if hasattr(result, "passed"):
        return list(getattr(result, "passed"))
    return fallback


# ---------------------------------------------------------------------------
# Market Regime Detection
# ---------------------------------------------------------------------------


def detect_regime(
    df: pd.DataFrame,
    close_col: str = "close",
    ma_period: int = 60,
    vol_period: int = 20,
    vol_threshold: float = 0.20,
) -> pd.DataFrame:
    """Classify each asset-date into bull/bear/sideways regime using eFactorCraft.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with OHLCV data.
    close_col : str
        Price column.
    ma_period : int
        Moving average period for trend detection.
    vol_period : int
        Volatility lookback period.
    vol_threshold : float
        Annualized vol threshold for sideways classification.

    Returns
    -------
    pd.DataFrame
        Panel with ``regime`` column appended.
    """
    import efactorcraft
    return efactorcraft.regime_detect(
        df, close_col=close_col, ma_period=ma_period,
        vol_period=vol_period, vol_threshold=vol_threshold, append=True,
    )
