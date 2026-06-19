"""Factor evaluation tool — assess indicator predictive power."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from oxq.core.registry import list_indicators
from oxq.data.loaders import resolve_data_dir
from oxq.tools._coerce import coerce_compute_params
from oxq.tools.registry import registry

_MIN_SYMBOLS_FOR_SIGNIFICANCE = 30


@registry.tool(
    name="factor_evaluate",
    description=(
        "Evaluate an indicator's predictive power as a factor. "
        "Computes IC, ICIR, RankIC, decay, and turnover on multi-symbol data. "
        "Returns a structured evaluation report."
    ),
)
def factor_evaluate(
    indicator: str,
    params: dict[str, Any],
    symbols: list[str],
    start: str,
    end: str,
    data_dir: str | None = None,
    forward_days: int = 5,
    decay_horizons: list[int] | None = None,
    min_obs: int = 3,
) -> dict[str, Any]:
    """Evaluate an indicator as a predictive factor."""
    try:
        from oxq.factor_eval.metrics import (
            compute_decay,
            compute_ic,
            compute_icir,
            compute_rank_ic,
            compute_ts_ic,
            compute_turnover,
        )
    except ImportError as exc:
        return {
            "error": (
                "factor_evaluate requires scipy. "
                "Install it with: uv sync --extra scipy"
            ),
            "detail": str(exc),
        }

    if decay_horizons is None:
        decay_horizons = [1, 5, 10, 20]

    # -- Resolve indicator class ------------------------------------------------
    indicators = list_indicators()
    if indicator not in indicators:
        return {"error": f"Unknown indicator '{indicator}'. Available: {sorted(indicators)}"}

    ind_cls = indicators[indicator]
    ind_instance = ind_cls()
    params = coerce_compute_params(ind_instance, params)

    # -- Load market data -------------------------------------------------------
    data_path = resolve_data_dir(Path(data_dir) if data_dir else None)
    mktdata: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for sym in symbols:
        parquet = data_path / f"{sym}.parquet"
        if not parquet.exists():
            errors.append(f"No data for '{sym}'")
            continue
        df = pd.read_parquet(parquet)
        mktdata[sym] = df.loc[start:end]

    if not mktdata:
        return {"error": f"No data found. {'; '.join(errors)}"}

    # -- Compute indicator per symbol -------------------------------------------
    factor_series: dict[str, pd.Series] = {}
    for sym, df in mktdata.items():
        factor_series[sym] = ind_instance.compute(df, **params)

    # -- Build cross-sectional DataFrames ---------------------------------------
    factor_df = pd.DataFrame(factor_series)
    prices_df = pd.DataFrame({sym: df["close"] for sym, df in mktdata.items()})

    # Align indices
    common_idx = factor_df.index.intersection(prices_df.index)
    factor_df = factor_df.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    # -- Compute forward returns ------------------------------------------------
    fwd_returns = prices_df.pct_change(forward_days).shift(-forward_days)

    # -- Warnings ---------------------------------------------------------------
    warnings: list[str] = []
    n_symbols = len(mktdata)
    if n_symbols < _MIN_SYMBOLS_FOR_SIGNIFICANCE:
        warnings.append(
            f"symbols_count ({n_symbols}) < {_MIN_SYMBOLS_FOR_SIGNIFICANCE}: "
            "statistical significance is weak"
        )
    if errors:
        warnings.append(f"Skipped symbols: {'; '.join(errors)}")

    # -- Compute metrics --------------------------------------------------------
    ic_result = compute_ic(factor_df, fwd_returns, min_obs=min_obs)
    rank_ic_result = compute_rank_ic(factor_df, fwd_returns, min_obs=min_obs)
    icir = compute_icir(float(ic_result["mean"]), float(ic_result["std"]))
    decay_result = compute_decay(factor_df, prices_df, decay_horizons, min_obs=min_obs)
    turnover = compute_turnover(factor_df)
    ts_ic_result = compute_ts_ic(factor_df, fwd_returns, min_obs=max(min_obs, 30))

    # -- Assemble report --------------------------------------------------------
    return {
        "indicator": indicator,
        "params": params,
        "symbols_count": n_symbols,
        "date_range": {"start": start, "end": end},
        "warnings": warnings,
        "metrics": {
            "ic": {
                "mean": float(ic_result["mean"]),
                "std": float(ic_result["std"]),
            },
            "icir": {
                "value": float(icir),
            },
            "rank_ic": {
                "mean": float(rank_ic_result["mean"]),
                "std": float(rank_ic_result["std"]),
            },
            "decay": {
                "horizons": decay_result["horizons"],
                "ic_values": [float(v) for v in decay_result["ic_values"]],
            },
            "turnover": {
                "mean": float(turnover),
            },
            "ts_ic": {
                "mean": float(ts_ic_result["mean"]),
                "per_symbol": {
                    k: float(v) for k, v in ts_ic_result["per_symbol"].items()
                },
            },
        },
        "ic_series": {
            "values": [float(v) for v in ic_result["series"]],
        },
    }
