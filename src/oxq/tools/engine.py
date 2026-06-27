"""Engine tools — run strategies and inspect results."""

from __future__ import annotations

import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from oxq.core.engine import Engine
from oxq.data.loaders import resolve_data_dir
from oxq.data.market import LocalMarketDataProvider
from oxq.tools import session
from oxq.tools.registry import registry
from oxq.trade.fees import PercentageFee
from oxq.trade.sim_broker import FillPriceMode, SimBroker
from oxq.trade.slippage import PercentageSlippage
from oxq.universe.static import StaticUniverse


@registry.tool(
    name="engine_run",
    description="Run a strategy backtest and return summary results. Supports fee and slippage models.",
)
def engine_run(
    strategy: str,
    start: str,
    end: str,
    symbols: list[str] | None = None,
    initial_cash: float = 100_000.0,
    run_through: Literal["indicator", "signal"] | None = None,
    data_dir: str | None = None,
    fee_rate: float | None = None,
    fee_min: float | None = None,
    slippage_rate: float | None = None,
    lot_size: int = 1,
    cash_annual_return: float = 0.0,
    data_start: str | None = None,
    fill_price_mode: Literal["close", "mid", "next_avg", "next_close", "next_hl2", "next_mid", "next_open"] | None = None,
    market_calendar: str | None = "XNYS",
) -> dict[str, Any]:
    """Run a strategy through the engine and store the result."""
    strat = session._strategies.get(strategy)
    if strat is None:
        return {"error": f"Strategy '{strategy}' not found"}

    supported_fill_modes = {"close", "mid", "next_avg", "next_close", "next_hl2", "next_mid", "next_open"}
    next_session_fill_modes = {"next_avg", "next_close", "next_hl2", "next_mid", "next_open"}
    if fill_price_mode is not None and fill_price_mode not in supported_fill_modes:
        valid = ", ".join(sorted(supported_fill_modes))
        return {"error": f"Unsupported fill_price_mode '{fill_price_mode}'. Valid: {valid}"}
    if fill_price_mode in next_session_fill_modes and not market_calendar:
        return {"error": f"market_calendar is required when fill_price_mode='{fill_price_mode}'"}

    # Use symbols as a per-run universe override; otherwise use the session's
    # default run universe. Do not mutate the reusable Strategy object.
    run_universe = StaticUniverse(tuple(symbols)) if symbols else session._strategy_universes.get(strategy)
    if run_universe is None:
        run_universe = getattr(strat, "_legacy_universe", None)
    if run_universe is None or not getattr(run_universe, "symbols", ()):
        return {"error": "No universe set for this run. Use strategy_set_universe first, or pass symbols parameter."}

    path = resolve_data_dir(Path(data_dir) if data_dir else None)
    provider_calendar = market_calendar if fill_price_mode in next_session_fill_modes else None
    market = LocalMarketDataProvider(path, calendar=provider_calendar)

    # Build fee and slippage models from params
    fee_model = None
    if fee_rate is not None:
        fee_model = PercentageFee(
            rate=Decimal(str(fee_rate)),
            min_fee=Decimal(str(fee_min)) if fee_min is not None else Decimal("0"),
        )

    slippage_model = None
    if slippage_rate is not None:
        slippage_model = PercentageSlippage(rate=Decimal(str(slippage_rate)))

    broker_kwargs: dict[str, Any] = {}
    if fee_model is not None:
        broker_kwargs["fee_model"] = fee_model
    if slippage_model is not None:
        broker_kwargs["slippage_model"] = slippage_model
    if fill_price_mode is not None:
        broker_kwargs["fill_price_mode"] = FillPriceMode(fill_price_mode)
        if fill_price_mode in next_session_fill_modes:
            broker_kwargs["market_calendar"] = market_calendar
    broker = SimBroker(**broker_kwargs)

    # Collect pending rules from strategy
    rules = getattr(strat, "rules", getattr(strat, "_pending_rules", [])) or []

    try:
        result = Engine().run(
            strat,
            market=market,
            broker=broker,
            start=start,
            end=end,
            initial_cash=initial_cash,
            run_through=run_through,
            rules=rules,
            universe=run_universe,
            lot_size=lot_size,
            cash_annual_return=cash_annual_return,
            data_start=data_start,
        )
    except Exception as e:
        import traceback

        return {"error": str(e), "type": type(e).__name__, "traceback": traceback.format_exc()}

    run_id = f"{strategy}_{int(time.time())}"
    session._run_results[run_id] = result
    session._save()

    # Get last market prices from mktdata for position valuation
    last_prices: dict[str, float] = {}
    for sym in result.mktdata:
        df = result.mktdata[sym]
        if not df.empty and "close" in df.columns:
            last_prices[sym] = float(df["close"].iloc[-1])

    positions = {
        sym: {
            "shares": int(pos.shares),
            "avg_cost": float(pos.avg_cost),
            "market_price": last_prices.get(sym, float(pos.avg_cost)),
        }
        for sym, pos in result.portfolio.positions.items()
    }

    total_value = float(
        result.equity_curve[-1][1] if result.equity_curve else result.portfolio.cash,
    )

    return {
        "run_id": run_id,
        "portfolio": {
            "cash": float(result.portfolio.cash),
            "currency": result.portfolio.currency,
            "positions": positions,
            "total_value": total_value,
        },
        "total_trades": len(result.trades),
        "equity_curve_length": len(result.equity_curve),
    }


@registry.tool(
    name="run_list",
    description="List all strategy runs in the current session with key metrics",
)
def run_list() -> dict[str, Any]:
    """Return run IDs with summary metrics for all runs in session state."""
    runs = []
    for run_id, result in sorted(session._run_results.items()):
        runs.append({
            "run_id": run_id,
            "total_return": round(result.total_return(), 4),
            "sharpe_ratio": round(result.sharpe_ratio(), 4),
            "max_drawdown": round(result.max_drawdown(), 4),
            "total_trades": len(result.trades),
        })
    return {"runs": runs}


@registry.tool(
    name="engine_results",
    description="Get performance metrics and objectives check for a run",
)
def engine_results(run_id: str) -> dict[str, Any]:
    """Compute metrics and check against strategy objectives."""
    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    metrics = {
        "total_return": float(result.total_return()),
        "annualized_return": float(result.annualized_return()),
        "annualized_volatility": float(result.annualized_volatility()),
        "max_drawdown": float(result.max_drawdown()),
        "sharpe_ratio": float(result.sharpe_ratio()),
        "calmar_ratio": float(result.calmar_ratio()),
        "sortino_ratio": float(result.sortino_ratio()),
    }

    # Check objectives from the strategy that produced this result
    objectives_check: list[dict[str, Any]] = []
    # Find the strategy name from the run_id prefix
    strategy_name = run_id.rsplit("_", 1)[0]
    strat = session._strategies.get(strategy_name)

    # Metrics where the value is negative and "max" means abs(actual) <= abs(max)
    _abs_compare_metrics = {"max_drawdown"}

    if strat and strat.objectives:
        for metric_name, bounds in strat.objectives.items():
            actual = metrics.get(metric_name)
            if actual is None:
                continue
            check: dict[str, Any] = {"metric": metric_name, "actual": actual}
            passed = True
            if "min" in bounds:
                check["min"] = bounds["min"]
                passed = passed and actual >= bounds["min"]
            if "max" in bounds:
                check["max"] = bounds["max"]
                if metric_name in _abs_compare_metrics:
                    # For drawdown: -0.08 is better than -0.15,
                    # so abs(actual) <= abs(max) means pass
                    passed = passed and abs(actual) <= abs(bounds["max"])
                else:
                    passed = passed and actual <= bounds["max"]
            if "target" in bounds:
                check["target"] = bounds["target"]
            check["pass"] = passed
            objectives_check.append(check)

    return {
        "run_id": run_id,
        "currency": result.portfolio.currency,
        "metrics": metrics,
        "objectives_check": objectives_check,
    }


@registry.tool(
    name="engine_trade_list",
    description="Get the list of trades from a backtest run",
)
def engine_trade_list(run_id: str) -> dict[str, Any]:
    """Return all trades from a run."""
    result = session._run_results.get(run_id)
    if result is None:
        return {"error": f"Run '{run_id}' not found"}

    trades = [
        {
            "symbol": fill.order.symbol,
            "side": fill.order.side,
            "shares": int(fill.order.shares),
            "order_type": fill.order.order_type,
            "price": float(fill.filled_price),
            "fee": float(fill.fee),
            "date": fill.filled_at,
            "currency": fill.order.currency,
        }
        for fill in result.trades
    ]

    return {
        "run_id": run_id,
        "total_trades": len(trades),
        "trades": trades,
    }
