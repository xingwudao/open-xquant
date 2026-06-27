"""Walk-forward analysis for out-of-sample validation."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from oxq.core.engine import Engine
from oxq.core.strategy import Strategy
from oxq.core.types import Broker
from oxq.data.providers import MarketDataProvider
from oxq.optimize.paramset import ParameterSet
from oxq.optimize.search import (
    GridSearch,
    _apply_params,
    _extract_metric,
    _resolve_direction,
)
from oxq.portfolio.analytics import RunResult
from oxq.universe.base import UniverseProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Period parsing
# ---------------------------------------------------------------------------

_PERIOD_RE = re.compile(r"^(\d+)\s*([YMDymd])$")


def _parse_period(period: str) -> pd.DateOffset:
    """Parse a period string like '2Y', '6M', '63D' into a DateOffset."""
    m = _PERIOD_RE.match(period.strip())
    if m is None:
        msg = (
            f"Invalid period format: {period!r}. "
            "Expected '<number><unit>' where unit is Y, M, or D. "
            "Examples: '2Y', '6M', '63D'"
        )
        raise ValueError(msg)
    n = int(m.group(1))
    unit = m.group(2).upper()
    if unit == "Y":
        return pd.DateOffset(years=n)
    if unit == "M":
        return pd.DateOffset(months=n)
    return pd.DateOffset(days=n)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class WindowResult:
    """Single walk-forward window result.

    Attributes
    ----------
    train_start, train_end : str
        Training period boundaries.
    test_start, test_end : str
        Testing period boundaries.
    best_params : dict
        Best parameters found in-sample.
    in_sample_metric : float
        Best metric value from in-sample optimization.
    oos_result : RunResult
        Full backtest result on the out-of-sample period.
    """

    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict[str, dict[str, Any]]
    in_sample_metric: float
    oos_result: RunResult


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result.

    Attributes
    ----------
    windows : list[WindowResult]
        Per-window details.
    metric : str
        Metric that was optimized.
    metric_direction : str
        ``"maximize"`` or ``"minimize"``.
    """

    windows: list[WindowResult]
    metric: str
    metric_direction: str

    @property
    def oos_equity_curve(self) -> list[tuple]:
        """Stitch all OOS equity curves into one continuous series."""
        combined: list[tuple] = []
        for window in self.windows:
            combined.extend(window.oos_result.equity_curve)
        return combined

    def oos_total_return(self) -> float:
        """Total return across all stitched OOS periods."""
        curve = self.oos_equity_curve
        if len(curve) < 2:
            return 0.0
        first = curve[0][1]
        last = curve[-1][1]
        if first == 0.0:
            return 0.0
        return (last - first) / first

    def oos_sharpe_ratio(self, trading_days: int = 252) -> float:
        """Sharpe ratio across all stitched OOS periods."""
        curve = self.oos_equity_curve
        if len(curve) < 2:
            return 0.0
        values = np.array([v for _, v in curve], dtype=float)
        returns = np.diff(values) / values[:-1]
        if len(returns) == 0 or np.std(returns) == 0.0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(trading_days))

    def oos_max_drawdown(self) -> float:
        """Max drawdown across all stitched OOS periods."""
        curve = self.oos_equity_curve
        if len(curve) < 2:
            return 0.0
        values = np.array([v for _, v in curve], dtype=float)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return float(np.min(drawdown))

    def deterioration(self) -> dict[str, float]:
        """IS vs OOS comparison per metric.

        Returns a dict mapping metric names to deterioration ratios.
        Negative means OOS is worse than IS.
        E.g. ``{"sharpe_ratio": -0.35}`` means OOS sharpe is 35% worse.
        """
        if not self.windows:
            return {}

        metrics_to_compare = [
            "total_return", "sharpe_ratio", "max_drawdown",
            "annualized_return", "calmar_ratio", "sortino_ratio",
        ]

        result: dict[str, float] = {}
        for metric_name in metrics_to_compare:
            is_values = []
            oos_values = []
            for w in self.windows:
                oos_fn = getattr(w.oos_result, metric_name, None)
                if oos_fn is not None and callable(oos_fn):
                    oos_values.append(oos_fn())

                # IS metric only available for the optimization metric
                if metric_name == self.metric:
                    is_values.append(w.in_sample_metric)

            if metric_name == self.metric and is_values and oos_values:
                mean_is = np.mean(is_values)
                mean_oos = np.mean(oos_values)
                if abs(mean_is) > 1e-10:
                    result[metric_name] = float((mean_oos - mean_is) / abs(mean_is))
                else:
                    result[metric_name] = 0.0

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """One row per window with train/test dates, params, and metrics."""
        rows = []
        for w in self.windows:
            row: dict[str, Any] = {
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "in_sample_metric": w.in_sample_metric,
            }
            # Flatten best params
            for comp, params in w.best_params.items():
                for param_name, param_val in params.items():
                    row[f"{comp}.{param_name}"] = param_val
            # OOS metrics
            rr = w.oos_result
            row["oos_total_return"] = rr.total_return()
            row["oos_sharpe_ratio"] = rr.sharpe_ratio()
            row["oos_max_drawdown"] = rr.max_drawdown()
            row["oos_annualized_return"] = rr.annualized_return()
            row["oos_calmar_ratio"] = rr.calmar_ratio()
            row["oos_sortino_ratio"] = rr.sortino_ratio()
            row["oos_num_trades"] = len(rr.trades)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# WalkForward
# ---------------------------------------------------------------------------


class WalkForward:
    """Walk-forward analysis: rolling or anchored out-of-sample validation.

    Splits the data into successive train/test windows. For each window,
    runs :class:`GridSearch` on the training period to find the best
    parameters, then evaluates those parameters on the test period.

    Rolling::

        |--train--|--test--|
              |--train--|--test--|
                    |--train--|--test--|

    Anchored::

        |----train----|--test--|
        |------train------|--test--|
        |--------train--------|--test--|

    Example::

        wf = WalkForward(
            paramset,
            train_period="2Y",
            test_period="6M",
            anchored=False,
        )
        result = wf.run(
            strategy=strategy,
            market=provider,
            broker_factory=lambda: SimBroker(),
            start="2018-01-01",
            end="2024-12-31",
            metric="sharpe_ratio",
        )
        print(result.deterioration())
    """

    def __init__(
        self,
        paramset: ParameterSet,
        train_period: str,
        test_period: str,
        step: str | None = None,
        anchored: bool = False,
    ) -> None:
        self.paramset = paramset
        self.train_offset = _parse_period(train_period)
        self.test_offset = _parse_period(test_period)
        self.step_offset = _parse_period(step) if step else self.test_offset
        self.anchored = anchored
        self._train_period_str = train_period
        self._test_period_str = test_period

    def _generate_windows(
        self, start: str, end: str,
    ) -> list[tuple[str, str, str, str]]:
        """Generate (train_start, train_end, test_start, test_end) tuples."""
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        windows: list[tuple[str, str, str, str]] = []

        # cursor advances by step each iteration
        cursor = start_dt
        while True:
            if self.anchored:
                # Anchored: train always starts at start_dt, grows longer
                train_start = start_dt
                train_end = cursor + self.train_offset - pd.DateOffset(days=1)
            else:
                # Rolling: train window slides forward
                train_start = cursor
                train_end = cursor + self.train_offset - pd.DateOffset(days=1)

            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + self.test_offset - pd.DateOffset(days=1)

            # Stop if test period exceeds data range
            if test_start > end_dt:
                break

            # Clip test_end to data range
            if test_end > end_dt:
                test_end = end_dt

            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

            cursor = cursor + self.step_offset

        return windows

    def run(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        broker_factory: Callable[[], Broker],
        start: str,
        end: str,
        metric: str | Callable[[RunResult], float] = "sharpe_ratio",
        metric_direction: str | None = None,
        initial_cash: float = 100_000.0,
        universe: UniverseProvider | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        Parameters
        ----------
        strategy : Strategy
            Base strategy definition.
        market : MarketDataProvider
            Data provider.
        broker_factory : Callable[[], Broker]
            Factory returning fresh broker per trial.
        start, end : str
            Overall date range.
        metric : str or callable
            Optimization metric.
        metric_direction : str or None
            ``"maximize"`` or ``"minimize"``.
        initial_cash : float
            Starting capital per run.

        Returns
        -------
        WalkForwardResult
        """
        direction = _resolve_direction(metric, metric_direction)
        metric_name = metric if isinstance(metric, str) else "<custom>"

        windows = self._generate_windows(start, end)
        logger.info(
            "WalkForward '%s': %d windows (train=%s, test=%s, anchored=%s)",
            self.paramset.name,
            len(windows),
            self._train_period_str,
            self._test_period_str,
            self.anchored,
        )

        if not windows:
            logger.warning(
                "No walk-forward windows could be generated for the "
                "given date range and period configuration."
            )
            return WalkForwardResult(
                windows=[], metric=metric_name, metric_direction=direction,
            )

        engine = Engine()
        window_results: list[WindowResult] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(
                "  Window %d/%d: train [%s, %s] -> test [%s, %s]",
                i + 1, len(windows),
                train_start, train_end, test_start, test_end,
            )

            # In-sample: GridSearch on training period
            gs = GridSearch(self.paramset)
            search_result = gs.run(
                strategy=strategy,
                market=market,
                broker_factory=broker_factory,
                start=train_start,
                end=train_end,
                metric=metric,
                metric_direction=metric_direction,
                initial_cash=initial_cash,
                universe=universe,
            )

            best = search_result.best
            logger.info(
                "    IS best: %s = %.4f, params = %s",
                metric_name, best.metric_value, best.params,
            )

            # Out-of-sample: run best params on test period
            oos_strategy = _apply_params(strategy, best.params)
            oos_result = engine.run(
                oos_strategy,
                market=market,
                broker=broker_factory(),
                start=test_start,
                end=test_end,
                initial_cash=initial_cash,
                universe=universe,
            )

            oos_metric = _extract_metric(oos_result, metric)
            logger.info("    OOS %s = %.4f", metric_name, oos_metric)

            window_results.append(WindowResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best.params,
                in_sample_metric=best.metric_value,
                oos_result=oos_result,
            ))

        return WalkForwardResult(
            windows=window_results,
            metric=metric_name,
            metric_direction=direction,
        )
