"""Time series cross-validation for strategy evaluation."""

from __future__ import annotations

import logging
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
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CVSplit:
    """Single cross-validation split definition.

    Attributes
    ----------
    train_start, train_end : str
        Training period boundaries.
    test_start, test_end : str
        Testing period boundaries.
    """

    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class CVSplitResult:
    """Result from a single CV split.

    Attributes
    ----------
    split : CVSplit
        The split definition.
    best_params : dict or None
        Best parameters found (``None`` if no paramset was provided).
    in_sample_metric : float or None
        In-sample metric value (``None`` if no paramset).
    oos_result : RunResult
        Full backtest result on the test period.
    """

    split: CVSplit
    best_params: dict[str, dict[str, Any]] | None
    in_sample_metric: float | None
    oos_result: RunResult


@dataclass
class CVResult:
    """Cross-validation results.

    Attributes
    ----------
    splits : list[CVSplitResult]
        Per-split results.
    metric : str
        Which metric was evaluated.
    metric_direction : str
        ``"maximize"`` or ``"minimize"``.
    """

    splits: list[CVSplitResult]
    metric: str
    metric_direction: str

    def mean_oos_metric(self) -> float:
        """Mean OOS metric value across all splits."""
        values = [
            _extract_metric(s.oos_result, self.metric)
            for s in self.splits
        ]
        return float(np.mean(values))

    def std_oos_metric(self) -> float:
        """Standard deviation of OOS metric across splits."""
        values = [
            _extract_metric(s.oos_result, self.metric)
            for s in self.splits
        ]
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """One row per split with dates, params, and OOS metrics."""
        rows = []
        for sr in self.splits:
            row: dict[str, Any] = {
                "train_start": sr.split.train_start,
                "train_end": sr.split.train_end,
                "test_start": sr.split.test_start,
                "test_end": sr.split.test_end,
                "in_sample_metric": sr.in_sample_metric,
            }
            if sr.best_params:
                for comp, params in sr.best_params.items():
                    for param_name, param_val in params.items():
                        row[f"{comp}.{param_name}"] = param_val
            rr = sr.oos_result
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
# TimeSeriesCV
# ---------------------------------------------------------------------------


class TimeSeriesCV:
    """Time series cross-validation splitter.

    Generates train/test splits that respect temporal ordering — no future
    data leaks into training. Supports an optional embargo gap between
    train and test to mitigate information leakage from autocorrelation.

    Expanding mode (default)::

        Split 1: |==train==|--test--|.........|
        Split 2: |=====train=====|--test--|...|
        Split 3: |========train========|--test--|

    Fixed-size sliding mode::

        Split 1: |==train==|--test--|.........|
        Split 2: ...|==train==|--test--|......|
        Split 3: ......|==train==|--test--|...|

    With embargo::

        Split 1: |==train==|.gap.|--test--|......|

    Example::

        cv = TimeSeriesCV(n_splits=5, embargo_days=5, expanding=True)
        for split in cv.split("2018-01-01", "2024-12-31"):
            print(split.train_start, split.train_end,
                  split.test_start, split.test_end)
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 0,
        expanding: bool = True,
    ) -> None:
        if n_splits < 2:
            msg = f"n_splits must be >= 2, got {n_splits}"
            raise ValueError(msg)
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.expanding = expanding

    def split(self, start: str, end: str) -> list[CVSplit]:
        """Generate train/test date splits.

        Parameters
        ----------
        start, end : str
            Overall date range in ISO format.

        Returns
        -------
        list[CVSplit]
            One split per fold.
        """
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        total_days = (end_dt - start_dt).days

        if self.expanding:
            return self._expanding_splits(start_dt, end_dt, total_days)
        return self._sliding_splits(start_dt, end_dt, total_days)

    def _expanding_splits(
        self,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        total_days: int,
    ) -> list[CVSplit]:
        """Generate expanding window splits.

        Divides the total period into (n_splits + 1) equal blocks.
        Split i uses blocks [0..i] for training and block [i+1] for testing.
        """
        n_blocks = self.n_splits + 1
        block_size = total_days / n_blocks

        splits: list[CVSplit] = []
        for i in range(self.n_splits):
            train_start = start_dt
            train_end = start_dt + pd.DateOffset(
                days=int(block_size * (i + 1)) - 1,
            )

            test_start = train_end + pd.DateOffset(
                days=1 + self.embargo_days,
            )
            test_end = start_dt + pd.DateOffset(
                days=int(block_size * (i + 2)) - 1,
            )

            # Clip to data range
            if test_end > end_dt:
                test_end = end_dt
            if test_start > end_dt:
                break

            splits.append(CVSplit(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            ))

        return splits

    def _sliding_splits(
        self,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        total_days: int,
    ) -> list[CVSplit]:
        """Generate fixed-size sliding window splits.

        Each split has the same train and test window sizes.
        """
        # Train and test each get equal portion within a split
        # Total span needed per split = train + embargo + test
        # We distribute: train = test = total / (n_splits + 1) * factor
        n_blocks = self.n_splits + 1
        block_size = total_days / n_blocks
        train_days = int(block_size)
        test_days = int(block_size)

        # Calculate step between splits
        remaining = total_days - train_days - self.embargo_days - test_days
        if self.n_splits > 1:
            step = remaining / (self.n_splits - 1)
        else:
            step = 0

        splits: list[CVSplit] = []
        for i in range(self.n_splits):
            offset = int(i * step)
            train_start = start_dt + pd.DateOffset(days=offset)
            train_end = train_start + pd.DateOffset(days=train_days - 1)
            test_start = train_end + pd.DateOffset(
                days=1 + self.embargo_days,
            )
            test_end = test_start + pd.DateOffset(days=test_days - 1)

            if test_end > end_dt:
                test_end = end_dt
            if test_start > end_dt:
                break

            splits.append(CVSplit(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            ))

        return splits

    def cross_validate(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        broker_factory: Callable[[], Broker],
        start: str,
        end: str,
        paramset: ParameterSet | None = None,
        metric: str | Callable[[RunResult], float] = "sharpe_ratio",
        metric_direction: str | None = None,
        initial_cash: float = 100_000.0,
        universe: UniverseProvider | None = None,
    ) -> CVResult:
        """Run cross-validation.

        If ``paramset`` is ``None``, runs the strategy as-is on each test
        split. If ``paramset`` is provided, runs :class:`GridSearch` on
        each training split and evaluates the best params on the test split.

        Parameters
        ----------
        strategy : Strategy
            Base strategy definition.
        market : MarketDataProvider
            Data provider.
        broker_factory : Callable[[], Broker]
            Factory returning a fresh broker per run.
        start, end : str
            Overall date range.
        paramset : ParameterSet or None
            If provided, optimize params on each training fold.
        metric : str or callable
            Metric to evaluate / optimize.
        metric_direction : str or None
            ``"maximize"`` or ``"minimize"``.
        initial_cash : float
            Starting capital per run.

        Returns
        -------
        CVResult
        """
        direction = _resolve_direction(metric, metric_direction)
        metric_name = metric if isinstance(metric, str) else "<custom>"

        cv_splits = self.split(start, end)
        logger.info(
            "TimeSeriesCV: %d splits (expanding=%s, embargo=%d days)",
            len(cv_splits), self.expanding, self.embargo_days,
        )

        engine = Engine()
        split_results: list[CVSplitResult] = []

        for i, split in enumerate(cv_splits):
            logger.info(
                "  Split %d/%d: train [%s, %s] -> test [%s, %s]",
                i + 1, len(cv_splits),
                split.train_start, split.train_end,
                split.test_start, split.test_end,
            )

            best_params: dict[str, dict[str, Any]] | None = None
            is_metric: float | None = None

            if paramset is not None:
                # Optimize on training period
                gs = GridSearch(paramset)
                search_result = gs.run(
                    strategy=strategy,
                    market=market,
                    broker_factory=broker_factory,
                    start=split.train_start,
                    end=split.train_end,
                    metric=metric,
                    metric_direction=metric_direction,
                    initial_cash=initial_cash,
                    universe=universe,
                )
                best_params = search_result.best.params
                is_metric = search_result.best.metric_value
                oos_strategy = _apply_params(strategy, best_params)
            else:
                oos_strategy = strategy

            # Evaluate on test period
            oos_result = engine.run(
                oos_strategy,
                market=market,
                broker=broker_factory(),
                start=split.test_start,
                end=split.test_end,
                initial_cash=initial_cash,
                universe=universe,
            )

            oos_metric = _extract_metric(oos_result, metric)
            logger.info(
                "    OOS %s = %.4f", metric_name, oos_metric,
            )

            split_results.append(CVSplitResult(
                split=split,
                best_params=best_params,
                in_sample_metric=is_metric,
                oos_result=oos_result,
            ))

        return CVResult(
            splits=split_results,
            metric=metric_name,
            metric_direction=direction,
        )
