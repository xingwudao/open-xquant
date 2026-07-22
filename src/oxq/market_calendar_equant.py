"""Trading calendar integration with edatatools.

Provides oxq-compatible calendar utilities backed by edatatools'
Chinese A-share trading calendar (``cn_calendar``) and configurable
regional calendars.

Usage::

    from oxq.market_calendar_equant import get_calendar, date_range, is_trading_day

    cal = get_calendar("CN")
    dates = date_range("2024-01-01", "2024-12-31", region="CN")
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class EQuantCalendar:
    """Trading calendar backed by edatatools.

    Wraps edatatools' ``TradingCalendar`` and ``cn_calendar`` singleton
    for use within the oxq ecosystem.

    Parameters
    ----------
    region : str
        Calendar region identifier ("CN", "US", etc.).
    """

    def __init__(self, region: str = "CN") -> None:
        self._region = region
        self._calendar = None

    @property
    def calendar(self):
        """Lazy-load the edatatools calendar."""
        if self._calendar is None:
            import edatatools
            if self._region == "CN":
                self._calendar = edatatools.cn_calendar
            else:
                # For non-CN regions, create a custom calendar
                self._calendar = edatatools.TradingCalendar(
                    region=self._region, dates=None,
                )
        return self._calendar

    def is_trading_day(self, date) -> bool:
        """Check if *date* is a trading day."""
        import edatatools
        return edatatools.date_is_bus_date(date, region=self._region)

    def trading_days(
        self,
        start: str,
        end: Optional[str] = None,
        n: Optional[int] = None,
    ) -> pd.DatetimeIndex:
        """Return trading days between *start* and *end*."""
        import edatatools
        return edatatools.date_range(start, end, n, region=self._region)

    def next_trading_day(self, date, shift: int = 1) -> pd.Timestamp:
        """Return the next (or previous) trading day."""
        import edatatools
        return edatatools.date_to_bus_date(
            date, region=self._region, shift=shift, forward=(shift > 0),
        )

    def trading_days_between(self, from_date, to_date) -> int:
        """Count trading days between two dates."""
        import edatatools
        return edatatools.date_bus_diff(from_date, to_date, region=self._region)


# Module-level convenience -----------------------------------------------------------------

_calendars: dict[str, EQuantCalendar] = {}


def get_calendar(region: str = "CN") -> EQuantCalendar:
    """Get or create a cached EQuantCalendar for *region*."""
    if region not in _calendars:
        _calendars[region] = EQuantCalendar(region)
    return _calendars[region]


def date_range(
    start: str,
    end: Optional[str] = None,
    n: Optional[int] = None,
    region: str = "CN",
) -> pd.DatetimeIndex:
    """Return trading days for *region* between *start* and *end*."""
    return get_calendar(region).trading_days(start, end, n)


def is_trading_day(date, region: str = "CN") -> bool:
    """Check if *date* is a trading day in *region*."""
    return get_calendar(region).is_trading_day(date)
