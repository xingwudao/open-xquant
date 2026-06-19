from __future__ import annotations

SUPPORTED_MARKET_CALENDARS = frozenset({"XNYS", "ARCX", "XSHG", "XSHE"})
EXCHANGE_CALENDAR_ALIASES = {
    # exchange_calendars does not expose XSHE; use the shared mainland China session calendar.
    "XSHE": "XSHG",
}


def normalize_exchange_calendar(calendar: str) -> str:
    return EXCHANGE_CALENDAR_ALIASES.get(calendar, calendar)


def is_supported_market_calendar(calendar: object) -> bool:
    if not isinstance(calendar, str):
        return False
    return calendar in SUPPORTED_MARKET_CALENDARS
