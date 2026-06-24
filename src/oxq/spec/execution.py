"""Effective execution semantics for strategy specs."""

from __future__ import annotations

from dataclasses import dataclass

from oxq.spec.schema import ExecutionSection


@dataclass(frozen=True)
class EffectiveExecution:
    order_timing: str
    price_bar: str
    price_type: str
    fill_price_mode: str
    compatibility_source: str


_LEGACY_TO_EXPLICIT: dict[str, tuple[str, str, str]] = {
    "next_open": ("next_session_open", "next_session", "open"),
    "next_close": ("next_session_close", "next_session", "close"),
    "next_mid": ("next_session_mid", "next_session", "mid"),
    "next_avg": ("next_session_avg", "next_session", "avg"),
    "next_hl2": ("next_session_hl2", "next_session", "hl2"),
    "close": ("same_session_close", "same_session", "close"),
    "mid": ("same_session_close", "same_session", "mid"),
}

_EXPLICIT_TO_LEGACY: dict[tuple[str, str, str], str] = {
    ("next_session_open", "next_session", "open"): "next_open",
    ("next_session_close", "next_session", "close"): "next_close",
    ("next_session_mid", "next_session", "mid"): "next_mid",
    ("next_session_avg", "next_session", "avg"): "next_avg",
    ("next_session_hl2", "next_session", "hl2"): "next_hl2",
    ("same_session_close", "same_session", "close"): "close",
    ("same_session_close", "same_session", "mid"): "mid",
}


def derive_execution_semantics(execution: ExecutionSection) -> EffectiveExecution:
    """Resolve explicit execution fields and legacy fill mode into one model."""
    explicit_values = (execution.order_timing, execution.price_bar, execution.price_type)
    explicit_count = sum(bool(value) for value in explicit_values)

    if 0 < explicit_count < len(explicit_values):
        raise ValueError("execution.order_timing, execution.price_bar, and execution.price_type must be provided together")

    if explicit_count == 0:
        legacy = _derive_from_legacy(execution.fill_price_mode)
        order_timing, price_bar, price_type = legacy
        return EffectiveExecution(
            order_timing=order_timing,
            price_bar=price_bar,
            price_type=price_type,
            fill_price_mode=execution.fill_price_mode,
            compatibility_source="legacy_fill_price_mode",
        )

    fill_price_mode = _derive_fill_price_mode(explicit_values)
    legacy_fill_mode_is_explicit = getattr(execution, "_fill_price_mode_explicit", False)
    legacy_fill_mode_is_default = execution.fill_price_mode == "next_open" and not legacy_fill_mode_is_explicit
    if execution.fill_price_mode and not legacy_fill_mode_is_default and execution.fill_price_mode != fill_price_mode:
        raise ValueError(
            "execution semantics conflict: "
            f"fill_price_mode={execution.fill_price_mode!r} does not match explicit semantics "
            f"{explicit_values!r} -> {fill_price_mode!r}"
        )

    return EffectiveExecution(
        order_timing=execution.order_timing,
        price_bar=execution.price_bar,
        price_type=execution.price_type,
        fill_price_mode=fill_price_mode,
        compatibility_source="explicit_fields",
    )


def _derive_from_legacy(fill_price_mode: str) -> tuple[str, str, str]:
    try:
        return _LEGACY_TO_EXPLICIT[fill_price_mode]
    except KeyError as exc:
        valid = ", ".join(sorted(_LEGACY_TO_EXPLICIT))
        raise ValueError(f"Unsupported execution.fill_price_mode {fill_price_mode!r}; expected one of: {valid}") from exc


def _derive_fill_price_mode(explicit_values: tuple[str, str, str]) -> str:
    try:
        return _EXPLICIT_TO_LEGACY[explicit_values]
    except KeyError as exc:
        order_timing, price_bar, price_type = explicit_values
        raise ValueError(
            "Unsupported execution semantics: "
            f"order_timing={order_timing!r}, price_bar={price_bar!r}, price_type={price_type!r}"
        ) from exc
