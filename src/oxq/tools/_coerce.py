"""Parameter coercion for tool boundaries.

LLM clients may send numeric values as strings in JSON tool calls.
This module coerces params to match the SDK's strict type annotations
before passing them to compute() methods.
"""

from __future__ import annotations

from typing import Any, get_type_hints

# Types we know how to coerce from string
_COERCIBLE: dict[type, type] = {
    float: float,
    int: int,
    bool: bool,
}


def coerce_compute_params(instance: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Coerce *params* to match the type annotations of *instance.compute()*.

    Returns a new dict — the original is not mutated.
    """
    try:
        hints = get_type_hints(instance.compute)
    except Exception:
        return dict(params)
    hints.pop("return", None)
    hints.pop("mktdata", None)

    out = dict(params)
    for key, value in out.items():
        expected = hints.get(key)
        if expected is None or not isinstance(expected, type):
            continue
        if isinstance(value, expected):
            continue
        coercer = _COERCIBLE.get(expected)
        if coercer is not None and isinstance(value, str):
            out[key] = coercer(value)
    return out
