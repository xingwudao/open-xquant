"""Adapter layer bridging open-xquant Protocols to eQuant-Py backing implementations.

Each adapter wraps an eQuant-Py function, satisfying an oxq Protocol
(Indicator, Signal, MarketDataProvider, etc.) while delegating computation
to the eQuant toolkit layer.
"""

from oxq.adapters.equant import EQuantAdapter, from_panel, to_panel

__all__ = [
    "EQuantAdapter",
    "to_panel",
    "from_panel",
]
