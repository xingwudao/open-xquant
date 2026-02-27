"""Tests for oxq.universe package-level exports."""

from __future__ import annotations


def test_universe_public_api() -> None:
    from oxq.universe import (
        Filter,
        FilterUniverse,
        StaticUniverse,
        UniverseProvider,
        UniverseSnapshot,
    )

    assert Filter is not None
    assert FilterUniverse is not None
    assert StaticUniverse is not None
    assert UniverseProvider is not None
    assert UniverseSnapshot is not None
