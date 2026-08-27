"""Security regressions for the stdlib-first exact-wheel child."""

from __future__ import annotations

from oxq.operators import _exact_wheel_child


def test_exact_child_has_no_ambient_platform_runtime_allowlist() -> None:
    assert not hasattr(_exact_wheel_child, "_PLATFORM_RUNTIME_ROOTS")
