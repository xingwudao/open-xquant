"""Built-in registration of components added for TongDaXin-style strategies."""

from oxq.core.registry import _INDICATOR_REGISTRY, _load_builtins


def test_zettaranc_indicators_are_builtin() -> None:
    _load_builtins()
    assert {"TdxSMA", "LinearCombination"} <= set(_INDICATOR_REGISTRY)
