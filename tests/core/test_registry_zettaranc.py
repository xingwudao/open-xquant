"""Built-in registration of components added for TongDaXin-style strategies."""

from oxq.core.registry import _INDICATOR_REGISTRY, _load_builtins


def test_zettaranc_indicators_are_builtin() -> None:
    _load_builtins()
    assert {"TdxSMA", "LinearCombination"} <= set(_INDICATOR_REGISTRY)


def test_formula_event_is_builtin() -> None:
    from oxq.core.registry import _SIGNAL_REGISTRY

    _load_builtins()
    assert "FormulaEvent" in _SIGNAL_REGISTRY


def test_formula_exit_rule_is_builtin() -> None:
    from oxq.core.registry import _RULE_REGISTRY

    _load_builtins()
    assert "FormulaExitRule" in _RULE_REGISTRY


def test_entry_stop_rules_are_builtin() -> None:
    from oxq.core.registry import _RULE_REGISTRY

    _load_builtins()
    assert {"EntryBarLowStopRule", "TimeStopRule"} <= set(_RULE_REGISTRY)
