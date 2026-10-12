"""Indicators that reference earlier indicators validate in declaration order."""

from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate

_BASE = {
    "strategy_id": "kdj_chain",
    "universe": {"type": "static", "symbols": ["AAA"]},
    "signal": {
        "indicators": {},
        "rules": {"buy": {"type": "Formula", "params": {"expr": "j < 0"}}},
    },
}


def _spec(indicators: dict) -> StrategySpec:
    raw = {**_BASE, "signal": {**_BASE["signal"], "indicators": indicators}}
    return StrategySpec.from_dict(raw)


def _codes(spec: StrategySpec) -> set[str]:
    return {error["check"] for error in validate(spec).errors}


def test_kdj_chain_validates_in_declaration_order() -> None:
    spec = _spec({
        "rsv": {"type": "StochK", "params": {"period": 9}},
        "k": {"type": "TdxSMA", "params": {"column": "rsv", "n": 3, "m": 1}},
        "d": {"type": "TdxSMA", "params": {"column": "k", "n": 3, "m": 1}},
        "j": {"type": "LinearCombination", "params": {"columns": ["k", "d"], "weights": [3.0, -2.0]}},
    })
    assert "compute_dry_run_failed" not in _codes(spec)


def test_forward_reference_fails_dry_run() -> None:
    spec = _spec({
        "j": {"type": "LinearCombination", "params": {"columns": ["k", "d"], "weights": [3.0, -2.0]}},
        "rsv": {"type": "StochK", "params": {"period": 9}},
        "k": {"type": "TdxSMA", "params": {"column": "rsv", "n": 3, "m": 1}},
        "d": {"type": "TdxSMA", "params": {"column": "k", "n": 3, "m": 1}},
    })
    assert "compute_dry_run_failed" in _codes(spec)
