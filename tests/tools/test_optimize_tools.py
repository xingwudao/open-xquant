"""Tests for optimization tool broker configuration."""

from __future__ import annotations

import pytest

import oxq.tools.optimize as optimize_tools
from oxq.tools import session


@pytest.fixture(autouse=True)
def _reset_session():
    session.clear()


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        (
            optimize_tools.grid_search,
            {
                "strategy": "strategy",
                "paramset": "params",
                "start": "2024-01-01",
                "end": "2024-12-31",
                "symbols": ["AAPL"],
            },
        ),
        (
            optimize_tools.walk_forward,
            {
                "strategy": "strategy",
                "paramset": "params",
                "symbols": ["AAPL"],
                "start": "2024-01-01",
                "end": "2024-12-31",
                "train_period": "90D",
                "test_period": "30D",
            },
        ),
        (
            optimize_tools.cross_validate,
            {
                "strategy": "strategy",
                "symbols": ["AAPL"],
                "start": "2024-01-01",
                "end": "2024-12-31",
            },
        ),
    ],
)
def test_optimization_tools_pass_insufficient_cash_policy_to_broker_factory(
    monkeypatch,
    tmp_path,
    tool,
    kwargs,
) -> None:
    session._strategies["strategy"] = object()
    session._paramsets["params"] = object()
    captured: dict[str, object] = {}

    def broker_factory_probe(
        fee_rate,
        fee_min,
        slippage_rate,
        insufficient_cash_policy,
    ):
        captured["insufficient_cash_policy"] = insufficient_cash_policy
        raise RuntimeError("broker factory probe complete")

    monkeypatch.setattr(optimize_tools, "_broker_factory", broker_factory_probe)

    with pytest.raises(RuntimeError, match="broker factory probe complete"):
        tool(
            **kwargs,
            data_dir=str(tmp_path),
            insufficient_cash_policy="reject",
        )

    assert captured == {"insufficient_cash_policy": "reject"}
