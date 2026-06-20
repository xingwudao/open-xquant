import pandas as pd

from oxq.core.types import PortfolioOptimizer
from oxq.portfolio.optimizers import SignalToPositionOptimizer


def test_signal_to_position_is_optimizer() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing")
    assert isinstance(optimizer, PortfolioOptimizer)
    assert optimizer.name == "SignalToPosition"


def test_signal_to_position_maps_buy_sell_hold() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing", buy_weight=1.0, sell_weight=0.0)

    buy = optimizer.optimize(
        {"CSI300": pd.DataFrame({"timing": ["BUY"]})},
        {},
    )
    hold = optimizer.optimize(
        {"CSI300": pd.DataFrame({"timing": ["HOLD"]})},
        {},
    )
    sell = optimizer.optimize(
        {"CSI300": pd.DataFrame({"timing": ["SELL"]})},
        {},
    )

    assert buy == {"CSI300": 1.0}
    assert hold == {"CSI300": 1.0}
    assert sell == {"CASH": 1.0}


def test_signal_to_position_only_skips_rebalance_for_hold() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing")

    optimizer.optimize({"CSI300": pd.DataFrame({"timing": ["BUY"]})}, {})
    assert optimizer.skip_rebalance is False

    optimizer.optimize({"CSI300": pd.DataFrame({"timing": ["BUY"]})}, {})
    assert optimizer.skip_rebalance is False

    optimizer.optimize({"CSI300": pd.DataFrame({"timing": ["HOLD"]})}, {})
    assert optimizer.skip_rebalance is True


def test_signal_to_position_reset_symbols_clears_exited_latches() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing")

    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["BUY"]})}, {})
    optimizer.reset_symbols(["AAA"])

    assert optimizer.optimize({"AAA": pd.DataFrame({"timing": ["HOLD"]})}, {}) == {"CASH": 1.0}


def test_signal_to_position_reset_clears_run_latches() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing")

    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["BUY"]})}, {})
    optimizer.reset()

    assert optimizer.skip_rebalance is False
    assert optimizer.optimize({"AAA": pd.DataFrame({"timing": ["HOLD"]})}, {}) == {"CASH": 1.0}


def test_signal_to_position_preserves_hold_weight_when_other_symbol_sells() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing")

    assert optimizer.optimize(
        {
            "AAA": pd.DataFrame({"timing": ["BUY"]}),
            "BBB": pd.DataFrame({"timing": ["BUY"]}),
        },
        {},
    ) == {"AAA": 0.5, "BBB": 0.5}

    assert optimizer.optimize(
        {
            "AAA": pd.DataFrame({"timing": ["SELL"]}),
            "BBB": pd.DataFrame({"timing": ["HOLD"]}),
        },
        {},
    ) == {"BBB": 0.5, "CASH": 0.5}


def test_hold_starts_in_cash() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing")

    result = optimizer.optimize(
        {"CSI300": pd.DataFrame({"timing": ["HOLD"]})},
        {},
    )

    assert result == {"CASH": 1.0}


def test_unknown_signal_value_fails() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing")

    try:
        optimizer.optimize({"CSI300": pd.DataFrame({"timing": ["WAIT"]})}, {})
    except ValueError as exc:
        assert "expected BUY, SELL, or HOLD" in str(exc)
    else:
        raise AssertionError("SignalToPosition accepted an unknown value")
