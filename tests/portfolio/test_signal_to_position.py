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


def test_signal_to_position_initial_sell_does_not_open_position() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing", sell_weight=0.25)

    result = optimizer.optimize(
        {"AAA": pd.DataFrame({"timing": ["SELL"]})},
        {},
    )

    assert result == {"CASH": 1.0}


def test_signal_to_position_sell_weight_only_reduces_latched_position() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing", sell_weight=0.25)

    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["BUY"]})}, {})

    result = optimizer.optimize(
        {"AAA": pd.DataFrame({"timing": ["SELL"]})},
        {},
    )

    assert result == {"AAA": 0.25, "CASH": 0.75}


def test_signal_to_position_sell_after_reset_stays_in_cash() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing", sell_weight=0.25)

    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["BUY"]})}, {})
    optimizer.reset_symbols(["AAA"])

    result = optimizer.optimize(
        {"AAA": pd.DataFrame({"timing": ["SELL"]})},
        {},
    )

    assert result == {"CASH": 1.0}


def test_signal_to_position_sell_weight_requires_actual_position_snapshot() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing", sell_weight=0.25)

    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["BUY"]})}, {})
    optimizer.set_held_symbols([])

    result = optimizer.optimize(
        {"AAA": pd.DataFrame({"timing": ["SELL"]})},
        {},
    )

    assert result == {"CASH": 1.0}
    assert "AAA" not in optimizer.pending_reduction_symbols


def test_signal_to_position_pending_buy_sell_does_not_keep_buyable_target() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing", sell_weight=0.25)

    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["BUY"]})}, {})
    optimizer.set_held_symbols([])
    optimizer.set_pending_buy_symbols(["AAA"])

    result = optimizer.optimize(
        {"AAA": pd.DataFrame({"timing": ["SELL"]})},
        {},
    )

    assert result == {"CASH": 1.0}
    assert optimizer.pending_reduction_symbols == {"AAA"}

    optimizer.set_pending_buy_symbols([])
    optimizer.clear_pending_reductions(["AAA"])

    assert optimizer.optimize({"AAA": pd.DataFrame({"timing": ["HOLD"]})}, {}) == {"CASH": 1.0}
    assert "AAA" not in optimizer.pending_reduction_symbols


def test_signal_to_position_clears_completed_partial_sell_pending_state() -> None:
    optimizer = SignalToPositionOptimizer(signal="timing", sell_weight=0.25)

    optimizer.set_held_symbols(["AAA"])
    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["BUY"]})}, {})
    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["SELL"]})}, {})
    optimizer.clear_pending_reductions(["AAA"])
    optimizer.optimize({"AAA": pd.DataFrame({"timing": ["HOLD"]})}, {})

    assert "AAA" not in optimizer.pending_reduction_symbols
    assert optimizer.held_symbols == {"AAA"}


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
