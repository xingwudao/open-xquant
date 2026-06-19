import pandas as pd

from oxq.portfolio.optimizers import SignalPositionOptimizer


def _frame(signal):
    return pd.DataFrame(
        {"roc_timing": [signal]},
        index=pd.date_range("2024-01-01", periods=1, tz="UTC"),
    )


def test_signal_position_optimizer_maps_buy_sell_hold_with_state():
    optimizer = SignalPositionOptimizer(signal_col="roc_timing", weight=1.0)

    assert optimizer.optimize({"510300.SS": _frame(1.0)}, {"510300.SS": _frame(1.0)}) == {"510300.SS": 1.0}
    assert optimizer.optimize({"510300.SS": _frame(-1.0)}, {"510300.SS": _frame(-1.0)}) == {"510300.SS": 1.0}
    assert optimizer.optimize({"510300.SS": _frame(0.0)}, {"510300.SS": _frame(0.0)}) == {"CASH": 1.0}
    assert optimizer.optimize({"510300.SS": _frame(-1.0)}, {"510300.SS": _frame(-1.0)}) == {"CASH": 1.0}
