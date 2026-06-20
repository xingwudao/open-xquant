import pandas as pd

from oxq.signals.roc_timing import ROCTiming


def test_roc_timing_fixed_threshold_emits_buy_sell_hold_states():
    df = pd.DataFrame(
        {"close": [100.0, 80.0, 95.0, 130.0]},
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
    )

    signal = ROCTiming().compute(
        df,
        column="close",
        lookback=1,
        threshold_mode="fixed",
        buy_threshold=-0.10,
        sell_threshold=0.20,
        stop_loss_pct=0,
    )

    assert list(signal.dropna()) == [1.0, -1.0, 0.0]


def test_roc_timing_rolling_quantile_emits_buy_sell_hold_states():
    df = pd.DataFrame(
        {"close": [60.0, 60.0, 60.0, 60.0, 70.0, 60.0, 60.0, 70.0]},
        index=pd.date_range("2024-01-01", periods=8, tz="UTC"),
    )

    signal = ROCTiming().compute(
        df,
        column="close",
        lookback=1,
        threshold_mode="rolling",
        q_window=3,
        q_bottom=0.25,
        q_top=0.75,
        stop_loss_pct=0,
    )

    assert list(signal.dropna()) == [-1.0, -1.0, -1.0, 0.0, 1.0, -1.0, 0.0]
