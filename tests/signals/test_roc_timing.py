import pandas as pd

from oxq.core.types import Signal
from oxq.signals.roc_timing import ROCTiming


def test_roc_timing_is_signal() -> None:
    assert isinstance(ROCTiming(), Signal)
    assert ROCTiming().name == "ROCTiming"


def test_fixed_threshold_outputs_buy_sell_hold() -> None:
    frame = pd.DataFrame({"roc_120": [-8.0, -3.0, 0.0, 7.0]})

    result = ROCTiming().compute(
        frame,
        column="roc_120",
        mode="fixed",
        bottom=-5.0,
        top=5.0,
    )

    assert result.tolist() == ["BUY", "HOLD", "HOLD", "SELL"]


def test_rolling_quantile_thresholds_are_causal() -> None:
    frame = pd.DataFrame({"roc_120": [-5.0, -4.0, -3.0, 4.0, 5.0, 6.0]})

    result = ROCTiming().compute(
        frame,
        column="roc_120",
        mode="rolling_quantile",
        q_window=3,
        q_bottom=0.05,
        q_top=0.95,
    )

    assert result.iloc[0] == "HOLD"
    assert result.iloc[1] == "HOLD"
    assert result.iloc[2] == "HOLD"
    assert result.iloc[3] == "SELL"
    assert result.iloc[4] == "SELL"
    assert result.iloc[5] == "SELL"


def test_rolling_quantile_collapsed_bands_hold() -> None:
    frame = pd.DataFrame({"roc_120": [0.0, 0.0, 0.0, 0.0]})

    result = ROCTiming().compute(
        frame,
        column="roc_120",
        mode="rolling_quantile",
        q_window=3,
        q_bottom=0.05,
        q_top=0.95,
    )

    assert result.tolist() == ["HOLD", "HOLD", "HOLD", "HOLD"]


def test_invalid_mode_raises_clear_error() -> None:
    frame = pd.DataFrame({"roc_120": [1.0]})

    try:
        ROCTiming().compute(frame, column="roc_120", mode="bad")
    except ValueError as exc:
        assert "mode must be 'fixed' or 'rolling_quantile'" in str(exc)
    else:
        raise AssertionError("ROCTiming accepted an invalid mode")


def test_fixed_thresholds_must_not_overlap() -> None:
    frame = pd.DataFrame({"roc_120": [0.0]})

    try:
        ROCTiming().compute(frame, column="roc_120", mode="fixed", bottom=5.0, top=-5.0)
    except ValueError as exc:
        assert "bottom must be less than top" in str(exc)
    else:
        raise AssertionError("ROCTiming accepted overlapping fixed thresholds")
