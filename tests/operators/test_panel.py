from __future__ import annotations

import pandas as pd
import pytest

from oxq.operators.errors import DuplicateKeyError, InvalidPanelError, MissingColumnError
from oxq.operators.panel import QuantPanelAdapter
from oxq.operators.types import OperatorContext


def test_daily_frames_round_trip_without_mutating_input(daily_context, daily_symbol_frames) -> None:
    snapshots = {code: frame.copy(deep=True) for code, frame in daily_symbol_frames.items()}

    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)

    assert list(panel.columns) == ["date", "code", "close", "volume"]
    assert panel[["date", "code"]].values.tolist() == [
        [pd.Timestamp("2026-01-05"), "000001.SZ"],
        [pd.Timestamp("2026-01-05"), "600000.SH"],
        [pd.Timestamp("2026-01-06"), "000001.SZ"],
        [pd.Timestamp("2026-01-06"), "600000.SH"],
        [pd.Timestamp("2026-01-07"), "000001.SZ"],
        [pd.Timestamp("2026-01-07"), "600000.SH"],
    ]
    assert panel["date"].dt.tz is None
    restored = QuantPanelAdapter.to_symbol_frames(panel, daily_context)
    for code, original in snapshots.items():
        pd.testing.assert_frame_equal(daily_symbol_frames[code], original)
        pd.testing.assert_frame_equal(restored[code], original, check_freq=False)


def test_intraday_panel_is_timezone_aware_and_canonicalized_to_utc() -> None:
    context = OperatorContext(
        timezone="Asia/Shanghai",
        calendar="XSHG",
        frequency="1min",
        timestamp_semantics="bar_close",
        currency="CNY",
        price_adjustment="raw",
        data_version="fixture-v1",
        source="fake",
        evaluation_time="intraday_t",
    )
    frames = {
        "000001.SZ": pd.DataFrame(
            {"close": [10.0, 10.1]},
            index=pd.DatetimeIndex(["2026-01-05 09:31", "2026-01-05 09:32"], tz="Asia/Shanghai"),
        )
    }

    panel = QuantPanelAdapter.to_panel(frames, context)

    assert str(panel["date"].dt.tz) == "UTC"
    assert panel.loc[0, "date"] == pd.Timestamp("2026-01-05 01:31", tz="UTC")
    restored = QuantPanelAdapter.to_symbol_frames(panel, context)
    assert str(restored["000001.SZ"].index.tz) == "Asia/Shanghai"
    assert restored["000001.SZ"].index[0] == frames["000001.SZ"].index[0]


def test_intraday_rejects_naive_or_mixed_timezone_indexes() -> None:
    context = OperatorContext(
        timezone="Asia/Shanghai",
        calendar="XSHG",
        frequency="1min",
        timestamp_semantics="bar_close",
        currency="CNY",
        price_adjustment="raw",
        data_version="fixture-v1",
        source="fake",
        evaluation_time="intraday_t",
    )
    aware = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-01-05 09:31"], tz="Asia/Shanghai"))
    naive = pd.DataFrame({"close": [2.0]}, index=pd.DatetimeIndex(["2026-01-05 09:31"]))

    with pytest.raises(InvalidPanelError, match="timezone-aware"):
        QuantPanelAdapter.to_panel({"aware": aware, "naive": naive}, context)


def test_empty_intraday_panel_keeps_timezone_aware_date_dtype() -> None:
    context = OperatorContext(
        timezone="Asia/Shanghai",
        calendar="XSHG",
        frequency="1min",
        timestamp_semantics="bar_close",
        currency="CNY",
        price_adjustment="raw",
        data_version="fixture-v1",
        source="fake",
        evaluation_time="intraday_t",
    )

    panel = QuantPanelAdapter.to_panel({}, context)

    assert isinstance(panel["date"].dtype, pd.DatetimeTZDtype)
    assert str(panel["date"].dt.tz) == "UTC"
    QuantPanelAdapter.validate_panel(panel, context)


def test_panel_rejects_duplicate_keys_and_missing_identifiers(daily_context) -> None:
    duplicate = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05")],
            "code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 11.0],
        }
    )
    with pytest.raises(DuplicateKeyError, match="duplicate"):
        QuantPanelAdapter.validate_panel(duplicate, daily_context)
    with pytest.raises(MissingColumnError, match="code"):
        QuantPanelAdapter.validate_panel(duplicate.drop(columns="code"), daily_context)


def test_symbol_frames_reject_reserved_quant_panel_columns(daily_context) -> None:
    frame = pd.DataFrame(
        {"date": ["shadow"], "close": [10.0]},
        index=pd.DatetimeIndex(["2026-01-05"], tz="Asia/Shanghai"),
    )
    with pytest.raises(InvalidPanelError, match="reserved"):
        QuantPanelAdapter.to_panel({"000001.SZ": frame}, daily_context)


def test_panel_rejects_non_session_daily_timestamps(daily_context) -> None:
    panel = pd.DataFrame(
        {"date": [pd.Timestamp("2026-01-05 12:00")], "code": ["000001.SZ"], "close": [10.0]}
    )
    with pytest.raises(InvalidPanelError, match="normalized session dates"):
        QuantPanelAdapter.validate_panel(panel, daily_context)


def test_validate_output_enforces_declared_alignment(daily_context, daily_symbol_frames) -> None:
    input_panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    output = input_panel[["date", "code"]].copy()
    output["sma_2"] = [None, None, 10.5, 19.0, 11.5, 19.5]

    QuantPanelAdapter.validate_output(input_panel, output, daily_context, alignment="canonical_order")
    with pytest.raises(InvalidPanelError, match="canonical_order"):
        QuantPanelAdapter.validate_output(
            input_panel,
            output.iloc[::-1].reset_index(drop=True),
            daily_context,
            alignment="canonical_order",
        )
    with pytest.raises(InvalidPanelError, match="same keys"):
        QuantPanelAdapter.validate_output(
            input_panel,
            output.iloc[:-1],
            daily_context,
            alignment="preserve_input_order",
        )


def test_explicit_keyed_output_allows_unique_subset(daily_context, daily_symbol_frames) -> None:
    input_panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    output = input_panel.loc[input_panel["code"] == "000001.SZ", ["date", "code"]].copy()
    output["signal"] = 1.0

    QuantPanelAdapter.validate_output(
        input_panel,
        output,
        daily_context,
        alignment="explicit_keyed_output",
    )
