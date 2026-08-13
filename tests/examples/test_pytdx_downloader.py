from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import examples.modules.pytdx_downloader as module
import pandas as pd
import pytest
from examples.modules.pytdx_downloader import PyTdxDownloader

from oxq.core.errors import DownloadError
from oxq.data.manifest import read_manifest, verify_manifest
from oxq.data.providers import Downloader


class FakeApi:
    def __init__(
        self,
        *,
        pages: dict[int, object],
        actions: object | None = None,
    ) -> None:
        self.pages = pages
        self.actions = [] if actions is None else actions
        self.bar_calls: list[tuple[int, int, str, int, int]] = []
        self.action_calls: list[tuple[int, str]] = []

    def get_security_bars(
        self,
        category: int,
        market: int,
        code: str,
        offset: int,
        count: int,
    ) -> object:
        self.bar_calls.append((category, market, code, offset, count))
        return self.pages.get(offset, [])

    def get_xdxr_info(self, market: int, code: str) -> object:
        self.action_calls.append((market, code))
        return self.actions


def bar(day: str, close: float, volume: object = 1000) -> dict[str, object]:
    return {
        "datetime": f"{day} 15:00",
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "vol": volume,
    }


def xdxr(
    day: str,
    *,
    category: int = 1,
    fenhong: object = 0.0,
    peigujia: object = 0.0,
    songzhuangu: object = 0.0,
    peigu: object = 0.0,
) -> dict[str, object]:
    parsed = module.datetime.strptime(day, "%Y-%m-%d")
    return {
        "year": parsed.year,
        "month": parsed.month,
        "day": parsed.day,
        "category": category,
        "fenhong": fenhong,
        "peigujia": peigujia,
        "songzhuangu": songzhuangu,
        "peigu": peigu,
    }


@pytest.fixture(autouse=True)
def forbid_real_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("tests must not open a real socket")

    monkeypatch.setattr(socket, "socket", fail_socket)


@pytest.mark.parametrize(
    "host",
    ["", " host", "host ", "http://host", "host:7709", "user@host", "a/b"],
)
def test_rejects_invalid_host(host: str) -> None:
    with pytest.raises(ValueError, match="host"):
        PyTdxDownloader(host=host)


@pytest.mark.parametrize("port", [True, 0, 65536, -1, 7709.0])
def test_rejects_invalid_port(port: object) -> None:
    with pytest.raises(ValueError, match="port"):
        PyTdxDownloader(host="quote.example", port=port)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "timeout",
    [True, "5", object(), 10**1000, 0.0, -1.0, float("nan"), float("inf")],
)
def test_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout"):
        PyTdxDownloader(host="quote.example", timeout=timeout)  # type: ignore[arg-type]


def test_rejects_non_boolean_auto_adjust() -> None:
    with pytest.raises(ValueError, match="auto_adjust"):
        PyTdxDownloader(host="quote.example", auto_adjust=1)  # type: ignore[arg-type]


def test_normalizes_supported_symbols() -> None:
    assert module._normalize_symbol("510300.sh") == ("510300.SH", 1, "510300")
    assert module._normalize_symbol("159919.sz") == ("159919.SZ", 0, "159919")


@pytest.mark.parametrize("symbol", ["510300", "SH510300", "510300.BJ", "abc.SH"])
def test_rejects_unsupported_symbols(symbol: str) -> None:
    with pytest.raises(ValueError, match="six digits.*SH.*SZ"):
        module._normalize_symbol(symbol)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("20200501", "2026-01-01"),
        ("2020-5-1", "2026-01-01"),
        ("2020-02-30", "2026-01-01"),
        ("2026-01-02", "2026-01-01"),
    ],
)
def test_rejects_invalid_date_range(start: str, end: str) -> None:
    with pytest.raises(ValueError, match="date"):
        module._parse_date_range(start, end)


def test_rejects_unlocalizable_date_before_connecting() -> None:
    with patch.object(module, "_connected_api") as connected:
        with pytest.raises(ValueError, match="date"):
            PyTdxDownloader(host="quote.example").download(
                "510300.SH", "2024-01-01", "9999-12-31"
            )

    connected.assert_not_called()


def test_missing_pytdx_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(DownloadError, match=r"uv run --with pytdx==1\.72"):
        module._load_pytdx()


def test_rejects_unreviewed_pytdx_version() -> None:
    hq = MagicMock()
    hq.TdxHq_API = MagicMock()

    with (
        patch.object(module.importlib, "import_module", return_value=hq),
        patch.object(module.metadata, "version", return_value="1.71"),
        pytest.raises(DownloadError, match=r"pytdx==1\.72"),
    ):
        module._load_pytdx()


def test_connected_api_uses_conservative_options_and_disconnects() -> None:
    api = MagicMock()
    api.connect.return_value = api
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with module._connected_api("quote.example", 7709, 4.5) as (opened, version):
            assert opened is api
            assert version == "1.72"

    api_class.assert_called_once_with(
        multithread=False,
        heartbeat=False,
        auto_retry=False,
        raise_exception=True,
    )
    api.connect.assert_called_once_with("quote.example", 7709, time_out=4.5)
    api.disconnect.assert_called_once_with()


def test_connected_api_rejects_false_connect_result() -> None:
    api = MagicMock()
    api.connect.return_value = False
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(DownloadError, match="Cannot connect"):
            with module._connected_api("quote.example", 7709, 5.0):
                pass

    api.disconnect.assert_called_once_with()


def test_connected_api_cleans_up_after_connect_exception() -> None:
    api = MagicMock()
    api.connect.side_effect = RuntimeError("connect failed")
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(DownloadError, match=r"quote\.example:7709"):
            with module._connected_api("quote.example", 7709, 5.0):
                pass

    api.disconnect.assert_called_once_with()


def test_connected_api_wraps_constructor_failure_with_endpoint() -> None:
    api_class = MagicMock(side_effect=RuntimeError("constructor failed"))

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(DownloadError, match=r"quote\.example:7709"):
            with module._connected_api("quote.example", 7709, 5.0):
                pass


def test_disconnect_failure_is_a_download_error() -> None:
    api = MagicMock()
    api.connect.return_value = api
    api.disconnect.side_effect = RuntimeError("disconnect failed")
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(DownloadError, match="disconnect"):
            with module._connected_api("quote.example", 7709, 5.0):
                pass


def test_disconnect_failure_does_not_mask_body_error() -> None:
    api = MagicMock()
    api.connect.return_value = api
    api.disconnect.side_effect = RuntimeError("disconnect failed")
    api_class = MagicMock(return_value=api)

    with patch.object(module, "_load_pytdx", return_value=(api_class, "1.72")):
        with pytest.raises(LookupError, match="body failed"):
            with module._connected_api("quote.example", 7709, 5.0):
                raise LookupError("body failed")


def test_fetches_800_bar_pages_backward_until_before_start() -> None:
    newest_days = pd.date_range("2022-01-01", periods=800, freq="D")
    newest = [
        bar(day.strftime("%Y-%m-%d"), 10.0 + number / 1000)
        for number, day in enumerate(newest_days)
    ]
    older = [bar("2020-04-30", 9.0), bar("2020-05-01", 9.1)]
    api = FakeApi(pages={0: newest, 800: older})

    frame = module._fetch_raw_bars(
        api,
        market=1,
        code="510300",
        start_date=module.date(2020, 5, 1),
        symbol="510300.SH",
    )

    assert api.bar_calls == [
        (9, 1, "510300", 0, 800),
        (9, 1, "510300", 800, 800),
    ]
    assert frame.index.is_monotonic_increasing
    assert frame.index.min().date().isoformat() == "2020-04-30"
    assert frame.index.max().date() == newest_days[-1].date()


def test_rejects_out_of_order_bar_within_full_page() -> None:
    days = pd.date_range("2022-01-01", periods=800, freq="D")
    page = [bar(day.strftime("%Y-%m-%d"), 10.0) for day in days]
    page[400]["datetime"] = "2019-01-01 15:00"
    api = FakeApi(pages={0: page})

    with pytest.raises(DownloadError, match="chronological order"):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(2020, 5, 1),
            "510300.SH",
        )


def test_rejects_bar_page_that_does_not_progress_backward() -> None:
    first_days = pd.date_range("2022-01-01", periods=800, freq="D")
    first = [bar(day.strftime("%Y-%m-%d"), 10.0) for day in first_days]
    second = [bar("2021-12-31", 9.0), bar("2022-01-02", 9.0)]
    api = FakeApi(pages={0: first, 800: second})

    with pytest.raises(DownloadError, match="chronological order"):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(2020, 5, 1),
            "510300.SH",
        )


def test_rejects_boundary_only_overlap_without_backward_progress() -> None:
    days = pd.date_range("2022-01-01", periods=800, freq="D")
    first = [bar(day.strftime("%Y-%m-%d"), 10.0) for day in days]
    api = FakeApi(pages={0: first, 800: [first[0]]})

    with pytest.raises(DownloadError, match="chronological order"):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(2020, 5, 1),
            "510300.SH",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "no bar response"),
        ({"datetime": "2024-01-01"}, "must be a list"),
        ([{"datetime": "bad"}], "required bar fields"),
        ([bar("2024-01-01", float("nan"))], "finite positive OHLC"),
        ([bar("2024-01-01", 10.0, True)], "safe non-negative integer volume"),
        ([bar("2024-01-01", 10.0, -1)], "safe non-negative integer volume"),
        ([bar("2024-01-01", 10.0, 2**63)], "safe non-negative integer volume"),
    ],
)
def test_rejects_invalid_bar_payload(payload: object, message: str) -> None:
    api = FakeApi(pages={0: payload})

    with pytest.raises(DownloadError, match=message):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(2020, 5, 1),
            "510300.SH",
        )


@pytest.mark.parametrize(
    ("decoded", "expected"),
    [
        (5.877471754111438e-39, 0),
        (32768.5, 32769),
        (1000.49, 1000),
    ],
)
def test_normalizes_lossy_pytdx_volume_decode(
    decoded: float,
    expected: int,
) -> None:
    assert module._volume(decoded, "510300.SH") == expected


def test_rejects_invalid_bar_date() -> None:
    payload = [bar("not-a-date", 10.0)]
    api = FakeApi(pages={0: payload})

    with pytest.raises(DownloadError, match="invalid bar date"):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(2020, 5, 1),
            "510300.SH",
        )


def test_rejects_nat_bar_date() -> None:
    payload = bar("2024-01-01", 10.0)
    payload["datetime"] = "NaT"

    with pytest.raises(DownloadError, match="invalid bar date"):
        module._parse_bar_page([payload], "510300.SH")


def test_rejects_bar_date_that_overflows_midnight_normalization() -> None:
    payload = bar("2024-01-01", 10.0)
    payload["datetime"] = "1677-09-21 15:00"

    with pytest.raises(DownloadError, match="invalid bar date"):
        module._parse_bar_page([payload], "510300.SH")


def test_rejects_inconsistent_ohlc() -> None:
    invalid = bar("2024-01-01", 10.0)
    invalid["high"] = 9.5
    api = FakeApi(pages={0: [invalid]})

    with pytest.raises(DownloadError, match="inconsistent OHLC"):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(2020, 5, 1),
            "510300.SH",
        )


def test_wraps_bar_request_errors() -> None:
    api = MagicMock()
    api.get_security_bars.side_effect = RuntimeError("protocol failed")

    with pytest.raises(DownloadError, match="bar request failed"):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(2020, 5, 1),
            "510300.SH",
        )


def test_rejects_repeated_full_page() -> None:
    days = pd.date_range("2022-01-01", periods=800, freq="D")
    page = [bar(day.strftime("%Y-%m-%d"), 10.0) for day in days]
    api = FakeApi(pages={0: page, 800: page})

    with pytest.raises(DownloadError, match="repeated a bar page"):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(1900, 1, 1),
            "510300.SH",
        )


def test_deduplicates_exact_cross_page_overlap() -> None:
    first = module._parse_bar_page([bar("2024-01-01", 10.0)], "510300.SH")
    second = module._parse_bar_page([bar("2024-01-01", 10.0)], "510300.SH")

    merged = module._merge_bar_pages([first, second], "510300.SH")

    assert len(merged) == 1


def test_rejects_conflicting_cross_page_dates() -> None:
    first = module._parse_bar_page([bar("2024-01-01", 10.0)], "510300.SH")
    second = module._parse_bar_page([bar("2024-01-01", 11.0)], "510300.SH")

    with pytest.raises(DownloadError, match="conflicting bars"):
        module._merge_bar_pages([first, second], "510300.SH")


def test_rejects_pagination_limit_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "_MAX_PAGES", 2)
    newer_days = pd.date_range("2022-01-01", periods=800, freq="D")
    older_days = pd.date_range("2019-01-01", periods=800, freq="D")
    api = FakeApi(
        pages={
            0: [bar(day.strftime("%Y-%m-%d"), 10.0) for day in newer_days],
            800: [bar(day.strftime("%Y-%m-%d"), 9.0) for day in older_days],
        }
    )

    with pytest.raises(DownloadError, match="exceeded 2 pages"):
        module._fetch_raw_bars(
            api,
            1,
            "510300",
            module.date(1900, 1, 1),
            "510300.SH",
        )


def test_cash_dividend_adjusts_ohlc_by_one_ratio_and_preserves_volume() -> None:
    frame = module._parse_bar_page(
        [
            bar("2024-01-01", 10.0),
            bar("2024-01-02", 10.2),
            bar("2024-01-03", 9.8),
        ],
        "510300.SH",
    )
    original_volume = frame["volume"].copy()

    adjusted, count = module._adjust_bars(
        frame,
        [xdxr("2024-01-03", fenhong=2.0)],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    ratio = 10.0 / 10.2
    assert adjusted.loc["2024-01-02", "close"] == pytest.approx(10.2 * ratio)
    assert adjusted.loc["2024-01-03", "close"] == pytest.approx(9.8)
    assert adjusted.loc["2024-01-02", "open"] == pytest.approx(10.1 * ratio)
    pd.testing.assert_series_equal(adjusted["volume"], original_volume)
    assert count == 1


def test_bonus_shares_use_theoretical_ex_price_ratio() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 12.0), bar("2024-01-02", 10.0)],
        "510300.SH",
    )

    adjusted, count = module._adjust_bars(
        frame,
        [xdxr("2024-01-02", songzhuangu=2.0)],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    assert adjusted.loc["2024-01-01", "close"] == pytest.approx(10.0)
    assert count == 1


def test_rights_issue_uses_price_and_share_ratio() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 12.0), bar("2024-01-02", 10.8)],
        "510300.SH",
    )

    adjusted, count = module._adjust_bars(
        frame,
        [xdxr("2024-01-02", peigu=2.0, peigujia=5.0)],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    assert adjusted.loc["2024-01-01", "close"] == pytest.approx(13.0 / 1.2)
    assert count == 1


def test_multiple_events_multiply_ratios_for_earlier_bars() -> None:
    frame = module._parse_bar_page(
        [
            bar("2024-01-01", 10.0),
            bar("2024-01-02", 10.2),
            bar("2024-01-03", 10.0),
            bar("2024-01-04", 9.0),
        ],
        "510300.SH",
    )

    adjusted, count = module._adjust_bars(
        frame,
        [
            xdxr("2024-01-03", fenhong=2.0),
            xdxr("2024-01-04", songzhuangu=1.0),
        ],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    expected = 10.0 * (10.0 / 10.2) * ((10.0 / 1.1) / 10.0)
    assert adjusted.loc["2024-01-01", "close"] == pytest.approx(expected)
    assert count == 2


def test_future_events_are_ignored_before_validation() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 10.1)],
        "510300.SH",
    )

    adjusted, count = module._adjust_bars(
        frame,
        [
            {
                "year": 2025,
                "month": 1,
                "day": 1,
                "category": 99,
            }
        ],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    pd.testing.assert_frame_equal(adjusted, frame)
    assert count == 0


def test_events_on_or_before_first_output_date_are_ignored() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 10.1)],
        "510300.SH",
    )

    adjusted, count = module._adjust_bars(
        frame,
        [{"year": 2024, "month": 1, "day": 1, "category": 99}],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    pd.testing.assert_frame_equal(adjusted, frame)
    assert count == 0


def test_identical_same_day_actions_are_deduplicated() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 9.9)],
        "510300.SH",
    )
    action = xdxr("2024-01-02", fenhong=1.0)

    adjusted, count = module._adjust_bars(
        frame,
        [action, dict(action)],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    assert adjusted.loc["2024-01-01", "close"] == pytest.approx(9.9)
    assert count == 1


def test_conflicting_same_day_actions_are_rejected() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 9.9)],
        "510300.SH",
    )

    with pytest.raises(DownloadError, match="conflicting corporate actions"):
        module._adjust_bars(
            frame,
            [
                xdxr("2024-01-02", fenhong=1.0),
                xdxr("2024-01-02", fenhong=2.0),
            ],
            module.date(2024, 1, 1),
            "510300.SH",
        )


@pytest.mark.parametrize("payload", [None, {}, "invalid"])
def test_rejects_invalid_action_response(payload: object) -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 9.9)],
        "510300.SH",
    )

    with pytest.raises(DownloadError, match="corporate-action response"):
        module._adjust_bars(
            frame,
            payload,
            module.date(2024, 1, 1),
            "510300.SH",
        )


@pytest.mark.parametrize(
    "record",
    [
        {"year": True, "month": 1, "day": 2, "category": 1},
        {"year": 10**1000, "month": 1, "day": 2, "category": 1},
        {"year": 2024, "month": 13, "day": 2, "category": 1},
        {"year": 2024, "month": 1, "day": 2, "category": True},
    ],
)
def test_rejects_invalid_action_date_or_category(
    record: dict[str, object],
) -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 9.9)],
        "510300.SH",
    )

    with pytest.raises(DownloadError, match="corporate-action"):
        module._adjust_bars(
            frame,
            [record],
            module.date(2024, 1, 1),
            "510300.SH",
        )


@pytest.mark.parametrize(
    ("category", "message"),
    [(11, "category 11"), (12, "category 12"), (99, "category 99")],
)
def test_rejects_relevant_unsupported_action_category(
    category: int,
    message: str,
) -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 9.9)],
        "510300.SH",
    )

    with pytest.raises(DownloadError, match=message):
        module._adjust_bars(
            frame,
            [xdxr("2024-01-02", category=category)],
            module.date(2024, 1, 1),
            "510300.SH",
        )


def test_ignores_documented_non_price_action_category() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 9.9)],
        "510300.SH",
    )

    adjusted, count = module._adjust_bars(
        frame,
        [xdxr("2024-01-02", category=5)],
        module.date(2024, 1, 1),
        "510300.SH",
    )

    pd.testing.assert_frame_equal(adjusted, frame)
    assert count == 0


@pytest.mark.parametrize(
    "value",
    [True, -1.0, float("nan"), float("inf"), None],
)
def test_rejects_invalid_adjustment_fields(value: object) -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 10.0), bar("2024-01-02", 9.9)],
        "510300.SH",
    )

    with pytest.raises(DownloadError, match="adjustment fields"):
        module._adjust_bars(
            frame,
            [xdxr("2024-01-02", fenhong=value)],
            module.date(2024, 1, 1),
            "510300.SH",
        )


def test_rejects_action_without_previous_close() -> None:
    frame = module._parse_bar_page([bar("2024-01-02", 9.9)], "510300.SH")

    with pytest.raises(DownloadError, match="No previous close"):
        module._adjust_bars(
            frame,
            [xdxr("2024-01-02", fenhong=1.0)],
            module.date(2024, 1, 1),
            "510300.SH",
        )


def test_rejects_non_positive_adjustment_reference() -> None:
    frame = module._parse_bar_page(
        [bar("2024-01-01", 1.0), bar("2024-01-02", 1.0)],
        "510300.SH",
    )

    with pytest.raises(DownloadError, match="invalid adjustment factor"):
        module._adjust_bars(
            frame,
            [xdxr("2024-01-02", fenhong=20.0)],
            module.date(2024, 1, 1),
            "510300.SH",
        )


def test_download_writes_standard_adjusted_artifacts(tmp_path: Path) -> None:
    api = FakeApi(
        pages={
            0: [
                bar("2024-01-02", 10.2),
                bar("2024-01-03", 9.8),
                bar("2024-01-04", 9.9),
            ]
        },
        actions=[xdxr("2024-01-03", fenhong=2.0)],
    )
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        path = PyTdxDownloader(host="quote.example").download(
            "510300.sh",
            "2024-01-02",
            "2024-01-03",
            tmp_path,
        )

    assert path == tmp_path / "510300.SH.parquet"
    frame = pd.read_parquet(path)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert str(frame.index.tz) == "Asia/Shanghai"
    assert frame["volume"].dtype == "int64"

    manifest = read_manifest(tmp_path / "510300.SH.manifest.json")
    assert manifest is not None
    assert manifest["provider"] == "pytdx"
    assert manifest["rows"] == 2
    assert manifest["extra"] == {
        "auto_adjust": True,
        "adjustment_method": "xdxr_ratio_yfinance_semantics",
        "adjustment_reference_date": "2024-01-04",
        "applied_event_count": 1,
        "bar_category": 9,
        "host": "quote.example",
        "period": "1d",
        "port": 7709,
        "pytdx_version": "1.72",
        "transport": "tdx_hq_tcp",
    }
    assert verify_manifest(path).status == "real"


def test_event_after_requested_end_adjusts_earlier_output(tmp_path: Path) -> None:
    api = FakeApi(
        pages={
            0: [
                bar("2024-01-01", 10.0),
                bar("2024-01-02", 10.2),
                bar("2024-01-03", 9.8),
                bar("2024-01-04", 9.9),
            ]
        },
        actions=[xdxr("2024-01-03", fenhong=2.0)],
    )
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        path = PyTdxDownloader(host="quote.example").download(
            "510300.SH",
            "2024-01-01",
            "2024-01-02",
            tmp_path,
        )

    frame = pd.read_parquet(path)
    expected_ratio = (10.2 - 0.2) / 10.2
    assert frame.loc["2024-01-02", "close"] == pytest.approx(
        10.2 * expected_ratio
    )
    manifest = read_manifest(tmp_path / "510300.SH.manifest.json")
    assert manifest is not None
    assert manifest["extra"]["applied_event_count"] == 1


def test_downloader_satisfies_protocol() -> None:
    downloader: Downloader = PyTdxDownloader(host="quote.example")
    assert isinstance(downloader, Downloader)


def test_no_auto_adjust_skips_actions_and_preserves_raw_prices(
    tmp_path: Path,
) -> None:
    api = FakeApi(
        pages={0: [bar("2024-01-01", 10.0), bar("2024-01-02", 10.2)]},
        actions=None,
    )
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        path = PyTdxDownloader(
            host="quote.example",
            auto_adjust=False,
        ).download("510300.SH", "2024-01-01", "2024-01-02", tmp_path)

    frame = pd.read_parquet(path)
    assert frame.loc["2024-01-01", "close"] == pytest.approx(10.0)
    assert api.action_calls == []
    manifest = read_manifest(tmp_path / "510300.SH.manifest.json")
    assert manifest is not None
    assert manifest["extra"]["auto_adjust"] is False
    assert manifest["extra"]["adjustment_method"] == "none"
    assert manifest["extra"]["applied_event_count"] == 0


def test_empty_requested_range_does_not_create_destination(tmp_path: Path) -> None:
    destination = tmp_path / "not-created"
    api = FakeApi(pages={0: [bar("2024-01-01", 10.0)]})
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        with pytest.raises(DownloadError, match="No data"):
            PyTdxDownloader(host="quote.example").download(
                "510300.SH",
                "2025-01-01",
                "2025-01-02",
                destination,
            )

    assert not destination.exists()


@pytest.mark.parametrize(
    ("pages", "actions", "message"),
    [
        ({0: [bar("2024-01-01", 10.0, volume=-1)]}, [], "volume"),
        ({0: [bar("2024-01-01", 10.0)]}, {}, "corporate-action response"),
    ],
)
def test_invalid_payload_does_not_create_destination(
    tmp_path: Path,
    pages: dict[int, object],
    actions: object,
    message: str,
) -> None:
    destination = tmp_path / "not-created"
    api = FakeApi(pages=pages, actions=actions)
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        with pytest.raises(DownloadError, match=message):
            PyTdxDownloader(host="quote.example").download(
                "510300.SH",
                "2024-01-01",
                "2024-01-01",
                destination,
            )

    assert not destination.exists()


def test_wraps_corporate_action_request_error(tmp_path: Path) -> None:
    api = FakeApi(pages={0: [bar("2024-01-01", 10.0)]})
    api.get_xdxr_info = MagicMock(side_effect=RuntimeError("protocol failed"))
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        with pytest.raises(DownloadError, match="corporate-action request failed"):
            PyTdxDownloader(host="quote.example").download(
                "510300.SH",
                "2024-01-01",
                "2024-01-01",
                tmp_path / "not-created",
            )


def test_download_many_uses_one_connection_and_both_market_mappings(
    tmp_path: Path,
) -> None:
    api = FakeApi(pages={0: [bar("2024-01-01", 10.0)]})
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        paths = PyTdxDownloader(host="quote.example").download_many(
            ["510300.SH", "159919.sz"],
            "2024-01-01",
            "2024-01-01",
            tmp_path,
        )

    assert list(paths) == ["510300.SH", "159919.sz"]
    assert connected.call_count == 1
    assert api.bar_calls == [
        (9, 1, "510300", 0, 800),
        (9, 0, "159919", 0, 800),
    ]
    assert api.action_calls == [(1, "510300"), (0, "159919")]


def test_empty_download_many_does_not_connect() -> None:
    with patch.object(module, "_connected_api") as connected:
        result = PyTdxDownloader(host="quote.example").download_many(
            [],
            "2024-01-01",
            "2024-01-01",
        )

    assert result == {}
    connected.assert_not_called()


def test_download_many_retains_first_result_and_stops_after_failure(
    tmp_path: Path,
) -> None:
    api = MagicMock()
    requested_codes: list[str] = []

    def get_bars(
        category: int,
        market: int,
        code: str,
        offset: int,
        count: int,
    ) -> object:
        del category, market, count
        requested_codes.append(code)
        if code == "510300":
            return [bar("2024-01-01", 10.0)] if offset == 0 else []
        if code == "159919":
            raise RuntimeError("second symbol failed")
        raise AssertionError("third symbol must not be requested")

    api.get_security_bars.side_effect = get_bars
    with patch.object(module, "_connected_api") as connected:
        connected.return_value.__enter__.return_value = (api, "1.72")
        with pytest.raises(DownloadError, match="bar request failed"):
            PyTdxDownloader(
                host="quote.example",
                auto_adjust=False,
            ).download_many(
                ["510300.SH", "159919.SZ", "512000.SH"],
                "2024-01-01",
                "2024-01-01",
                tmp_path,
            )

    assert (tmp_path / "510300.SH.parquet").is_file()
    assert not (tmp_path / "512000.SH.parquet").exists()
    assert requested_codes == ["510300", "159919"]


def test_main_downloads_with_default_adjustment_and_prints_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "510300.SH.parquet"
    downloader = MagicMock()
    downloader.download.return_value = output

    with patch.object(module, "PyTdxDownloader", return_value=downloader) as cls:
        result = module.main(
            [
                "510300.SH",
                "2020-05-01",
                "2026-01-01",
                "--host",
                "quote.example",
                "--dest-dir",
                str(tmp_path),
            ]
        )

    assert result == 0
    cls.assert_called_once_with(
        host="quote.example",
        port=7709,
        timeout=5.0,
        auto_adjust=True,
    )
    downloader.download.assert_called_once_with(
        "510300.SH",
        "2020-05-01",
        "2026-01-01",
        tmp_path,
    )
    assert capsys.readouterr().out.strip() == str(output)


def test_main_no_auto_adjust_flag_is_forwarded(tmp_path: Path) -> None:
    downloader = MagicMock()
    downloader.download.return_value = tmp_path / "510300.SH.parquet"

    with patch.object(module, "PyTdxDownloader", return_value=downloader) as cls:
        module.main(
            [
                "510300.SH",
                "2020-05-01",
                "2026-01-01",
                "--host",
                "quote.example",
                "--no-auto-adjust",
            ]
        )

    assert cls.call_args.kwargs["auto_adjust"] is False


def test_main_requires_explicit_host(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        module.main(["510300.SH", "2020-05-01", "2026-01-01"])

    assert exc_info.value.code == 2
    assert "--host" in capsys.readouterr().err
