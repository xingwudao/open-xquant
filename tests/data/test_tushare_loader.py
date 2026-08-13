from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from oxq.core.errors import DownloadError
from oxq.data.loaders import Downloader, TushareDownloader


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600519.SH", "600519.SH"],
            "trade_date": ["20240102", "20240103"],
            "open": [100.0, 110.0],
            "high": [105.0, 115.0],
            "low": [95.0, 105.0],
            "close": [102.0, 112.0],
            "vol": [10.25, 20.0],
        }
    )


def _factors_with_later_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["600519.SH", "600519.SH", "600519.SH"],
            "trade_date": ["20240102", "20240103", "20240104"],
            "adj_factor": [1.0, 1.5, 2.0],
        }
    )


def _module(client: MagicMock) -> SimpleNamespace:
    return SimpleNamespace(__version__="1.4.29", pro_api=MagicMock(return_value=client))


def _download_with_responses(
    tmp_path: Path, daily: object, factors: object
) -> Path:
    client = MagicMock()
    client.daily.return_value = daily
    client.adj_factor.return_value = factors
    tushare = _module(client)
    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        return TushareDownloader(token="secret").download(
            "600519.SH", "20240102", "20240104", tmp_path
        )


def test_tushare_downloader_satisfies_protocol() -> None:
    downloader: Downloader = TushareDownloader(token="secret")
    assert isinstance(downloader, Downloader)


def test_tushare_rejects_noncanonical_date_before_token_lookup() -> None:
    with pytest.raises(
        DownloadError,
        match=r"Invalid Tushare start date '2024-1-2'\. Use YYYY-MM-DD or YYYYMMDD\.",
    ):
        TushareDownloader().download("600519.SH", "2024-1-2", "2024-01-03")


def test_tushare_download_writes_qfq_share_volume_and_manifest(tmp_path: Path) -> None:
    client = MagicMock()
    client.daily.return_value = _daily().iloc[::-1].reset_index(drop=True)
    client.adj_factor.return_value = _factors_with_later_reference().iloc[::-1].reset_index(drop=True)
    tushare = _module(client)

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        path = TushareDownloader(token="secret").download(
            "600519.SH", "2024-01-02", "2024-01-04", dest_dir=tmp_path
        )

    result = pd.read_parquet(path)
    assert path == tmp_path / "600519.SH.parquet"
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert str(result.index.tz) == "Asia/Shanghai"
    assert result.index.is_monotonic_increasing
    assert result["open"].tolist() == [50.0, 82.5]
    assert result["close"].tolist() == [51.0, 84.0]
    assert result["volume"].tolist() == [1025, 2000]
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "tushare"
    assert manifest["extra"] == {
        "adjust": "qfq",
        "adjustment_reference_date": "20240104",
        "adjustment_reference_factor": 2.0,
        "source_volume_unit": "lot",
        "volume_unit": "share",
        "tushare_version": "1.4.29",
    }
    assert "secret" not in path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    client.daily.assert_called_once_with(
        ts_code="600519.SH", start_date="20240102", end_date="20240104"
    )
    client.adj_factor.assert_called_once_with(
        ts_code="600519.SH", start_date="20240102", end_date="20240104"
    )


def test_tushare_ignores_factors_after_requested_end(tmp_path: Path) -> None:
    client = MagicMock()
    client.daily.return_value = _daily()
    client.adj_factor.return_value = _factors_with_later_reference()
    tushare = _module(client)

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        path = TushareDownloader(token="secret").download(
            "600519.SH", "2024-01-02", "2024-01-03", dest_dir=tmp_path
        )

    result = pd.read_parquet(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert result["open"].tolist() == pytest.approx([100.0 / 1.5, 110.0])
    assert manifest["extra"]["adjustment_reference_date"] == "20240103"
    assert manifest["extra"]["adjustment_reference_factor"] == 1.5


def test_tushare_rejects_volume_outside_int64_range(tmp_path: Path) -> None:
    client = MagicMock()
    client.daily.return_value = _daily().assign(vol=[100_000_000_000_000_000.0] * 2)
    client.adj_factor.return_value = _factors_with_later_reference().iloc[:2]
    tushare = _module(client)

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        with pytest.raises(DownloadError, match="int64"):
            TushareDownloader(token="secret").download(
                "600519.SH", "2024-01-02", "2024-01-03", dest_dir=tmp_path
            )


def test_tushare_rejects_volume_at_int64_overflow_boundary(tmp_path: Path) -> None:
    client = MagicMock()
    client.daily.return_value = _daily().assign(vol=[float(2**63) / 100] * 2)
    client.adj_factor.return_value = _factors_with_later_reference().iloc[:2]
    tushare = _module(client)

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        with pytest.raises(DownloadError, match="int64"):
            TushareDownloader(token="secret").download(
                "600519.SH", "2024-01-02", "2024-01-03", dest_dir=tmp_path
            )


def test_explicit_token_precedes_environment_and_client_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "environment-secret")
    client = MagicMock()
    client.daily.return_value = _daily()
    client.adj_factor.return_value = _factors_with_later_reference()
    tushare = _module(client)
    downloader = TushareDownloader(token="explicit-secret")

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        downloader.download("600519.SH", "20240102", "20240104", tmp_path)
        downloader.download("600519.SH", "20240102", "20240104", tmp_path)

    tushare.pro_api.assert_called_once_with("explicit-secret")


def test_missing_token_fails_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    with patch("oxq.data.loaders.importlib.import_module") as import_module:
        with pytest.raises(DownloadError, match="TUSHARE_TOKEN"):
            TushareDownloader().download(
                "600519.SH", "20240102", "20240104", tmp_path
            )

    import_module.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_environment_token_is_read_once_when_client_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "first-secret")
    client = MagicMock()
    client.daily.return_value = _daily()
    client.adj_factor.return_value = _factors_with_later_reference()
    tushare = _module(client)
    downloader = TushareDownloader()

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        downloader.download("600519.SH", "20240102", "20240104", tmp_path)
        monkeypatch.setenv("TUSHARE_TOKEN", "replacement-secret")
        downloader.download("600519.SH", "20240102", "20240104", tmp_path)

    tushare.pro_api.assert_called_once_with("first-secret")


def test_upstream_error_removes_token_from_exception_graph(tmp_path: Path) -> None:
    token = "super" + "-secret"
    client = MagicMock()
    client.daily.side_effect = RuntimeError(f"request included {token}")
    tushare = _module(client)
    output_dir = tmp_path / "new-output"

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        with pytest.raises(DownloadError) as captured:
            TushareDownloader(token=token).download(
                "600519.SH", "20240102", "20240104", output_dir
            )

    rendered = "".join(traceback.format_exception(captured.value))
    assert token not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert not output_dir.exists()


def test_missing_tushare_dependency_has_install_hint_and_no_exception_chain(
    tmp_path: Path,
) -> None:
    missing = ModuleNotFoundError("No module named 'tushare'", name="tushare")

    with patch("oxq.data.loaders.importlib.import_module", side_effect=missing):
        with pytest.raises(DownloadError, match=r"uv sync --extra tushare") as captured:
            TushareDownloader(token="secret").download(
                "600519.SH", "20240102", "20240104", tmp_path
            )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert not list(tmp_path.iterdir())


def test_client_creation_error_removes_token_from_exception_graph(
    tmp_path: Path,
) -> None:
    token = "client" + "-secret"
    tushare = SimpleNamespace(
        __version__="1.4.29",
        pro_api=MagicMock(side_effect=RuntimeError(f"invalid token {token}")),
    )

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        with pytest.raises(DownloadError, match="client initialization") as captured:
            TushareDownloader(token=token).download(
                "600519.SH", "20240102", "20240104", tmp_path
            )

    rendered = "".join(traceback.format_exception(captured.value))
    assert token not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("symbol", ["600519", "600519.sh", "../600519.SH"])
def test_invalid_symbol_fails_before_import(
    tmp_path: Path, symbol: str
) -> None:
    with patch("oxq.data.loaders.importlib.import_module") as import_module:
        with pytest.raises(DownloadError, match="Invalid Tushare symbol"):
            TushareDownloader(token="secret").download(
                symbol, "20240102", "20240104", tmp_path
            )

    import_module.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("20240230", "20240301", "Invalid Tushare start date"),
        ("2024/01/02", "20240104", "Invalid Tushare start date"),
        ("20240104", "20240102", "start date must not be after end date"),
    ],
)
def test_invalid_date_range_fails_before_import(
    tmp_path: Path, start: str, end: str, message: str
) -> None:
    with patch("oxq.data.loaders.importlib.import_module") as import_module:
        with pytest.raises(DownloadError, match=message):
            TushareDownloader(token="secret").download(
                "600519.SH", start, end, tmp_path
            )

    import_module.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", ["daily", "factors"])
@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (None, "DataFrame"),
        (pd.DataFrame(), "No Tushare"),
        ({"trade_date": ["20240102"]}, "DataFrame"),
    ],
)
def test_invalid_tushare_response_is_rejected_without_writes(
    tmp_path: Path, source: str, response: object, kind: str
) -> None:
    daily: object = _daily()
    factors: object = _factors_with_later_reference()
    if source == "daily":
        daily = response
    else:
        factors = response

    with pytest.raises(DownloadError, match=kind):
        _download_with_responses(tmp_path, daily, factors)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("source", "column"),
    [
        ("daily", "ts_code"),
        ("daily", "trade_date"),
        ("daily", "open"),
        ("daily", "high"),
        ("daily", "low"),
        ("daily", "close"),
        ("daily", "vol"),
        ("factors", "ts_code"),
        ("factors", "trade_date"),
        ("factors", "adj_factor"),
    ],
)
def test_missing_required_response_column_is_rejected(
    tmp_path: Path, source: str, column: str
) -> None:
    daily = _daily()
    factors = _factors_with_later_reference()
    if source == "daily":
        daily = daily.drop(columns=column)
    else:
        factors = factors.drop(columns=column)

    with pytest.raises(DownloadError, match=column):
        _download_with_responses(tmp_path, daily, factors)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", ["daily", "factors"])
def test_response_for_another_symbol_is_rejected(
    tmp_path: Path, source: str
) -> None:
    daily = _daily()
    factors = _factors_with_later_reference()
    if source == "daily":
        daily.loc[0, "ts_code"] = "000001.SZ"
    else:
        factors.loc[0, "ts_code"] = "000001.SZ"

    with pytest.raises(DownloadError, match="ts_code"):
        _download_with_responses(tmp_path, daily, factors)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", ["daily", "factors"])
def test_nullable_response_symbol_is_rejected(
    tmp_path: Path, source: str
) -> None:
    daily = _daily()
    factors = _factors_with_later_reference()
    if source == "daily":
        daily["ts_code"] = daily["ts_code"].astype("string")
        daily.loc[0, "ts_code"] = pd.NA
    else:
        factors["ts_code"] = factors["ts_code"].astype("string")
        factors.loc[0, "ts_code"] = pd.NA

    with pytest.raises(DownloadError, match="ts_code"):
        _download_with_responses(tmp_path, daily, factors)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("trade_date", ["20240101", "20240105"])
def test_daily_date_outside_requested_range_is_rejected(
    tmp_path: Path, trade_date: str
) -> None:
    daily = _daily()
    daily.loc[0, "trade_date"] = trade_date

    with pytest.raises(DownloadError, match="requested date range"):
        _download_with_responses(tmp_path, daily, _factors_with_later_reference())

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("source", ["daily", "factors"])
def test_duplicate_response_dates_are_rejected(
    tmp_path: Path, source: str
) -> None:
    daily = _daily()
    factors = _factors_with_later_reference()
    if source == "daily":
        daily.loc[1, "trade_date"] = daily.loc[0, "trade_date"]
    else:
        factors.loc[1, "trade_date"] = factors.loc[0, "trade_date"]

    with pytest.raises(DownloadError, match="duplicate"):
        _download_with_responses(tmp_path, daily, factors)

    assert not list(tmp_path.iterdir())


def test_missing_factor_for_daily_date_is_rejected(tmp_path: Path) -> None:
    factors = _factors_with_later_reference().query("trade_date != '20240102'")

    with pytest.raises(DownloadError, match="missing adjustment factors"):
        _download_with_responses(tmp_path, _daily(), factors)

    assert not list(tmp_path.iterdir())


def test_factor_reference_selection_ignores_dataframe_index(tmp_path: Path) -> None:
    factors = _factors_with_later_reference()
    factors.index = [0, 1, 0]

    path = _download_with_responses(tmp_path, _daily(), factors)

    result = pd.read_parquet(path)
    assert result["open"].tolist() == [50.0, 82.5]


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.0, -1.0, "not-a-number"])
def test_invalid_price_is_rejected(tmp_path: Path, value: object) -> None:
    daily = _daily()
    daily["open"] = daily["open"].astype(object)
    daily.loc[0, "open"] = value

    with pytest.raises(DownloadError, match="prices must be positive and finite"):
        _download_with_responses(tmp_path, daily, _factors_with_later_reference())

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.0, -1.0, "not-a-number"])
def test_invalid_adjustment_factor_is_rejected(
    tmp_path: Path, value: object
) -> None:
    factors = _factors_with_later_reference()
    factors["adj_factor"] = factors["adj_factor"].astype(object)
    factors.loc[0, "adj_factor"] = value

    with pytest.raises(DownloadError, match="factors must be positive and finite"):
        _download_with_responses(tmp_path, _daily(), factors)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("column", "value"),
    [("high", 101.0), ("low", 101.0)],
)
def test_invalid_unadjusted_price_envelope_is_rejected(
    tmp_path: Path, column: str, value: float
) -> None:
    daily = _daily()
    daily.loc[0, column] = value

    with pytest.raises(DownloadError, match="unadjusted price envelope"):
        _download_with_responses(tmp_path, daily, _factors_with_later_reference())

    assert not list(tmp_path.iterdir())


def test_invalid_adjusted_price_envelope_is_rejected(tmp_path: Path) -> None:
    smallest = np.nextafter(0.0, 1.0)
    daily = _daily()
    daily.loc[0, ["open", "high", "low", "close"]] = smallest
    factors = _factors_with_later_reference()
    factors.loc[0, "adj_factor"] = smallest
    factors.loc[2, "adj_factor"] = np.finfo(np.float64).max

    with pytest.raises(DownloadError, match="adjusted price envelope"):
        _download_with_responses(tmp_path, daily, factors)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (-0.01, "non-negative"),
        (0.00000002, "whole shares"),
        (np.nan, "finite"),
        (np.inf, "finite"),
    ],
)
def test_invalid_volume_is_rejected(
    tmp_path: Path, value: float, message: str
) -> None:
    daily = _daily()
    daily.loc[0, "vol"] = value

    with pytest.raises(DownloadError, match=message):
        _download_with_responses(tmp_path, daily, _factors_with_later_reference())

    assert not list(tmp_path.iterdir())


def test_qfq_arithmetic_overflow_is_rejected(tmp_path: Path) -> None:
    factors = _factors_with_later_reference()
    factors.loc[0, "adj_factor"] = np.finfo(np.float64).max
    factors.loc[2, "adj_factor"] = np.nextafter(0.0, 1.0)

    with pytest.raises(DownloadError, match="overflow"):
        _download_with_responses(tmp_path, _daily(), factors)

    assert not list(tmp_path.iterdir())


def test_download_many_returns_paths_for_all_symbols(tmp_path: Path) -> None:
    client = MagicMock()
    client.daily.side_effect = lambda *, ts_code, **_: _daily().assign(ts_code=ts_code)
    client.adj_factor.side_effect = (
        lambda *, ts_code, **_: _factors_with_later_reference().assign(ts_code=ts_code)
    )
    tushare = _module(client)

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        paths = TushareDownloader(token="secret").download_many(
            ["600519.SH", "000001.SZ"],
            "20240102",
            "20240104",
            tmp_path,
        )

    assert paths == {
        "600519.SH": tmp_path / "600519.SH.parquet",
        "000001.SZ": tmp_path / "000001.SZ.parquet",
    }
    assert all(path.is_file() for path in paths.values())


def test_download_many_stops_at_first_failing_symbol(tmp_path: Path) -> None:
    client = MagicMock()
    client.daily.side_effect = lambda *, ts_code, **_: (
        _daily().assign(ts_code=ts_code)
        if ts_code == "600519.SH"
        else (_ for _ in ()).throw(RuntimeError("upstream unavailable"))
    )
    client.adj_factor.side_effect = (
        lambda *, ts_code, **_: _factors_with_later_reference().assign(ts_code=ts_code)
    )
    tushare = _module(client)

    with patch("oxq.data.loaders.importlib.import_module", return_value=tushare):
        with pytest.raises(DownloadError, match="000001.SZ"):
            TushareDownloader(token="secret").download_many(
                ["600519.SH", "000001.SZ", "300750.SZ"],
                "20240102",
                "20240104",
                tmp_path,
            )

    assert [call.kwargs["ts_code"] for call in client.daily.call_args_list] == [
        "600519.SH",
        "000001.SZ",
    ]
    assert (tmp_path / "600519.SH.parquet").is_file()
    assert not (tmp_path / "000001.SZ.parquet").exists()
    assert not (tmp_path / "300750.SZ.parquet").exists()
