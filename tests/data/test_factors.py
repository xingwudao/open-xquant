"""Tests for oxq.data.factors — WorldBankDownloader and read_factor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from oxq.data.factors import (
    INDICATOR_MAP,
    WorldBankDownloader,
    _records_to_dataframe,
    read_factor,
    resolve_factor_dir,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_wb_response(
    indicator_code: str,
    data: dict[tuple[str, int], float | None],
) -> list:
    """Build a fake World Bank API JSON response.

    Parameters
    ----------
    indicator_code : str
        World Bank indicator code.
    data : dict[(country, year), value]
        Mapping of (country_iso3, year) → value.
    """
    records = [
        {
            "indicator": {"id": indicator_code},
            "country": {"id": country},
            "countryiso3code": country,
            "date": str(year),
            "value": value,
        }
        for (country, year), value in data.items()
    ]
    metadata = {"page": 1, "pages": 1, "per_page": 10000, "total": len(records)}
    return [metadata, records]


def _write_sample_factor(tmp_path: Path, indicator: str = "gdp") -> Path:
    """Write a small sample factor parquet for read tests."""
    df = pd.DataFrame(
        {"CHN": [14.7e12, 17.7e12], "USA": [21.3e12, 23.3e12]},
        index=pd.Index([2020, 2021], name="year"),
    )
    factor_dir = tmp_path / "factor"
    factor_dir.mkdir(parents=True, exist_ok=True)
    path = factor_dir / f"{indicator}.parquet"
    df.to_parquet(path)
    return factor_dir


# ---------------------------------------------------------------------------
# resolve_factor_dir
# ---------------------------------------------------------------------------

class TestResolveFactorDir:
    def test_explicit_dir(self, tmp_path: Path) -> None:
        assert resolve_factor_dir(tmp_path) == tmp_path

    def test_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path))
        assert resolve_factor_dir() == tmp_path / "factor"

    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OXQ_DATA_DIR", raising=False)
        result = resolve_factor_dir()
        assert result == Path.home() / ".oxq" / "data" / "factor"


# ---------------------------------------------------------------------------
# _records_to_dataframe
# ---------------------------------------------------------------------------

class TestRecordsToDataframe:
    def test_basic_conversion(self) -> None:
        records = [
            {"date": "2020", "countryiso3code": "USA", "value": 21.3e12},
            {"date": "2020", "countryiso3code": "CHN", "value": 14.7e12},
            {"date": "2021", "countryiso3code": "USA", "value": 23.3e12},
            {"date": "2021", "countryiso3code": "CHN", "value": 17.7e12},
        ]
        df = _records_to_dataframe(records)
        assert df.index.name == "year"
        assert list(df.index) == [2020, 2021]
        assert sorted(df.columns) == ["CHN", "USA"]
        assert df.loc[2020, "USA"] == pytest.approx(21.3e12)

    def test_null_value_becomes_nan(self) -> None:
        records = [
            {"date": "2020", "countryiso3code": "USA", "value": None},
        ]
        df = _records_to_dataframe(records)
        assert pd.isna(df.loc[2020, "USA"])


# ---------------------------------------------------------------------------
# WorldBankDownloader
# ---------------------------------------------------------------------------

class TestWorldBankDownloader:
    def _mock_urlopen(self, response_data: list):
        """Create a mock for urllib.request.urlopen."""
        from unittest.mock import MagicMock

        body = json.dumps(response_data).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = lambda s, *a: None
        return mock_resp

    def test_download_gdp(self, tmp_path: Path) -> None:
        wb_response = _make_wb_response("NY.GDP.MKTP.CD", {
            ("USA", 2020): 21.3e12,
            ("USA", 2021): 23.3e12,
            ("CHN", 2020): 14.7e12,
            ("CHN", 2021): 17.7e12,
        })
        mock_resp = self._mock_urlopen(wb_response)

        with patch("oxq.data.factors.urllib.request.urlopen", return_value=mock_resp):
            dl = WorldBankDownloader()
            path = dl.download("gdp", ["USA", "CHN"], 2020, 2021, dest_dir=tmp_path)

        assert path == tmp_path / "gdp.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert list(df.index) == [2020, 2021]
        assert sorted(df.columns) == ["CHN", "USA"]

    def test_download_all_indicators(self, tmp_path: Path) -> None:
        """Verify all 4 indicators can be downloaded (with mocked API)."""
        for indicator, code in INDICATOR_MAP.items():
            wb_response = _make_wb_response(code, {
                ("USA", 2023): 100.0,
                ("CHN", 2023): 200.0,
            })
            mock_resp = self._mock_urlopen(wb_response)

            with patch("oxq.data.factors.urllib.request.urlopen", return_value=mock_resp):
                dl = WorldBankDownloader()
                path = dl.download(indicator, ["USA", "CHN"], 2023, 2023, dest_dir=tmp_path)

            assert path.exists()
            assert path.name == f"{indicator}.parquet"

    def test_unknown_indicator_raises_value_error(self, tmp_path: Path) -> None:
        dl = WorldBankDownloader()
        with pytest.raises(ValueError, match="Unknown indicator 'fake'"):
            dl.download("fake", ["USA"], dest_dir=tmp_path)

    def test_empty_response_raises_download_error(self, tmp_path: Path) -> None:
        from oxq.core.errors import DownloadError

        wb_response = [{"page": 1, "total": 0}, None]
        mock_resp = self._mock_urlopen(wb_response)

        with patch("oxq.data.factors.urllib.request.urlopen", return_value=mock_resp):
            dl = WorldBankDownloader()
            with pytest.raises(DownloadError, match="No data returned"):
                dl.download("gdp", ["USA"], dest_dir=tmp_path)

    def test_network_error_raises_download_error(self, tmp_path: Path) -> None:
        from oxq.core.errors import DownloadError

        with patch(
            "oxq.data.factors.urllib.request.urlopen",
            side_effect=ConnectionError("timeout"),
        ):
            dl = WorldBankDownloader()
            with pytest.raises(DownloadError, match="Failed to download"):
                dl.download("gdp", ["USA"], dest_dir=tmp_path)


# ---------------------------------------------------------------------------
# read_factor
# ---------------------------------------------------------------------------

class TestReadFactor:
    def test_read_all(self, tmp_path: Path) -> None:
        factor_dir = _write_sample_factor(tmp_path)
        df = read_factor("gdp", data_dir=factor_dir)
        assert list(df.index) == [2020, 2021]
        assert sorted(df.columns) == ["CHN", "USA"]

    def test_filter_countries(self, tmp_path: Path) -> None:
        factor_dir = _write_sample_factor(tmp_path)
        df = read_factor("gdp", countries=["USA"], data_dir=factor_dir)
        assert list(df.columns) == ["USA"]

    def test_filter_missing_country_ignored(self, tmp_path: Path) -> None:
        factor_dir = _write_sample_factor(tmp_path)
        df = read_factor("gdp", countries=["USA", "JPN"], data_dir=factor_dir)
        assert list(df.columns) == ["USA"]

    def test_filter_year_range(self, tmp_path: Path) -> None:
        factor_dir = _write_sample_factor(tmp_path)
        df = read_factor("gdp", start_year=2021, data_dir=factor_dir)
        assert list(df.index) == [2021]

    def test_filter_end_year(self, tmp_path: Path) -> None:
        factor_dir = _write_sample_factor(tmp_path)
        df = read_factor("gdp", end_year=2020, data_dir=factor_dir)
        assert list(df.index) == [2020]

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Factor file not found"):
            read_factor("gdp", data_dir=tmp_path)
