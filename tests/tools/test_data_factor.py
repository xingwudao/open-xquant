"""Tests for factor-related tools in oxq.tools.data."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from oxq.tools.data import factor_download, factor_inspect, factor_list


def _make_wb_response(data: dict[tuple[str, int], float | None]) -> list:
    """Build a fake World Bank API response."""
    records = [
        {
            "indicator": {"id": "NY.GDP.MKTP.CD"},
            "country": {"id": c},
            "countryiso3code": c,
            "date": str(y),
            "value": v,
        }
        for (c, y), v in data.items()
    ]
    return [{"page": 1, "total": len(records)}, records]


def _mock_urlopen(response_data: list) -> MagicMock:
    body = json.dumps(response_data).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None
    return mock_resp


def _write_factor(tmp_path: Path, name: str = "gdp") -> Path:
    factor_dir = tmp_path / "factor"
    factor_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"CHN": [14.7e12], "USA": [21.3e12]},
        index=pd.Index([2023], name="year"),
    )
    df.to_parquet(factor_dir / f"{name}.parquet")
    return factor_dir


class TestFactorDownload:
    def test_success(self, tmp_path: Path) -> None:
        wb = _make_wb_response({("USA", 2023): 27.4e12, ("CHN", 2023): 17.9e12})
        with patch("oxq.data.factors.urllib.request.urlopen", return_value=_mock_urlopen(wb)):
            result = factor_download("gdp", ["USA", "CHN"], 2023, 2023, data_dir=str(tmp_path))

        assert result["indicator"] == "gdp"
        assert sorted(result["countries"]) == ["CHN", "USA"]
        assert result["rows"] == 1

    def test_unknown_indicator(self, tmp_path: Path) -> None:
        result = factor_download("fake", ["USA"], data_dir=str(tmp_path))
        assert "error" in result
        assert "Unknown indicator" in result["error"]


class TestFactorList:
    def test_empty_dir(self, tmp_path: Path) -> None:
        result = factor_list(data_dir=str(tmp_path))
        assert result["factors"] == []
        assert result["count"] == 0

    def test_with_files(self, tmp_path: Path) -> None:
        factor_dir = _write_factor(tmp_path, "gdp")
        _write_factor(tmp_path, "cpi")  # writes to same factor_dir
        result = factor_list(data_dir=str(factor_dir))
        assert sorted(result["factors"]) == ["cpi", "gdp"]
        assert result["count"] == 2


class TestFactorInspect:
    def test_existing_factor(self, tmp_path: Path) -> None:
        factor_dir = _write_factor(tmp_path, "gdp")
        result = factor_inspect("gdp", data_dir=str(factor_dir))
        assert result["indicator"] == "gdp"
        assert sorted(result["countries"]) == ["CHN", "USA"]
        assert result["rows"] == 1

    def test_missing_factor(self, tmp_path: Path) -> None:
        result = factor_inspect("gdp", data_dir=str(tmp_path))
        assert "error" in result
        assert "available_indicators" in result
