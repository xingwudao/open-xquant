"""Tests for oxq.tools.universe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from oxq.tools.universe import (
    universe_history,
    universe_inspect,
    universe_list_indexes,
    universe_set,
)


@pytest.fixture()
def sample_data_dir(tmp_path: Path) -> Path:
    dates = pd.date_range("2024-01-02", periods=5, freq="B", name="date")
    for sym, close_base, vol_base in [
        ("AAPL", 150.0, 2_000_000),
        ("PENNY", 0.3, 300_000),
        ("GOOG", 140.0, 1_500_000),
    ]:
        df = pd.DataFrame(
            {"close": [close_base + i for i in range(5)], "volume": [vol_base + i * 100_000 for i in range(5)]},
            index=dates,
        )
        df.to_parquet(tmp_path / f"{sym}.parquet")
    return tmp_path


def test_universe_set_static() -> None:
    result = universe_set(type="static", symbols=["AAPL", "GOOG"], as_of_date="2024-01-04")
    assert result["symbols"] == ["AAPL", "GOOG"]
    assert result["count"] == 2
    assert result["as_of_date"] == "2024-01-04"


def test_universe_set_static_requires_symbols() -> None:
    result = universe_set(type="static")
    assert "error" in result


def test_universe_set_filter(sample_data_dir: Path) -> None:
    # PENNY close starts at 0.3 and increments by 1 per day, so at 2024-01-04
    # (3rd business day, index=2) close=2.3; use volume filter instead:
    # PENNY volume starts at 300_000, so volume >= 1_000_000 filters it out
    result = universe_set(
        type="filter",
        symbols=["AAPL", "PENNY", "GOOG"],
        filters=[{"column": "volume", "op": ">=", "value": 1_000_000}],
        as_of_date="2024-01-04",
        data_dir=str(sample_data_dir),
    )
    assert "PENNY" not in result["symbols"]
    assert "AAPL" in result["symbols"]
    assert result["as_of_date"] == "2024-01-04"


def test_universe_set_filter_requires_symbols() -> None:
    result = universe_set(type="filter", filters=[{"column": "close", "op": ">=", "value": 1.0}])
    assert "error" in result


def test_universe_set_filter_requires_filters() -> None:
    result = universe_set(type="filter", symbols=["AAPL"])
    assert "error" in result


def test_universe_set_index_phase2() -> None:
    result = universe_set(type="index", code="SP500")
    assert "error" in result
    assert "Phase 2" in result["error"]


def test_universe_set_unknown_type() -> None:
    result = universe_set(type="unknown")
    assert "error" in result


def test_universe_list_indexes() -> None:
    result = universe_list_indexes()
    assert result["indexes"] == []
    assert "Phase 2" in result["note"]


def test_universe_inspect(sample_data_dir: Path) -> None:
    result = universe_inspect(symbols=["AAPL", "MISSING"], data_dir=str(sample_data_dir))
    assert result["total"] == 2
    assert result["available"] == 1
    assert result["missing"] == 1


def test_universe_history() -> None:
    result = universe_history(symbols=["AAPL", "GOOG"], start="2024-01-01", end="2024-01-31")
    assert len(result["snapshots"]) == 1
    assert result["snapshots"][0]["as_of_date"] == "2024-01-01"
    assert result["snapshots"][0]["symbols"] == ["AAPL", "GOOG"]
