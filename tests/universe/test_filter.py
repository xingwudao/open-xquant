import pytest

from oxq.universe.base import Filter, UniverseProvider
from oxq.universe.filter import FilterUniverse


def test_filter_satisfies_protocol(sample_mktdata) -> None:
    universe: UniverseProvider = FilterUniverse(
        base=("AAPL",),
        filters=(Filter(column="close", op=">=", value=1.0),),
        mktdata=sample_mktdata,
    )
    assert isinstance(universe, UniverseProvider)


def test_filter_get_universe_passes_all(sample_mktdata) -> None:
    universe = FilterUniverse(
        base=("AAPL", "PENNY", "GOOG"),
        filters=(Filter(column="close", op=">=", value=0.1),),
        mktdata=sample_mktdata,
    )
    snapshot = universe.get_universe("2024-01-04")
    assert snapshot.as_of_date == "2024-01-04"
    assert snapshot.symbols == ("AAPL", "PENNY", "GOOG")


def test_filter_get_universe_filters_out(sample_mktdata) -> None:
    universe = FilterUniverse(
        base=("AAPL", "PENNY", "GOOG"),
        filters=(
            Filter(column="close", op=">=", value=1.0),
            Filter(column="volume", op=">=", value=1_000_000),
        ),
        mktdata=sample_mktdata,
    )
    snapshot = universe.get_universe("2024-01-04")
    # PENNY: close=0.3 < 1.0 → filtered out
    assert snapshot.symbols == ("AAPL", "GOOG")
    assert snapshot.metadata["base_count"] == 3
    assert snapshot.metadata["filtered_count"] == 2


def test_filter_get_universe_skips_missing_symbol(sample_mktdata) -> None:
    universe = FilterUniverse(
        base=("AAPL", "MISSING"),
        filters=(Filter(column="close", op=">=", value=1.0),),
        mktdata=sample_mktdata,
    )
    snapshot = universe.get_universe("2024-01-04")
    assert snapshot.symbols == ("AAPL",)


def test_filter_get_universe_requires_mktdata() -> None:
    universe = FilterUniverse(
        base=("AAPL",), filters=(Filter(column="close", op=">=", value=1.0),)
    )
    with pytest.raises(ValueError, match="requires mktdata to get_universe"):
        universe.get_universe("2024-01-04")


def test_filter_accepts_list_input() -> None:
    universe = FilterUniverse(
        base=["AAPL", "GOOG"],
        filters=[Filter(column="close", op=">=", value=1.0)],
    )
    assert isinstance(universe.base, tuple)
    assert isinstance(universe.filters, tuple)


def test_filter_source_string() -> None:
    import pandas as pd

    universe = FilterUniverse(
        base=("AAPL",),
        filters=(
            Filter(column="volume", op=">=", value=1000000),
            Filter(column="close", op=">=", value=1.0),
        ),
        mktdata={"AAPL": pd.DataFrame(
            {"close": [150.0], "volume": [2_000_000]},
            index=pd.DatetimeIndex(["2024-01-04"]),
        )},
        name="liquidity",
    )
    snapshot = universe.get_universe("2024-01-04")
    assert "filter:liquidity:" in snapshot.source
    assert "volume>=1000000" in snapshot.source


def test_filter_get_history(sample_mktdata) -> None:
    universe = FilterUniverse(
        base=("AAPL", "GOOG"),
        filters=(Filter(column="close", op=">=", value=1.0),),
        mktdata=sample_mktdata,
    )
    history = universe.get_history("2024-01-02", "2024-01-04")
    assert len(history) == 2
    assert history[0].as_of_date == "2024-01-02"
    assert history[1].as_of_date == "2024-01-04"
