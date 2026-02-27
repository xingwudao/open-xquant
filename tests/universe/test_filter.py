import pytest

from oxq.universe.base import Filter, UniverseProvider
from oxq.universe.filter import FilterUniverse


def test_filter_satisfies_protocol() -> None:
    universe: UniverseProvider = FilterUniverse(
        base=("AAPL",), filters=(Filter(column="close", op=">=", value=1.0),)
    )
    assert isinstance(universe, UniverseProvider)


def test_filter_resolve_passes_all(sample_mktdata) -> None:
    universe = FilterUniverse(
        base=("AAPL", "PENNY", "GOOG"),
        filters=(Filter(column="close", op=">=", value=0.1),),
    )
    snapshot = universe.resolve(sample_mktdata)
    assert snapshot.symbols == ("AAPL", "PENNY", "GOOG")


def test_filter_resolve_filters_out(sample_mktdata) -> None:
    universe = FilterUniverse(
        base=("AAPL", "PENNY", "GOOG"),
        filters=(
            Filter(column="close", op=">=", value=1.0),
            Filter(column="volume", op=">=", value=1_000_000),
        ),
    )
    snapshot = universe.resolve(sample_mktdata)
    # PENNY: close=0.3 < 1.0 → filtered out
    assert snapshot.symbols == ("AAPL", "GOOG")
    assert snapshot.metadata["base_count"] == 3
    assert snapshot.metadata["filtered_count"] == 2


def test_filter_resolve_skips_missing_symbol(sample_mktdata) -> None:
    universe = FilterUniverse(
        base=("AAPL", "MISSING"),
        filters=(Filter(column="close", op=">=", value=1.0),),
    )
    snapshot = universe.resolve(sample_mktdata)
    assert snapshot.symbols == ("AAPL",)


def test_filter_resolve_requires_mktdata() -> None:
    universe = FilterUniverse(
        base=("AAPL",), filters=(Filter(column="close", op=">=", value=1.0),)
    )
    with pytest.raises(ValueError, match="requires mktdata"):
        universe.resolve()


def test_filter_accepts_list_input() -> None:
    universe = FilterUniverse(
        base=["AAPL", "GOOG"],
        filters=[Filter(column="close", op=">=", value=1.0)],
    )
    assert isinstance(universe.base, tuple)
    assert isinstance(universe.filters, tuple)


def test_filter_source_string() -> None:
    universe = FilterUniverse(
        base=("AAPL",),
        filters=(
            Filter(column="volume", op=">=", value=1000000),
            Filter(column="close", op=">=", value=1.0),
        ),
        name="liquidity",
    )
    snapshot = universe.resolve({"AAPL": __import__("pandas").DataFrame(
        {"close": [150.0], "volume": [2_000_000]},
    )})
    assert "filter:liquidity:" in snapshot.source
    assert "volume>=1000000" in snapshot.source
