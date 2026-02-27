from oxq.universe.base import Filter, UniverseSnapshot


def test_filter_is_frozen() -> None:
    f = Filter(column="volume", op=">=", value=1_000_000)
    assert f.column == "volume"
    assert f.op == ">="
    assert f.value == 1_000_000


def test_universe_snapshot_is_frozen() -> None:
    snap = UniverseSnapshot(
        symbols=("AAPL", "GOOG"),
        source="static",
        metadata={},
    )
    assert snap.symbols == ("AAPL", "GOOG")
    assert snap.source == "static"
    assert snap.metadata == {}
