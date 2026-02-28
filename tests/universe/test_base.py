from oxq.universe.base import Filter, UniverseProvider, UniverseSnapshot


def test_filter_is_frozen() -> None:
    f = Filter(column="volume", op=">=", value=1_000_000)
    assert f.column == "volume"
    assert f.op == ">="
    assert f.value == 1_000_000


def test_universe_snapshot_is_frozen() -> None:
    snap = UniverseSnapshot(
        as_of_date="2024-01-04",
        symbols=("AAPL", "GOOG"),
        source="static",
        metadata={},
    )
    assert snap.as_of_date == "2024-01-04"
    assert snap.symbols == ("AAPL", "GOOG")
    assert snap.source == "static"
    assert snap.metadata == {}


def test_universe_provider_is_runtime_checkable() -> None:
    class FakeUniverse:
        def get_universe(self, as_of_date: str) -> UniverseSnapshot:
            return UniverseSnapshot(
                as_of_date=as_of_date, symbols=(), source="fake", metadata={}
            )

        def get_history(self, start: str, end: str) -> list[UniverseSnapshot]:
            return [self.get_universe(start)]

    provider = FakeUniverse()
    assert isinstance(provider, UniverseProvider)
