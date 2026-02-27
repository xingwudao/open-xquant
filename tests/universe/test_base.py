import pandas as pd

from oxq.universe.base import Filter, UniverseProvider, UniverseSnapshot


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


def test_universe_provider_is_runtime_checkable() -> None:
    class FakeUniverse:
        def resolve(self, mktdata: dict[str, pd.DataFrame] | None = None) -> UniverseSnapshot:
            return UniverseSnapshot(symbols=(), source="fake", metadata={})

    provider = FakeUniverse()
    assert isinstance(provider, UniverseProvider)
