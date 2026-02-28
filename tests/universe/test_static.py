from oxq.universe.base import UniverseProvider
from oxq.universe.static import StaticUniverse


def test_static_satisfies_protocol() -> None:
    universe: UniverseProvider = StaticUniverse(symbols=("AAPL",))
    assert isinstance(universe, UniverseProvider)


def test_static_get_universe_returns_snapshot() -> None:
    universe = StaticUniverse(symbols=("AAPL", "GOOG", "MSFT"), name="test")
    snapshot = universe.get_universe("2024-01-04")
    assert snapshot.as_of_date == "2024-01-04"
    assert snapshot.symbols == ("AAPL", "GOOG", "MSFT")
    assert snapshot.source == "static:test"
    assert snapshot.metadata == {}


def test_static_get_universe_no_name() -> None:
    universe = StaticUniverse(symbols=("AAPL",))
    snapshot = universe.get_universe("2024-01-04")
    assert snapshot.source == "static"


def test_static_accepts_list_input() -> None:
    universe = StaticUniverse(symbols=["AAPL", "GOOG"])
    assert universe.symbols == ("AAPL", "GOOG")
    assert isinstance(universe.symbols, tuple)


def test_static_get_history() -> None:
    universe = StaticUniverse(symbols=("AAPL",))
    history = universe.get_history("2024-01-01", "2024-01-31")
    assert len(history) == 1
    assert history[0].as_of_date == "2024-01-01"
    assert history[0].symbols == ("AAPL",)
