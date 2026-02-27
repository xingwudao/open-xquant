from oxq.universe.base import UniverseProvider
from oxq.universe.static import StaticUniverse


def test_static_satisfies_protocol() -> None:
    universe: UniverseProvider = StaticUniverse(symbols=("AAPL",))
    assert isinstance(universe, UniverseProvider)


def test_static_resolve_returns_snapshot() -> None:
    universe = StaticUniverse(symbols=("AAPL", "GOOG", "MSFT"), name="test")
    snapshot = universe.resolve()
    assert snapshot.symbols == ("AAPL", "GOOG", "MSFT")
    assert snapshot.source == "static:test"
    assert snapshot.metadata == {}


def test_static_resolve_no_name() -> None:
    universe = StaticUniverse(symbols=("AAPL",))
    snapshot = universe.resolve()
    assert snapshot.source == "static"


def test_static_accepts_list_input() -> None:
    universe = StaticUniverse(symbols=["AAPL", "GOOG"])
    assert universe.symbols == ("AAPL", "GOOG")
    assert isinstance(universe.symbols, tuple)


def test_static_resolve_ignores_mktdata() -> None:
    universe = StaticUniverse(symbols=("AAPL",))
    snapshot = universe.resolve(mktdata={"AAPL": "not_used"})
    assert snapshot.symbols == ("AAPL",)
