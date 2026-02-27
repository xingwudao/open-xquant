import pandas as pd

from oxq.data.providers import MarketDataProvider


class FakeProvider:
    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_latest(self, symbol: str) -> pd.Series:
        return pd.Series()


def test_fake_provider_satisfies_protocol() -> None:
    provider: MarketDataProvider = FakeProvider()
    assert isinstance(provider, MarketDataProvider)
