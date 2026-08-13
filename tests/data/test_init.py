def test_data_public_api() -> None:
    from oxq.data import (
        AkShareDownloader,
        Downloader,
        LocalMarketDataProvider,
        MarketDataProvider,
        TushareDownloader,
        YFinanceDownloader,
        resolve_data_dir,
    )
    assert MarketDataProvider is not None
    assert LocalMarketDataProvider is not None
    assert Downloader is not None
    assert YFinanceDownloader is not None
    assert AkShareDownloader is not None
    assert TushareDownloader is not None
    assert resolve_data_dir is not None


def test_new_exports_available() -> None:
    from oxq.data import (
        EastMoneyFetcher,
        FactorDownloader,
        FactorFetcher,
        MarketDataAdapter,
        WorldBankFetcher,
        YFinanceFinancialFetcher,
    )
    assert EastMoneyFetcher is not None
    assert FactorDownloader is not None
    assert FactorFetcher is not None
    assert WorldBankFetcher is not None
    assert YFinanceFinancialFetcher is not None
    assert MarketDataAdapter is not None


def test_backward_compat_exports() -> None:
    from oxq.data import WorldBankDownloader
    from oxq.data.factors import WorldBankFetcher
    assert WorldBankDownloader is WorldBankFetcher


def test_core_errors_public_api() -> None:
    from oxq.core import DownloadError, OxqError, SymbolNotFoundError
    assert OxqError is not None
    assert SymbolNotFoundError is not None
    assert DownloadError is not None
