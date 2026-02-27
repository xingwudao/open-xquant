def test_data_public_api() -> None:
    from oxq.data import (
        MarketDataProvider,
        LocalMarketDataProvider,
        Downloader,
        YFinanceDownloader,
        AkShareDownloader,
        resolve_data_dir,
    )
    assert MarketDataProvider is not None
    assert LocalMarketDataProvider is not None
    assert Downloader is not None
    assert YFinanceDownloader is not None
    assert AkShareDownloader is not None
    assert resolve_data_dir is not None


def test_core_errors_public_api() -> None:
    from oxq.core import OxqError, SymbolNotFoundError, DownloadError
    assert OxqError is not None
    assert SymbolNotFoundError is not None
    assert DownloadError is not None
