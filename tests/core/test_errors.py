from oxq.core.errors import DownloadError, OxqError, SymbolNotFoundError


def test_oxq_error_is_exception():
    assert issubclass(OxqError, Exception)


def test_symbol_not_found_error_is_oxq_error():
    err = SymbolNotFoundError("AAPL")
    assert isinstance(err, OxqError)
    assert "AAPL" in str(err)


def test_download_error_is_oxq_error():
    err = DownloadError("timeout")
    assert isinstance(err, OxqError)
    assert "timeout" in str(err)
