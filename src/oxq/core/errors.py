from __future__ import annotations


class OxqError(Exception):
    """Framework base exception."""


class SymbolNotFoundError(OxqError):
    """Local data file not found for symbol."""


class DownloadError(OxqError):
    """Data download failed."""
