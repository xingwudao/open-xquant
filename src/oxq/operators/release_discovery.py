"""Bounded discovery of official GitHub operator releases."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from oxq.operators.install_errors import OperatorInstallError, install_error
from oxq.operators.release_index import (
    OfficialProvider,
    OperatorReleaseIndex,
    ReleaseAsset,
    parse_release_index,
)

_ALLOWED_HOSTS = frozenset(
    {
        "api.github.com",
        "github.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
        "files.pythonhosted.org",
    }
)
_MAX_REDIRECTS = 5
_MAX_INDEX_BYTES = 1024 * 1024
_CHUNK_BYTES = 64 * 1024


class _Opener(Protocol):
    def open(self, request: Request, timeout: float | None = None) -> object: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: object, code: int, msg: str, headers: object, newurl: str) -> None:
        return None


@dataclass(frozen=True)
class ResolvedOfficialRelease:
    """A validated release index rooted in the frozen official-provider map."""

    index: OperatorReleaseIndex
    trust_state: str = "github-source-trusted"

    @property
    def release_index(self) -> OperatorReleaseIndex:
        """Compatibility-friendly explicit name for the resolved index."""
        return self.index


class OfficialReleaseResolver:
    """Resolve one exact official provider release without importing provider code."""

    def __init__(self, *, opener: _Opener | None = None) -> None:
        self._opener = opener

    def resolve(self, provider: OfficialProvider, release: str) -> ResolvedOfficialRelease:
        """Fetch the exact tag's index asset and bind it to provider identity."""
        api_url = f"https://api.github.com/repos/{provider.repository}/releases/tags/v{release}"
        try:
            release_payload = _read_json(_fetch_bytes(api_url, _MAX_INDEX_BYTES, self._opener))
            asset_url = _release_asset_url(release_payload, provider.release_asset, release)
            index = parse_release_index(_fetch_bytes(asset_url, _MAX_INDEX_BYTES, self._opener))
        except OperatorInstallError:
            raise
        except (OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
            raise _release_error(provider, release, "official release discovery failed") from None
        if index.provider != provider.name or index.release != release:
            raise _release_error(provider, release, "official release index identity is invalid")
        return ResolvedOfficialRelease(index=index)


def download_verified_asset(
    asset: ReleaseAsset,
    destination: str | Path,
    *,
    opener: _Opener | None = None,
) -> Path:
    """Stream one trusted-index asset to an atomic destination after verification."""
    destination_path = Path(destination)
    temporary_path: Path | None = None
    response: object | None = None
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        response = _open_following_redirects(asset.url, opener)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{asset.filename}.", suffix=".part", dir=destination_path.parent
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        total = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    chunk = response.read(_CHUNK_BYTES)  # type: ignore[attr-defined]
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > asset.size_bytes:
                        raise ValueError("download exceeds declared size")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        finally:
            _close(response)
            response = None
        if total != asset.size_bytes or digest.hexdigest() != _digest_value(asset.digest):
            raise ValueError("download does not match declared digest or size")
        os.replace(temporary_path, destination_path)
        temporary_path = None
        return destination_path
    except (OSError, TypeError, ValueError, URLError, HTTPError):
        raise install_error(
            "operator_download_failed",
            f"verified asset download failed: {asset.url}",
            stage="download",
        ) from None
    finally:
        if response is not None:
            _close(response)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _fetch_bytes(url: str, limit: int, opener: _Opener | None) -> bytes:
    response = _open_following_redirects(url, opener)
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(_CHUNK_BYTES)  # type: ignore[attr-defined]
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > limit:
                raise ValueError("response exceeds fixed limit")
            chunks.append(chunk)
    finally:
        _close(response)


def _open_following_redirects(url: str, opener: _Opener | None) -> object:
    current = url
    for redirects in range(_MAX_REDIRECTS + 1):
        _validate_official_url(current)
        response = _open(current, opener)
        status = getattr(response, "status", getattr(response, "code", 200))
        if status not in {301, 302, 303, 307, 308}:
            if status != 200:
                _close(response)
                raise ValueError("unexpected HTTP response")
            return response
        location = getattr(response, "headers", {}).get("Location")
        _close(response)
        if not isinstance(location, str) or not location:
            raise ValueError("redirect lacks location")
        current = urljoin(current, location)
        if redirects == _MAX_REDIRECTS:
            raise ValueError("redirect limit exceeded")
    raise AssertionError("redirect loop escaped bounds")


def _open(url: str, opener: _Opener | None) -> object:
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "open-xquant"})
    try:
        if opener is not None:
            return opener.open(request, timeout=30)
        return build_opener(_NoRedirect()).open(request, timeout=30)
    except HTTPError as error:
        if error.code in {301, 302, 303, 307, 308}:
            return error
        raise


def _validate_official_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError("official URL is not trusted HTTPS")


def _read_json(raw_bytes: bytes) -> dict[str, object]:
    value = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("release response is not an object")
    return value


def _release_asset_url(payload: dict[str, object], filename: str, release: str) -> str:
    if payload.get("tag_name") != f"v{release}":
        raise ValueError("release tag differs from requested tag")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("release assets are invalid")
    matches = [
        asset.get("browser_download_url")
        for asset in assets
        if isinstance(asset, dict) and asset.get("name") == filename
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError("release index asset is missing or ambiguous")
    return matches[0]


def _digest_value(value: str) -> str:
    prefix, separator, digest = value.partition(":")
    if prefix != "sha256" or not separator or len(digest) != 64:
        raise ValueError("asset digest is invalid")
    return digest


def _close(response: object) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _release_error(provider: OfficialProvider, release: str, message: str) -> OperatorInstallError:
    return install_error(
        "operator_release_invalid",
        message,
        stage="release",
        provider=provider.name,
        release=release,
    )
