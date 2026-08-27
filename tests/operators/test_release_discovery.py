"""Official GitHub release resolution and verified-download tests."""

from __future__ import annotations

import hashlib
import json
from http.client import IncompleteRead
from pathlib import Path

import pytest

from oxq.operators.install_errors import OperatorInstallError
from oxq.operators.release_discovery import OfficialReleaseResolver, download_verified_asset
from oxq.operators.release_index import OfficialProvider, ReleaseAsset


class _Response:
    def __init__(self, body: bytes, url: str, *, status: int = 200, location: str | None = None) -> None:
        self._body = body
        self._url = url
        self.status = status
        self.headers = {} if location is None else {"Location": location}

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self._body)
        result, self._body = self._body[:amount], self._body[amount:]
        return result

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        return None


class _Opener:
    def __init__(self, responses: dict[str, _Response]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def open(self, request: object, timeout: float | None = None) -> _Response:
        url = request.full_url  # type: ignore[attr-defined]
        self.urls.append(url)
        return self.responses[url]


def _provider() -> OfficialProvider:
    return OfficialProvider("equant-py", "xingwudao/equant-py", "operator-release-v1.json")


def _index_bytes() -> bytes:
    asset = {
        "digest": "sha256:" + "a" * 64,
        "filename": "bundle.zip",
        "size_bytes": 1,
        "url": "https://github.com/xingwudao/equant-py/releases/download/v1.0.0/bundle.zip",
    }
    value = {
        "certification_state": "research-certified",
        "operator_count": 1,
        "provider": "equant-py",
        "release": "1.0.0",
        "release_type": "open-xquant-operator-release",
        "schema_version": 1,
        "source_commit": "git-sha1:" + "b" * 40,
        "submission_commit": "git-sha1:" + "a" * 40,
        "targets": [
            {
                "abi_tag": "cp312",
                "bundle": asset,
                "platform_tag": "macosx_14_0_arm64",
                "python_tag": "cp312",
                "wheels": [
                    {
                        **asset,
                        "distribution": "equant-core",
                        "filename": "equant_core-1.0.0-py3-none-any.whl",
                        "role": "implementation",
                        "tags": ["py3-none-any"],
                        "version": "1.0.0",
                    }
                ],
            }
        ],
    }
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _release_response(url: str) -> bytes:
    return json.dumps({"tag_name": "v1.0.0", "assets": [{"name": "operator-release-v1.json", "browser_download_url": url}]}).encode()


def test_resolves_exact_github_tag_and_returns_source_trust() -> None:
    api = "https://api.github.com/repos/xingwudao/equant-py/releases/tags/v1.0.0"
    index_url = "https://github.com/xingwudao/equant-py/releases/download/v1.0.0/operator-release-v1.json"
    opener = _Opener({api: _Response(_release_response(index_url), api), index_url: _Response(_index_bytes(), index_url)})

    resolved = OfficialReleaseResolver(opener=opener).resolve(_provider(), "1.0.0")

    assert resolved.index.provider == "equant-py"
    assert resolved.index.release == "1.0.0"
    assert resolved.trust_state == "github-source-trusted"


def test_rejects_release_response_without_named_asset() -> None:
    api = "https://api.github.com/repos/xingwudao/equant-py/releases/tags/v1.0.0"
    response = _release_response("https://github.com/other.json").replace(
        b"operator-release-v1.json", b"other.json"
    )
    opener = _Opener({api: _Response(response, api)})

    with pytest.raises(OperatorInstallError) as caught:
        OfficialReleaseResolver(opener=opener).resolve(_provider(), "1.0.0")
    assert caught.value.code == "operator_release_invalid"


@pytest.mark.parametrize(
    ("first_url", "redirect"),
    [
        ("http://github.com/file", "https://github.com/file"),
        ("https://github.com/file", "https://evil.example/file"),
    ],
)
def test_download_rejects_untrusted_initial_or_redirect_url(tmp_path: Path, first_url: str, redirect: str) -> None:
    asset = ReleaseAsset("asset.bin", first_url, 1, "sha256:" + hashlib.sha256(b"x").hexdigest())
    opener = _Opener({first_url: _Response(b"", first_url, status=302, location=redirect)})

    with pytest.raises(OperatorInstallError) as caught:
        download_verified_asset(asset, tmp_path / "asset.bin", opener=opener)
    assert caught.value.code == "operator_download_failed"


def test_download_rejects_excessive_redirects(tmp_path: Path) -> None:
    urls = [f"https://github.com/{number}" for number in range(9)]
    asset = ReleaseAsset("asset.bin", urls[0], 1, "sha256:" + hashlib.sha256(b"x").hexdigest())
    opener = _Opener({url: _Response(b"", url, status=302, location=urls[number + 1]) for number, url in enumerate(urls[:-1])})

    with pytest.raises(OperatorInstallError) as caught:
        download_verified_asset(asset, tmp_path / "asset.bin", opener=opener)
    assert caught.value.code == "operator_download_failed"


@pytest.mark.parametrize(
    ("body", "size", "digest"),
    [
        (b"ab", 3, hashlib.sha256(b"ab").hexdigest()),
        (b"ab", 2, hashlib.sha256(b"wrong").hexdigest()),
    ],
)
def test_download_rejects_size_and_digest_mismatches(tmp_path: Path, body: bytes, size: int, digest: str) -> None:
    url = "https://github.com/file"
    asset = ReleaseAsset("asset.bin", url, size, "sha256:" + digest)

    with pytest.raises(OperatorInstallError) as caught:
        download_verified_asset(asset, tmp_path / "asset.bin", opener=_Opener({url: _Response(body, url)}))
    assert caught.value.code == "operator_download_failed"
    assert not (tmp_path / "asset.bin").exists()


def test_download_removes_partial_file_on_interrupted_stream(tmp_path: Path) -> None:
    url = "https://github.com/file"
    asset = ReleaseAsset("asset.bin", url, 2, "sha256:" + hashlib.sha256(b"ab").hexdigest())
    response = _Response(b"ab", url)
    response.read = lambda amount=-1: (_ for _ in ()).throw(OSError("interrupted"))  # type: ignore[method-assign]

    with pytest.raises(OperatorInstallError) as caught:
        download_verified_asset(asset, tmp_path / "asset.bin", opener=_Opener({url: response}))
    assert caught.value.code == "operator_download_failed"
    assert not (tmp_path / "asset.bin").exists()


def test_resolver_converts_interrupted_index_read_to_install_error() -> None:
    api = "https://api.github.com/repos/xingwudao/equant-py/releases/tags/v1.0.0"
    index_url = "https://github.com/xingwudao/equant-py/releases/download/v1.0.0/operator-release-v1.json"
    interrupted = _Response(b"", index_url)
    interrupted.read = lambda amount=-1: (_ for _ in ()).throw(IncompleteRead(b"partial", 1))  # type: ignore[method-assign]
    opener = _Opener(
        {
            api: _Response(_release_response(index_url), api),
            index_url: interrupted,
        }
    )

    with pytest.raises(OperatorInstallError) as caught:
        OfficialReleaseResolver(opener=opener).resolve(_provider(), "1.0.0")
    assert caught.value.code == "operator_release_invalid"


def test_download_converts_incomplete_read_to_install_error(tmp_path: Path) -> None:
    url = "https://github.com/file"
    asset = ReleaseAsset("asset.bin", url, 2, "sha256:" + hashlib.sha256(b"ab").hexdigest())
    response = _Response(b"", url)
    response.read = lambda amount=-1: (_ for _ in ()).throw(IncompleteRead(b"a", 1))  # type: ignore[method-assign]

    with pytest.raises(OperatorInstallError) as caught:
        download_verified_asset(asset, tmp_path / "asset.bin", opener=_Opener({url: response}))
    assert caught.value.code == "operator_download_failed"
    assert not (tmp_path / "asset.bin").exists()


def test_download_error_redacts_credentials_and_query(tmp_path: Path) -> None:
    url = "https://user:secret@github.com/file?token=secret"
    asset = ReleaseAsset("asset.bin", url, 1, "sha256:" + hashlib.sha256(b"x").hexdigest())

    with pytest.raises(OperatorInstallError) as caught:
        download_verified_asset(
            asset,
            tmp_path / "asset.bin",
            opener=_Opener({url: _Response(b"", url, status=500)}),
        )

    serialized = caught.value.as_dict()
    assert "secret" not in repr(serialized)
    assert "?" not in serialized["message"]
