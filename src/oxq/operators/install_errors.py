"""Structured, sanitized failures for operator installation workflows."""

from __future__ import annotations

import re
from urllib.parse import SplitResult, urlsplit, urlunsplit


class OperatorInstallError(ValueError):
    """A stable, safe-to-render operator installation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        provider: str | None = None,
        release: str | None = None,
        operator_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.provider = provider
        self.release = release
        self.operator_id = operator_id

    def as_dict(self) -> dict[str, str]:
        """Return the public error envelope without URL secrets."""
        result = {
            "status": "fail",
            "stage": _sanitize(self.stage),
            "code": _sanitize(self.code),
            "message": _sanitize(self.message),
        }
        for name, value in (
            ("provider", self.provider),
            ("release", self.release),
            ("operator_id", self.operator_id),
        ):
            if value is not None:
                result[name] = _sanitize(value)
        return result


def install_error(
    code: str,
    message: str,
    *,
    stage: str,
    provider: str | None = None,
    release: str | None = None,
    operator_id: str | None = None,
) -> OperatorInstallError:
    """Construct an installation error with its stable public metadata."""
    return OperatorInstallError(
        code,
        message,
        stage=stage,
        provider=provider,
        release=release,
        operator_id=operator_id,
    )


def _sanitize(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        return _safe_url(parsed)
    return re.sub(r"https?://[^\s]+", _sanitize_url_match, value)


def _sanitize_url_match(match: re.Match[str]) -> str:
    return _safe_url(urlsplit(match.group()))


def _safe_url(parsed: SplitResult) -> str:
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""
    return urlunsplit((parsed.scheme, f"{parsed.hostname or ''}{port}", parsed.path, "", ""))
