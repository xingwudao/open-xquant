"""Immutable models for a verified local provider submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType


@dataclass(frozen=True)
class CatalogEntry:
    """One operator keyed by its catalog identity and version."""

    operator_id: str
    operator_version: str
    manifest_path: Path
    baseline_path: Path


@dataclass(frozen=True)
class BuildArtifact:
    """A build record artifact whose on-disk bytes were verified."""

    distribution: str
    version: str
    filename: str
    role: str
    digest: str
    wheel_path: Path


@dataclass(frozen=True)
class BaselineCase:
    """One baseline case retained as authoritative decoded input."""

    operator_id: str
    operator_version: str
    parameters: dict[str, object]
    input: dict[str, object]
    expected: dict[str, object]
    tolerance: dict[str, object]


@dataclass(frozen=True)
class ProviderSubmission:
    """A verified archive, retaining its temporary directory until exit."""

    provider: str
    release: str
    submission_commit: str
    source_commit: str
    archive_root: Path
    source_root: Path
    operators: tuple[CatalogEntry, ...]
    artifacts: tuple[BuildArtifact, ...]
    baseline_cases: tuple[BaselineCase, ...]
    _archive: TemporaryDirectory[str] = field(repr=False, compare=False)

    def __enter__(self) -> ProviderSubmission:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self._archive.cleanup()
