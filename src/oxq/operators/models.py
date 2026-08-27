"""Immutable models for a verified local provider submission."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType

_TARGET_PART = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class CertificationTarget:
    """One immutable Python wheel target for a v2 certification."""

    python_tag: str
    abi_tag: str
    platform_tag: str

    @classmethod
    def parse(cls, value: str) -> CertificationTarget:
        parts = value.split("-", 2)
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError("certification target must be python-abi-platform")
        target = cls(*parts)
        if target.key != value or not all(_TARGET_PART.fullmatch(part) for part in parts):
            raise ValueError("certification target is not canonical")
        return target

    @property
    def key(self) -> str:
        return f"{self.python_tag}-{self.abi_tag}-{self.platform_tag}"


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
    build_identifier: str
    digest: str
    wheel_path: Path


@dataclass(frozen=True)
class BaselineCase:
    """One baseline case retained as authoritative decoded input."""

    case_id: str
    operator_id: str
    operator_version: str
    parameters: Mapping[str, object]
    input: Mapping[str, object]
    expected: Mapping[str, object]
    tolerance: Mapping[str, object]
    baseline_path: Path | None = None
    baseline_relative_path: str | None = None
    case_index: int | None = None


@dataclass(frozen=True)
class ContractCandidate:
    """One schema-valid, provenance-bound operator candidate."""

    manifest: Mapping[str, object]
    binding: Mapping[str, object]
    manifest_path: Path
    implementation_artifact: Path


@dataclass(frozen=True)
class ContractCertification:
    """A provider release whose operators satisfy the frozen contract."""

    provider: str
    release: str
    submission_commit: str
    source_commit: str
    source_root: Path
    operators: tuple[ContractCandidate, ...]
    artifacts: tuple[BuildArtifact, ...]
    baseline_cases: tuple[BaselineCase, ...]


@dataclass(frozen=True)
class BaselineResult:
    """One numerical baseline case that passed exact-wheel execution."""

    operator_id: str
    operator_version: str
    case_id: str
    status: str


@dataclass(frozen=True)
class ResearchCertification:
    """A provider release whose exact wheels passed every numerical baseline."""

    provider: str
    release: str
    submission_commit: str
    source_commit: str
    source_root: Path
    operators: tuple[ContractCandidate, ...]
    artifacts: tuple[BuildArtifact, ...]
    baseline_cases: tuple[BaselineCase, ...]
    baseline_results: tuple[BaselineResult, ...]


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
