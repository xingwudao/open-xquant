"""Validate loaded providers against the frozen operator contract."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Protocol, cast

from jsonschema import (  # type: ignore[import-untyped]
    Draft202012Validator,
    FormatChecker,
    SchemaError,
    ValidationError,
)

from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import (
    BuildArtifact,
    CatalogEntry,
    ContractCandidate,
    ContractCertification,
    ProviderSubmission,
)
from oxq.operators.resources import materialize_contract_surface

_SCHEMA_ID = (
    "https://open-xquant.dev/contracts/quant-operators/"
    "operator-manifest-v1.schema.json"
)
_CONTRACT_RELEASE = "1.0.0"
_FROZEN_SURFACE_DIGESTS = {
    "quant_panel_schema": "sha256:fd6fcd7f3102cdd63913644f87a154a22713c0286a6e9e1cc16e84ca6b283a9c",
    "operator_manifest_schema": "sha256:adea87a6caec3984d65d9fbaaa0ba132be76e5609ed17407de5e8b85c38bf82e",
    "operator_binding_schema": "sha256:1d0e3ed12acde2a2d0c1fe2309f9a090ea7b0f8193bc0f3f6fd659c178047de6",
    "reference_validator": "sha256:48099f887ebfc9fd9857ba8cececaa8b52c1dd5a2020ccc5eca21c3120664d9a",
}
_SURFACE_FILENAMES = {
    "quant_panel_schema": "quant-panel-v1.schema.json",
    "operator_manifest_schema": "operator-manifest-v1.schema.json",
    "operator_binding_schema": "operator-binding-v1.schema.json",
    "reference_validator": "reference_validator_v1.py",
}


class _ReferenceValidator(Protocol):
    def validate_operator_manifest(self, manifest: Mapping[str, object]) -> None: ...

    def validate_operator_binding(
        self,
        binding: Mapping[str, object],
        manifest: Mapping[str, object],
        manifest_path: Path,
        source_root: Path,
        implementation_artifact_path: Path,
        contract_surface_paths: Mapping[str, Path],
    ) -> None: ...


def validate_provider_contract(
    submission: ProviderSubmission,
) -> ContractCertification:
    """Validate all provider manifests and construct contract-valid bindings."""
    candidates: list[ContractCandidate] = []
    with _snapshot_contract_surface() as (surface_bytes, surface_paths):
        manifest_schema = _load_schema(
            surface_bytes["operator_manifest_schema"],
            "manifest_schema_invalid",
            "manifest",
        )
        binding_schema = _load_schema(
            surface_bytes["operator_binding_schema"],
            "binding_validation_failed",
            "binding",
        )
        validator = _load_reference_validator(
            surface_bytes["reference_validator"],
            surface_paths["reference_validator"],
        )
        for entry in submission.operators:
            manifest, manifest_bytes = _read_manifest(entry)
            _validate_schema(
                manifest,
                manifest_schema,
                code="manifest_schema_invalid",
                message="operator manifest does not match frozen schema",
                stage="manifest",
                operator_id=entry.operator_id,
            )
            _validate_manifest_semantics(validator, manifest, entry.operator_id)
            implementation, artifact = _validate_manifest_identity(
                submission, entry, manifest
            )
            binding = _construct_binding(manifest, implementation, manifest_bytes)
            _validate_schema(
                binding,
                binding_schema,
                code="binding_validation_failed",
                message="operator binding validation failed",
                stage="binding",
                operator_id=entry.operator_id,
            )
            _validate_binding_semantics(
                validator,
                binding,
                manifest,
                entry,
                submission.source_root,
                artifact.wheel_path,
                surface_paths,
            )
            candidates.append(
                ContractCandidate(
                    manifest=manifest,
                    binding=binding,
                    manifest_path=entry.manifest_path,
                    implementation_artifact=artifact.wheel_path,
                )
            )

    return ContractCertification(
        provider=submission.provider,
        release=submission.release,
        submission_commit=submission.submission_commit,
        source_commit=submission.source_commit,
        source_root=submission.source_root,
        operators=tuple(candidates),
        artifacts=submission.artifacts,
        baseline_cases=submission.baseline_cases,
    )


def _read_manifest(entry: CatalogEntry) -> tuple[dict[str, object], bytes]:
    try:
        manifest_bytes = entry.manifest_path.read_bytes()
        manifest_text = manifest_bytes.decode("utf-8")
        value = json.loads(
            manifest_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise _error(
            "manifest_schema_invalid",
            "operator manifest is not strict UTF-8 JSON",
            "manifest",
            entry.operator_id,
        ) from None
    return cast(dict[str, object], value), manifest_bytes


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    del value
    raise ValueError("non-standard manifest number")


@contextmanager
def _snapshot_contract_surface() -> Iterator[
    tuple[dict[str, bytes], dict[str, Path]]
]:
    try:
        with materialize_contract_surface() as materialized_paths:
            if set(materialized_paths) != set(_FROZEN_SURFACE_DIGESTS):
                raise _surface_validation_error()
            surface_bytes = {
                name: materialized_paths[name].read_bytes()
                for name in _FROZEN_SURFACE_DIGESTS
            }
            actual_digests = {
                name: _sha256_bytes(value)
                for name, value in surface_bytes.items()
            }
            if actual_digests != _FROZEN_SURFACE_DIGESTS:
                raise _surface_validation_error()

            with TemporaryDirectory(prefix="oxq-contract-snapshot-") as directory:
                snapshot_root = Path(directory)
                snapshot_paths: dict[str, Path] = {}
                for name, filename in _SURFACE_FILENAMES.items():
                    snapshot_path = snapshot_root / filename
                    snapshot_path.write_bytes(surface_bytes[name])
                    snapshot_path.chmod(stat.S_IRUSR)
                    snapshot_paths[name] = snapshot_path
                snapshot_root.chmod(stat.S_IRUSR | stat.S_IXUSR)
                yield surface_bytes, snapshot_paths
    except OperatorCertificationError:
        raise
    except OSError:
        raise _error(
            "binding_validation_failed",
            "frozen contract resources are unavailable",
            "binding",
        ) from None


def _surface_validation_error() -> OperatorCertificationError:
    return _error(
        "binding_validation_failed",
        "frozen contract surface validation failed",
        "binding",
    )


def _load_schema(
    schema_bytes: bytes,
    code: str,
    stage: str,
) -> dict[str, object]:
    try:
        value = json.loads(schema_bytes.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("schema is not an object")
        Draft202012Validator.check_schema(value)
    except (
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        SchemaError,
    ):
        message = (
            "operator manifest schema is unavailable"
            if stage == "manifest"
            else "operator binding schema is unavailable"
        )
        raise _error(code, message, stage) from None
    return cast(dict[str, object], value)


def _validate_schema(
    instance: object,
    schema: Mapping[str, object],
    *,
    code: str,
    message: str,
    stage: str,
    operator_id: str,
) -> None:
    try:
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).validate(instance)
    except (SchemaError, ValidationError):
        raise _error(code, message, stage, operator_id) from None


def _load_reference_validator(
    source_bytes: bytes,
    snapshot_path: Path,
) -> _ReferenceValidator:
    try:
        source = source_bytes.decode("utf-8")
        code = compile(
            source,
            str(snapshot_path),
            "exec",
            dont_inherit=True,
        )
        module = ModuleType("_oxq_packaged_operator_reference_validator_v1")
        module.__file__ = str(snapshot_path)
        exec(code, module.__dict__)
        _require_validator_functions(module)
    except (UnicodeError, ValueError, SyntaxError, TypeError, ImportError):
        raise _error(
            "binding_validation_failed",
            "frozen reference validator is unavailable",
            "binding",
        ) from None
    return cast(_ReferenceValidator, module)


def _require_validator_functions(module: ModuleType) -> None:
    if not callable(getattr(module, "validate_operator_manifest", None)):
        raise ImportError("manifest validator unavailable")
    if not callable(getattr(module, "validate_operator_binding", None)):
        raise ImportError("binding validator unavailable")


def _validate_manifest_semantics(
    validator: _ReferenceValidator,
    manifest: Mapping[str, object],
    operator_id: str,
) -> None:
    try:
        validator.validate_operator_manifest(manifest)
    except Exception:
        raise _error(
            "manifest_semantics_invalid",
            "operator manifest violates frozen semantics",
            "manifest",
            operator_id,
        ) from None


def _validate_manifest_identity(
    submission: ProviderSubmission,
    entry: CatalogEntry,
    manifest: Mapping[str, object],
) -> tuple[Mapping[str, object], BuildArtifact]:
    implementation = cast(Mapping[str, object], manifest["implementation"])
    matching_artifacts = [
        artifact
        for artifact in submission.artifacts
        if artifact.role == "implementation"
        and artifact.distribution == manifest["distribution"]
        and artifact.version == implementation["package_version"]
    ]
    if (
        manifest["operator_id"] != entry.operator_id
        or manifest["operator_version"] != entry.operator_version
        or implementation["source_commit"] != submission.source_commit
        or len(matching_artifacts) != 1
    ):
        raise _manifest_identity_error(entry.operator_id)
    artifact = matching_artifacts[0]
    if (
        implementation["build_identifier"] != artifact.build_identifier
        or implementation["implementation_digest"] != artifact.digest
    ):
        raise _manifest_identity_error(entry.operator_id)
    return implementation, artifact


def _manifest_identity_error(operator_id: str) -> OperatorCertificationError:
    return _error(
        "manifest_identity_mismatch",
        "operator manifest identity does not match submission",
        "manifest",
        operator_id,
    )


def _construct_binding(
    manifest: Mapping[str, object],
    implementation: Mapping[str, object],
    manifest_bytes: bytes,
) -> dict[str, object]:
    return {
        "binding_version": 1,
        "operator_id": manifest["operator_id"],
        "operator_version": manifest["operator_version"],
        "distribution": manifest["distribution"],
        "distribution_version": implementation["package_version"],
        "source_commit": implementation["source_commit"],
        "source_tree_digest": implementation["source_tree_digest"],
        "schema_id": _SCHEMA_ID,
        "schema_release": _CONTRACT_RELEASE,
        "schema_digest": _FROZEN_SURFACE_DIGESTS["operator_manifest_schema"],
        "manifest_digest": _sha256_bytes(manifest_bytes),
        "implementation_digest": implementation["implementation_digest"],
        "surface_release": _CONTRACT_RELEASE,
        "contract_surface": {
            name: {"release": _CONTRACT_RELEASE, "digest": digest}
            for name, digest in _FROZEN_SURFACE_DIGESTS.items()
        },
        "certification_state": "contract-valid",
    }


def _validate_binding_semantics(
    validator: _ReferenceValidator,
    binding: Mapping[str, object],
    manifest: Mapping[str, object],
    entry: CatalogEntry,
    source_root: Path,
    implementation_artifact: Path,
    surface_paths: Mapping[str, Path],
) -> None:
    try:
        validator.validate_operator_binding(
            binding,
            manifest,
            entry.manifest_path,
            source_root,
            implementation_artifact,
            surface_paths,
        )
    except Exception:
        raise _error(
            "binding_validation_failed",
            "operator binding validation failed",
            "binding",
            entry.operator_id,
        ) from None


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _error(
    code: str,
    message: str,
    stage: str,
    operator_id: str | None = None,
) -> OperatorCertificationError:
    return OperatorCertificationError(
        code,
        message,
        stage=stage,
        operator_id=operator_id,
    )
