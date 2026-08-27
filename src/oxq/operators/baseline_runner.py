"""Execute numerical baselines in an isolated exact-wheel child process."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import BinaryIO, cast

import numpy as np

from oxq.operators.certification import (
    _load_reference_validator,
    _load_schema,
    _snapshot_contract_surface,
    _validate_schema,
)
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import BaselineCase, BaselineResult, BuildArtifact, ContractCertification
from oxq.operators.runtime_protocol import run_exact_wheel_request

_CHILD_FAILURE_CODES = {
    "provider_import_failed",
    "provider_execution_failed",
    "provider_mutated_input",
    "provider_alignment_failed",
    "baseline_mismatch",
}
_OUTPUT_DTYPES = {"boolean", "int64", "float64", "string", "date", "datetime"}
_AUXILIARY_REQUIREMENTS = {
    "requires_benchmark",
    "requires_industry_data",
    "requires_market_cap_data",
    "requires_fundamental_data",
}
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_ARTIFACT_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_CHILD_RESPONSE_BYTES = 1024 * 1024
_PROVIDER_POLICY_VIOLATION_EXIT_CODE = 86


@dataclass(frozen=True)
class _VerifiedArtifact:
    artifact: BuildArtifact
    source_path: Path
    snapshot: BinaryIO


def run_research_baselines(
    candidate: ContractCertification,
    runtime_artifacts: Sequence[BuildArtifact],
    timeout_seconds: float = 30,
) -> tuple[BaselineResult, ...]:
    """Run every declared baseline against only the verified provider wheels."""
    artifacts = tuple(runtime_artifacts)
    operators = {
        (
            cast(str, item.manifest["operator_id"]),
            cast(str, item.manifest["operator_version"]),
        ): item
        for item in candidate.operators
    }
    if not operators or len(operators) != len(candidate.operators):
        raise _error("baseline_input_invalid", "baseline operator identity is invalid")
    timeout_seconds = _validated_timeout(
        timeout_seconds,
        next(iter(operators))[0],
    )

    cases_by_operator = {identity: 0 for identity in operators}
    results: list[BaselineResult] = []
    with (
        ExitStack() as artifact_snapshots,
        _snapshot_contract_surface() as (
            surface_bytes,
            surface_paths,
        ),
    ):
        verified_artifacts = _verified_artifacts(
            candidate,
            artifacts,
            artifact_snapshots,
        )
        quant_panel_schema = _load_schema(
            surface_bytes["quant_panel_schema"],
            "baseline_input_invalid",
            "baseline",
        )
        validator = _load_reference_validator(
            surface_bytes["reference_validator"],
            surface_paths["reference_validator"],
        )
        validate_quant_panel = cast(
            Callable[[Mapping[str, object]], None],
            getattr(validator, "validate_quant_panel", None),
        )
        validate_parameters = cast(
            Callable[[Mapping[str, object], Mapping[str, object]], None],
            getattr(validator, "validate_operator_request_parameters", None),
        )
        if not callable(validate_quant_panel) or not callable(validate_parameters):
            raise _error(
                "baseline_input_invalid",
                "frozen baseline validators are unavailable",
            )

        for case in candidate.baseline_cases:
            identity = (case.operator_id, case.operator_version)
            operator = operators.get(identity)
            if operator is None:
                raise _error(
                    "baseline_input_invalid",
                    "baseline does not identify a contract-valid operator",
                    case.operator_id,
                )
            cases_by_operator[identity] += 1
            _validate_case_input(
                case,
                operator.manifest,
                quant_panel_schema,
                validate_quant_panel,
            )
            _validate_case_parameters(
                case,
                operator.manifest,
                validate_parameters,
            )
            resolved_parameters = _resolved_parameters(case, operator.manifest)
            output_fields = _output_fields(
                case,
                operator.manifest,
                resolved_parameters,
            )
            _validate_case_tolerance(case)
            implementation_artifact = _implementation_artifact(
                operator.implementation_artifact,
                verified_artifacts,
                case.operator_id,
            )
            dependency_artifacts = [verified for verified in verified_artifacts if verified.artifact.role == "runtime-dependency"]
            actual, repeated = _run_child(
                manifest=operator.manifest,
                case=case,
                parameters=resolved_parameters,
                output_fields=output_fields,
                implementation_artifact=implementation_artifact,
                dependency_artifacts=dependency_artifacts,
                timeout_seconds=timeout_seconds,
            )
            for output_field, output_dtype in output_fields:
                _assert_expected(
                    case,
                    output_field,
                    actual[output_field],
                    output_dtype,
                )
            for output_field, output_dtype in output_fields:
                _assert_deterministic(
                    case,
                    operator.manifest,
                    output_dtype,
                    actual[output_field],
                    repeated[output_field],
                )
            results.append(
                BaselineResult(
                    operator_id=case.operator_id,
                    operator_version=case.operator_version,
                    case_id=case.case_id,
                    status="passed",
                )
            )

    missing = [identity for identity, count in cases_by_operator.items() if count == 0]
    if missing:
        raise _error(
            "baseline_input_invalid",
            "every contract-valid operator requires a numerical baseline",
            missing[0][0],
        )
    return tuple(results)


def _verified_artifacts(
    candidate: ContractCertification,
    artifacts: tuple[BuildArtifact, ...],
    snapshots: ExitStack,
) -> list[_VerifiedArtifact]:
    if artifacts != candidate.artifacts or not artifacts:
        raise _error(
            "provider_import_failed",
            "runtime artifacts do not match the verified contract artifacts",
        )
    verified: list[_VerifiedArtifact] = []
    try:
        for artifact in artifacts:
            path = artifact.wheel_path.resolve(strict=True)
            if not path.is_file():
                raise OSError("runtime artifact is not a regular file")
            snapshot = cast(
                BinaryIO,
                snapshots.enter_context(TemporaryFile(mode="w+b")),
            )
            os.set_inheritable(snapshot.fileno(), False)
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(_ARTIFACT_COPY_CHUNK_BYTES):
                    snapshot.write(chunk)
                    digest.update(chunk)
            snapshot.flush()
            snapshot.seek(0)
            if f"sha256:{digest.hexdigest()}" != artifact.digest:
                raise OSError("runtime artifact digest mismatch")
            verified.append(
                _VerifiedArtifact(
                    artifact=artifact,
                    source_path=path,
                    snapshot=snapshot,
                )
            )
    except OSError:
        raise _error(
            "provider_import_failed",
            "verified runtime artifact is unavailable",
        ) from None
    implementation_paths = {item.source_path for item in verified if item.artifact.role == "implementation"}
    candidate_paths = {operator.implementation_artifact.resolve() for operator in candidate.operators}
    if not candidate_paths.issubset(implementation_paths):
        raise _error(
            "provider_import_failed",
            "operator implementation artifact is not verified",
        )
    return verified


def _implementation_artifact(
    implementation_artifact: Path,
    verified_artifacts: list[_VerifiedArtifact],
    operator_id: str,
) -> _VerifiedArtifact:
    resolved = implementation_artifact.resolve()
    matches = [item for item in verified_artifacts if item.artifact.role == "implementation" and item.source_path == resolved]
    if len(matches) != 1:
        raise _error(
            "provider_import_failed",
            "operator implementation artifact is not uniquely verified",
            operator_id,
        )
    return matches[0]


def _validate_case_input(
    case: BaselineCase,
    manifest: Mapping[str, object],
    schema: Mapping[str, object],
    validate_quant_panel: Callable[[Mapping[str, object]], None],
) -> None:
    try:
        _validate_schema(
            case.input,
            schema,
            code="baseline_input_invalid",
            message="baseline input does not match the frozen QuantPanel schema",
            stage="baseline",
            operator_id=case.operator_id,
        )
        validate_quant_panel(case.input)
        _validate_manifest_input(case.input, manifest)
    except OperatorCertificationError:
        raise
    except Exception:
        raise _error(
            "baseline_input_invalid",
            "baseline input violates frozen QuantPanel semantics",
            case.operator_id,
        ) from None


def _validate_manifest_input(
    panel: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    input_contract = cast(Mapping[str, object], manifest["input"])
    columns = cast(list[Mapping[str, object]], panel["columns"])
    declared_dtypes = {cast(str, column["name"]): cast(str, column["dtype"]) for column in columns}
    required_panel_columns = {cast(str, column["name"]) for column in columns if cast(bool, column["required"])}
    required_columns = cast(list[str], input_contract["required_columns"])
    optional_columns = cast(list[str], input_contract["optional_columns"])
    supported_dtypes = set(cast(list[str], input_contract["supported_dtypes"]))
    if any(name not in required_panel_columns or declared_dtypes[name] not in supported_dtypes for name in required_columns):
        raise ValueError("baseline does not supply required input columns and dtypes")
    if not set(declared_dtypes).issubset({*required_columns, *optional_columns}):
        raise ValueError("baseline supplies columns not declared by the manifest")
    if any(dtype not in supported_dtypes for dtype in declared_dtypes.values()):
        raise ValueError("baseline supplies a column with an unsupported dtype")
    if any(cast(bool, input_contract[name]) for name in _AUXILIARY_REQUIREMENTS):
        raise ValueError("baseline runner does not support auxiliary input requirements")

    context = cast(Mapping[str, object], panel["context"])
    required_context = cast(list[str], input_contract["required_context"])
    if any(name not in context for name in required_context):
        raise ValueError("baseline does not supply required input context")

    records = cast(list[Mapping[str, object]], panel["records"])
    assets = {record["code"] for record in records}
    if len(assets) < cast(int, input_contract["minimum_assets"]):
        raise ValueError("baseline does not meet minimum asset count")
    time_by_asset: dict[object, set[object]] = {}
    for record in records:
        time_by_asset.setdefault(record["code"], set()).add(record["date"])
    minimum_time_length = cast(int, input_contract["minimum_time_length"])
    if not time_by_asset or min(map(len, time_by_asset.values())) < minimum_time_length:
        raise ValueError("baseline does not meet minimum time length")

    if cast(bool, input_contract["requires_complete_cross_section"]):
        assets_by_time: dict[object, set[object]] = {}
        for record in records:
            assets_by_time.setdefault(record["date"], set()).add(record["code"])
        if any(codes != assets for codes in assets_by_time.values()):
            raise ValueError("baseline cross section is incomplete")

    if cast(bool, input_contract["requires_sorted_input"]):
        sort_order = cast(list[str], input_contract["required_sort_order"])
        observed = [tuple(record[name] for name in sort_order) for record in records]
        if observed != sorted(observed):
            raise ValueError("baseline does not meet required sort order")


def _validate_case_parameters(
    case: BaselineCase,
    manifest: Mapping[str, object],
    validate_parameters: Callable[
        [Mapping[str, object], Mapping[str, object]],
        None,
    ],
) -> None:
    try:
        validate_parameters(manifest, case.parameters)
    except Exception:
        raise _error(
            "baseline_parameters_invalid",
            "baseline parameters violate the frozen request contract",
            case.operator_id,
        ) from None


def _validate_case_tolerance(case: BaselineCase) -> None:
    try:
        values = (case.tolerance["absolute"], case.tolerance["relative"])
        if any(
            type(value) not in {int, float} or not math.isfinite(float(cast(int | float, value))) or cast(int | float, value) < 0
            for value in values
        ):
            raise ValueError("baseline tolerance must be finite and nonnegative")
    except (KeyError, OverflowError, TypeError, ValueError):
        raise _error(
            "baseline_input_invalid",
            "baseline tolerance must be finite and nonnegative",
            case.operator_id,
        ) from None


def _resolved_parameters(
    case: BaselineCase,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    definitions = cast(Mapping[str, Mapping[str, object]], manifest["parameters"])
    return {name: case.parameters.get(name, definition["default"]) for name, definition in definitions.items()}


def _output_fields(
    case: BaselineCase,
    manifest: Mapping[str, object],
    resolved_parameters: Mapping[str, object],
) -> list[tuple[str, str]]:
    try:
        output = cast(Mapping[str, object], manifest["output"])
        fields = cast(list[Mapping[str, object]], output["fields"])
        declared_fields = [
            (
                cast(str, field["name_template"]).format(**resolved_parameters),
                cast(str, field["dtype"]),
            )
            for field in fields
        ]
    except (IndexError, KeyError, TypeError, ValueError):
        raise _error(
            "baseline_input_invalid",
            "manifest output field cannot be resolved for the baseline",
            case.operator_id,
        ) from None
    declared_names = [name for name, _ in declared_fields]
    if (
        len(declared_names) != len(set(declared_names))
        or set(case.expected) != set(declared_names)
        or any(dtype not in _OUTPUT_DTYPES for _, dtype in declared_fields)
    ):
        raise _error(
            "baseline_input_invalid",
            "baseline must cover every resolved manifest output",
            case.operator_id,
        )
    return declared_fields


def _run_child(
    *,
    manifest: Mapping[str, object],
    case: BaselineCase,
    parameters: Mapping[str, object],
    output_fields: list[tuple[str, str]],
    implementation_artifact: _VerifiedArtifact,
    dependency_artifacts: list[_VerifiedArtifact],
    timeout_seconds: float,
) -> tuple[dict[str, list[object]], dict[str, list[object]]]:
    timeout_seconds = _validated_timeout(timeout_seconds, case.operator_id)
    with (
        TemporaryDirectory(prefix="oxq-baseline-") as directory,
    ):
        root = Path(directory)
        try:
            implementation_path, dependency_paths = _materialize_artifacts(
                root / "artifacts",
                implementation_artifact,
                dependency_artifacts,
            )
        except OSError:
            raise _error(
                "provider_import_failed",
                "verified runtime artifact is unavailable",
                case.operator_id,
            ) from None
        request = {
            "implementation_artifact": implementation_path,
            "dependency_artifacts": dependency_paths,
            "module": manifest["module"],
            "callable": manifest["callable"],
            "parameters": parameters,
            "input": case.input,
            "output_fields": [{"name": name, "dtype": dtype} for name, dtype in output_fields],
            "output_alignment": cast(Mapping[str, object], manifest["output"])["alignment"],
        }
        try:
            response = run_exact_wheel_request(
                request,
                [implementation_path, *dependency_paths],
                timeout_seconds=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raise _error(
                "provider_execution_timeout",
                "provider baseline execution timed out",
                case.operator_id,
            ) from None
        except OSError:
            raise _error(
                "provider_execution_failed",
                "provider baseline child process failed",
                case.operator_id,
            ) from None
        except ValueError:
            raise _error(
                "provider_execution_failed",
                "provider baseline child response is invalid",
                case.operator_id,
            ) from None

    if response["status"] == "error":
        code = response.get("code")
        if not isinstance(code, str) or code not in _CHILD_FAILURE_CODES:
            code = "provider_execution_failed"
        raise _error(code, "provider baseline execution failed", case.operator_id)
    outputs = response.get("outputs")
    repeated_outputs = response.get("repeated_outputs")
    expected_names = {name for name, _ in output_fields}
    if (
        response["status"] != "ok"
        or not isinstance(outputs, dict)
        or not isinstance(repeated_outputs, dict)
        or set(outputs) != expected_names
        or set(repeated_outputs) != expected_names
        or not all(isinstance(value, list) for value in outputs.values())
        or not all(isinstance(value, list) for value in repeated_outputs.values())
    ):
        raise _error(
            "provider_execution_failed",
            "provider baseline child response is invalid",
            case.operator_id,
        )
    return (
        cast(dict[str, list[object]], outputs),
        cast(dict[str, list[object]], repeated_outputs),
    )


def _validated_timeout(timeout_seconds: float, operator_id: str) -> float:
    try:
        if type(timeout_seconds) not in {int, float}:
            raise TypeError("timeout is not numeric")
        normalized = float(timeout_seconds)
    except (OverflowError, TypeError, ValueError):
        normalized = math.nan
    if not math.isfinite(normalized) or normalized <= 0:
        raise _error(
            "provider_execution_timeout",
            "provider baseline execution timed out",
            operator_id,
        )
    return normalized


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    if "pytest" in sys.modules:
        environment["OXQ_BASELINE_TEST_RUNTIME"] = "1"
        environment["OXQ_BASELINE_TEST_RUNTIME_PATHS"] = os.pathsep.join(path for path in sys.path if "site-packages" in Path(path).parts)
    return environment


def _materialize_artifacts(
    root: Path,
    implementation: _VerifiedArtifact,
    dependencies: list[_VerifiedArtifact],
) -> tuple[str, list[str]]:
    paths: list[Path] = []
    root.mkdir()
    for index, verified in enumerate((implementation, *dependencies)):
        destination_root = root / str(index)
        destination_root.mkdir()
        destination = destination_root / verified.artifact.filename
        digest = hashlib.sha256()
        verified.snapshot.seek(0)
        with destination.open("wb") as target:
            while chunk := verified.snapshot.read(_ARTIFACT_COPY_CHUNK_BYTES):
                target.write(chunk)
                digest.update(chunk)
        verified.snapshot.seek(0)
        if f"sha256:{digest.hexdigest()}" != verified.artifact.digest:
            raise OSError("runtime artifact snapshot digest mismatch")
        destination.chmod(0o444)
        paths.append(destination)
    return str(paths[0]), [str(path) for path in paths[1:]]


def _read_response(
    response_path: Path,
    returncode: int,
    operator_id: str,
    response_secret: bytes,
) -> dict[str, object]:
    if returncode == _PROVIDER_POLICY_VIOLATION_EXIT_CODE:
        raise _error(
            "provider_import_failed",
            "provider attempted process creation outside the verified closure",
            operator_id,
        )
    try:
        with response_path.open("rb") as stream:
            raw = stream.read(_MAX_CHILD_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_CHILD_RESPONSE_BYTES:
            raise ValueError("child response is too large")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
        if returncode != 0 or not isinstance(value, dict):
            raise ValueError("invalid child response")
        auth = value.pop("auth", None)
        if not isinstance(auth, str):
            raise ValueError("missing child response authentication")
        expected_auth = (
            "hmac-sha256:"
            + hmac.new(
                response_secret,
                json.dumps(
                    value,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        )
        if not hmac.compare_digest(auth, expected_auth):
            raise ValueError("invalid child response authentication")
        status = value.get("status")
        valid_ok = status == "ok" and set(value) == {
            "status",
            "outputs",
            "repeated_outputs",
        }
        valid_error = status == "error" and set(value) == {"status", "code"}
        if not valid_ok and not valid_error:
            raise ValueError("invalid child response fields")
        return cast(dict[str, object], value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise _error(
            "provider_execution_failed",
            "provider baseline child response is invalid",
            operator_id,
        ) from None


def _assert_expected(
    case: BaselineCase,
    output_field: str,
    actual: list[object],
    output_dtype: str,
) -> None:
    expected = case.expected[output_field]
    try:
        if (
            not isinstance(expected, list)
            or len(actual) != len(expected)
            or [value is None for value in actual] != [value is None for value in expected]
        ):
            raise TypeError("baseline output shape or missing positions differ")
        if output_dtype == "float64":
            if not all(_valid_scalar(value, output_dtype) for value in (*actual, *expected)):
                raise TypeError("float output contains an invalid scalar")
            np.testing.assert_allclose(
                np.asarray([value for value in actual if value is not None], dtype=float),
                np.asarray([value for value in expected if value is not None], dtype=float),
                rtol=float(cast(float, case.tolerance["relative"])),
                atol=float(cast(float, case.tolerance["absolute"])),
            )
        else:
            if not all(_valid_scalar(value, output_dtype) for value in (*actual, *expected)):
                raise TypeError("exact output contains an invalid scalar")
            if actual != expected:
                raise AssertionError("exact output differs")
    except (AssertionError, KeyError, OverflowError, TypeError, ValueError):
        raise _error(
            "baseline_mismatch",
            "provider output does not match the numerical baseline",
            case.operator_id,
        ) from None


def _assert_deterministic(
    case: BaselineCase,
    manifest: Mapping[str, object],
    output_dtype: str,
    actual: list[object],
    repeated: list[object],
) -> None:
    determinism = cast(Mapping[str, object], manifest["determinism"])
    try:
        if (
            len(actual) != len(repeated)
            or [value is None for value in actual] != [value is None for value in repeated]
            or not all(_valid_scalar(value, output_dtype) for value in (*actual, *repeated))
        ):
            raise AssertionError("repeated output shape or values are invalid")
        if output_dtype == "float64" and not cast(
            bool,
            determinism["bitwise_deterministic"],
        ):
            tolerance = cast(Mapping[str, object], determinism["tolerance"])
            np.testing.assert_allclose(
                np.asarray(
                    [value for value in actual if value is not None],
                    dtype=float,
                ),
                np.asarray(
                    [value for value in repeated if value is not None],
                    dtype=float,
                ),
                rtol=float(cast(float, tolerance["relative"])),
                atol=float(cast(float, tolerance["absolute"])),
            )
        elif json.dumps(actual, allow_nan=False, separators=(",", ":")) != json.dumps(
            repeated,
            allow_nan=False,
            separators=(",", ":"),
        ):
            raise AssertionError("repeated output differs")
    except (AssertionError, KeyError, OverflowError, TypeError, ValueError):
        raise _error(
            "baseline_mismatch",
            "provider output is not deterministic",
            case.operator_id,
        ) from None


def _valid_scalar(value: object, output_dtype: str) -> bool:
    if value is None:
        return True
    if output_dtype == "float64":
        return _valid_float64_scalar(value)
    if output_dtype == "int64":
        return _valid_int64_scalar(value)
    if output_dtype == "boolean":
        return type(value) is bool
    if output_dtype == "string":
        return type(value) is str
    if output_dtype == "date":
        return isinstance(value, str) and _valid_iso_date(value)
    if output_dtype == "datetime":
        return isinstance(value, str) and _valid_iso_datetime(value)
    return False


def _valid_float64_scalar(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        converted = float(cast(int | float, value))
        return math.isfinite(converted) and (type(value) is float or int(converted) == value)
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_int64_scalar(value: object) -> bool:
    return type(value) is int and _INT64_MIN <= value <= _INT64_MAX


def _valid_iso_date(value: str) -> bool:
    try:
        return len(value) == 10 and value[4] == "-" and value[7] == "-" and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _valid_iso_datetime(value: str) -> bool:
    if len(value) <= 10 or value[10] != "T":
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _child_script_path() -> Path:
    return Path(__file__).with_name("_baseline_child.py").resolve(strict=True)


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate child response key")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    del value
    raise ValueError("non-standard JSON number")


def _error(
    code: str,
    message: str,
    operator_id: str | None = None,
) -> OperatorCertificationError:
    return OperatorCertificationError(
        code,
        message,
        stage="baseline",
        operator_id=operator_id,
    )
