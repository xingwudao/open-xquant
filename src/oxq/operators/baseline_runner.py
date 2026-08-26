"""Execute numerical baselines in an isolated exact-wheel child process."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np

from oxq.operators.certification import (
    _load_reference_validator,
    _load_schema,
    _snapshot_contract_surface,
    _validate_schema,
)
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import BaselineCase, BaselineResult, BuildArtifact, ContractCertification

_CHILD_FAILURE_CODES = {
    "provider_import_failed",
    "provider_execution_failed",
    "provider_mutated_input",
    "provider_alignment_failed",
    "baseline_mismatch",
}


def run_research_baselines(
    candidate: ContractCertification,
    runtime_artifacts: Sequence[BuildArtifact],
    timeout_seconds: float = 30,
) -> tuple[BaselineResult, ...]:
    """Run every declared baseline against only the verified provider wheels."""
    artifacts = tuple(runtime_artifacts)
    verified_artifacts = _verified_artifacts(candidate, artifacts)
    operators = {
        (
            cast(str, item.manifest["operator_id"]),
            cast(str, item.manifest["operator_version"]),
        ): item
        for item in candidate.operators
    }
    if not operators or len(operators) != len(candidate.operators):
        raise _error("baseline_input_invalid", "baseline operator identity is invalid")

    cases_by_operator = {identity: 0 for identity in operators}
    results: list[BaselineResult] = []
    with _snapshot_contract_surface() as (surface_bytes, surface_paths):
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
                quant_panel_schema,
                validate_quant_panel,
            )
            _validate_case_parameters(
                case,
                operator.manifest,
                validate_parameters,
            )
            output_field = _output_field(case, operator.manifest)
            actual = _run_child(
                manifest=operator.manifest,
                case=case,
                output_field=output_field,
                implementation_artifact=_implementation_artifact(
                    operator.implementation_artifact,
                    verified_artifacts,
                    case.operator_id,
                ),
                dependency_artifacts=[
                    str(path)
                    for artifact, path in verified_artifacts
                    if artifact.role == "runtime-dependency"
                ],
                timeout_seconds=timeout_seconds,
            )
            _assert_expected(case, actual)
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
) -> list[tuple[BuildArtifact, Path]]:
    if artifacts != candidate.artifacts or not artifacts:
        raise _error(
            "provider_import_failed",
            "runtime artifacts do not match the verified contract artifacts",
        )
    verified: list[tuple[BuildArtifact, Path]] = []
    try:
        for artifact in artifacts:
            path = artifact.wheel_path.resolve(strict=True)
            if not path.is_file() or _sha256_file(path) != artifact.digest:
                raise OSError("runtime artifact digest mismatch")
            verified.append((artifact, path))
    except OSError:
        raise _error(
            "provider_import_failed",
            "verified runtime artifact is unavailable",
        ) from None
    implementation_paths = {path for artifact, path in verified if artifact.role == "implementation"}
    candidate_paths = {
        operator.implementation_artifact.resolve()
        for operator in candidate.operators
    }
    if not candidate_paths.issubset(implementation_paths):
        raise _error(
            "provider_import_failed",
            "operator implementation artifact is not verified",
        )
    return verified


def _implementation_artifact(
    implementation_artifact: Path,
    verified_artifacts: list[tuple[BuildArtifact, Path]],
    operator_id: str,
) -> str:
    resolved = implementation_artifact.resolve()
    matches = [
        path
        for artifact, path in verified_artifacts
        if artifact.role == "implementation" and path == resolved
    ]
    if len(matches) != 1:
        raise _error(
            "provider_import_failed",
            "operator implementation artifact is not uniquely verified",
            operator_id,
        )
    return str(matches[0])


def _validate_case_input(
    case: BaselineCase,
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
    except OperatorCertificationError:
        raise
    except Exception:
        raise _error(
            "baseline_input_invalid",
            "baseline input violates frozen QuantPanel semantics",
            case.operator_id,
        ) from None


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


def _output_field(
    case: BaselineCase,
    manifest: Mapping[str, object],
) -> str:
    if len(case.expected) != 1:
        raise _error(
            "baseline_input_invalid",
            "baseline must declare exactly one expected output field",
            case.operator_id,
        )
    output_field = next(iter(case.expected))
    if not isinstance(output_field, str) or not output_field:
        raise _error(
            "baseline_input_invalid",
            "baseline expected output field is invalid",
            case.operator_id,
        )
    try:
        definitions = cast(Mapping[str, Mapping[str, object]], manifest["parameters"])
        resolved_parameters = {
            name: case.parameters.get(name, definition["default"])
            for name, definition in definitions.items()
        }
        output = cast(Mapping[str, object], manifest["output"])
        fields = cast(list[Mapping[str, object]], output["fields"])
        declared_fields = {
            cast(str, field["name_template"]).format(**resolved_parameters)
            for field in fields
        }
    except (IndexError, KeyError, TypeError, ValueError):
        raise _error(
            "baseline_input_invalid",
            "manifest output field cannot be resolved for the baseline",
            case.operator_id,
        ) from None
    if output_field not in declared_fields:
        raise _error(
            "baseline_input_invalid",
            "baseline expected output is not declared by the manifest",
            case.operator_id,
        )
    return output_field


def _run_child(
    *,
    manifest: Mapping[str, object],
    case: BaselineCase,
    output_field: str,
    implementation_artifact: str,
    dependency_artifacts: list[str],
    timeout_seconds: float,
) -> list[object]:
    request = {
        "implementation_artifact": implementation_artifact,
        "dependency_artifacts": dependency_artifacts,
        "module": manifest["module"],
        "callable": manifest["callable"],
        "parameters": case.parameters,
        "input": case.input,
        "output_field": output_field,
    }
    try:
        request_bytes = json.dumps(
            request,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise _error(
            "baseline_input_invalid",
            "baseline request is not strict JSON",
            case.operator_id,
        ) from None

    if timeout_seconds <= 0:
        raise _error(
            "provider_execution_timeout",
            "provider baseline execution timed out",
            case.operator_id,
        )
    with TemporaryDirectory(prefix="oxq-baseline-") as directory:
        root = Path(directory)
        request_path = root / "request.json"
        response_path = root / "response.json"
        request_path.write_bytes(request_bytes)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(_child_script_path()),
                    str(request_path),
                    str(response_path),
                ],
                check=False,
                capture_output=True,
                timeout=timeout_seconds,
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
        response = _read_response(response_path, result.returncode, case.operator_id)

    if response["status"] == "error":
        code = response.get("code")
        if not isinstance(code, str) or code not in _CHILD_FAILURE_CODES:
            code = "provider_execution_failed"
        raise _error(code, "provider baseline execution failed", case.operator_id)
    output = response.get("output")
    if response["status"] != "ok" or not isinstance(output, list):
        raise _error(
            "provider_execution_failed",
            "provider baseline child response is invalid",
            case.operator_id,
        )
    return output


def _read_response(
    response_path: Path,
    returncode: int,
    operator_id: str,
) -> dict[str, object]:
    try:
        raw = response_path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
        if returncode != 0 or not isinstance(value, dict):
            raise ValueError("invalid child response")
        status = value.get("status")
        valid_ok = status == "ok" and set(value) == {"status", "output"}
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


def _assert_expected(case: BaselineCase, actual: list[object]) -> None:
    expected = next(iter(case.expected.values()))
    try:
        if not isinstance(expected, list):
            raise TypeError("expected output is not a list")
        actual_values = np.asarray(
            [np.nan if value is None else value for value in actual],
            dtype=float,
        )
        expected_values = np.asarray(
            [np.nan if value is None else value for value in expected],
            dtype=float,
        )
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=float(cast(float, case.tolerance["relative"])),
            atol=float(cast(float, case.tolerance["absolute"])),
            equal_nan=True,
        )
    except (AssertionError, KeyError, TypeError, ValueError):
        raise _error(
            "baseline_mismatch",
            "provider output does not match the numerical baseline",
            case.operator_id,
        ) from None


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
