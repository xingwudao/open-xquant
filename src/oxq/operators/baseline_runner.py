"""Execute numerical baselines in an isolated exact-wheel child process."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
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
_OUTPUT_DTYPES = {"boolean", "int64", "float64", "string", "date", "datetime"}
_AUXILIARY_REQUIREMENTS = {
    "requires_benchmark",
    "requires_industry_data",
    "requires_market_cap_data",
    "requires_fundamental_data",
}
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


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
            for output_field, output_dtype in output_fields:
                actual = _run_child(
                    manifest=operator.manifest,
                    case=case,
                    parameters=resolved_parameters,
                    output_field=output_field,
                    output_dtype=output_dtype,
                    implementation_artifact=_implementation_artifact(
                        operator.implementation_artifact,
                        verified_artifacts,
                        case.operator_id,
                    ),
                    dependency_artifacts=[str(path) for artifact, path in verified_artifacts if artifact.role == "runtime-dependency"],
                    timeout_seconds=timeout_seconds,
                )
                _assert_expected(case, output_field, actual, output_dtype)
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
    candidate_paths = {operator.implementation_artifact.resolve() for operator in candidate.operators}
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
    matches = [path for artifact, path in verified_artifacts if artifact.role == "implementation" and path == resolved]
    if len(matches) != 1:
        raise _error(
            "provider_import_failed",
            "operator implementation artifact is not uniquely verified",
            operator_id,
        )
    return str(matches[0])


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
    output_field: str,
    output_dtype: str,
    implementation_artifact: str,
    dependency_artifacts: list[str],
    timeout_seconds: float,
) -> list[object]:
    request = {
        "implementation_artifact": implementation_artifact,
        "dependency_artifacts": dependency_artifacts,
        "module": manifest["module"],
        "callable": manifest["callable"],
        "parameters": parameters,
        "input": case.input,
        "output_field": output_field,
        "output_dtype": output_dtype,
        "output_alignment": cast(Mapping[str, object], manifest["output"])["alignment"],
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
            returncode = _run_child_process(
                [
                    sys.executable,
                    "-I",
                    str(_child_script_path()),
                    str(request_path),
                    str(response_path),
                ],
                timeout_seconds,
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
        response = _read_response(response_path, returncode, case.operator_id)

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


def _run_child_process(command: list[str], timeout_seconds: float) -> int:
    if os.name == "nt":
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=cast(
                int,
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            ),
        )
    else:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    try:
        process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        process.communicate()
        raise
    return int(process.returncode)


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
    else:
        descendants = _posix_descendant_pids(process.pid)
        try:
            os.killpg(process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        descendants.update(_posix_descendant_pids(process.pid))
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if process.poll() is None:
        process.kill()


def _posix_descendant_pids(root_pid: int) -> set[int]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    children_by_parent: dict[int, set[int]] = {}
    for line in result.stdout.splitlines():
        try:
            pid_text, parent_text = line.split()
            pid = int(pid_text)
            parent = int(parent_text)
        except ValueError:
            continue
        children_by_parent.setdefault(parent, set()).add(pid)
    descendants: set[int] = set()
    pending = list(children_by_parent.get(root_pid, set()))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children_by_parent.get(pid, set()))
    return descendants


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
