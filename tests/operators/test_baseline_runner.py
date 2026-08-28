"""Exact-wheel numerical baseline execution and promotion tests."""

from __future__ import annotations

import base64
import builtins
import hashlib
import importlib
import io
import json
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import oxq.operators.baseline_runner as baseline_runner
import oxq.operators.certification as certification
import oxq.operators.runtime_protocol as runtime_protocol
from oxq.operators import _baseline_child, _exact_wheel_child
from oxq.operators.baseline_runner import run_research_baselines
from oxq.operators.certification import certify_provider, validate_provider_contract
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import (
    BaselineCase,
    BuildArtifact,
    ContractCandidate,
    ContractCertification,
    ProviderSubmission,
)
from oxq.operators.submission import load_provider_submission
from tests.operators.helpers import (
    CATALOG_NAME,
    COMPATIBILITY_ROOT,
    rewrite_json,
    write_provider_repository,
)


@pytest.fixture(autouse=True)
def _provide_explicit_fixture_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_paths = [
        path for path in sys.path if "site-packages" in Path(path).parts
    ]

    def run_fixture_request(
        request: Mapping[str, object],
        wheel_snapshots: Sequence[str | Path],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        return runtime_protocol.run_exact_wheel_request(
            request,
            wheel_snapshots,
            timeout_seconds=timeout_seconds,
            _test_runtime_paths=runtime_paths,
        )

    monkeypatch.setattr(
        baseline_runner,
        "run_exact_wheel_request",
        run_fixture_request,
    )


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _wheel_bytes(
    distribution: str,
    package: str,
    source: str,
    *,
    purelib_data: bool = False,
    prefix_data: Mapping[str, str] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    dist_info = distribution.replace("-", "_")
    with zipfile.ZipFile(buffer, "w") as archive:
        package_path = f"{package}/__init__.py"
        if purelib_data:
            package_path = f"{dist_info}-1.0.0.data/purelib/{package_path}"
        archive.writestr(package_path, source)
        archive.writestr(
            f"{dist_info}-1.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            f"{dist_info}-1.0.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.0.0\n",
        )
        archive.writestr(f"{dist_info}-1.0.0.dist-info/RECORD", "")
        for relative_path, value in (prefix_data or {}).items():
            archive.writestr(
                f"{dist_info}-1.0.0.data/data/{relative_path}",
                value,
            )
    return buffer.getvalue()


DEPENDENCY_SOURCE = """
def rolling_mean(values, window):
    return [
        None if index + 1 < window else sum(values[index + 1 - window:index + 1]) / window
        for index in range(len(values))
    ]
"""

SUCCESS_SOURCE = """
import pandas as pd
from baseline_dependency import rolling_mean

def sma(frame, *, window):
    return pd.Series(
        rolling_mean(frame["close"].tolist(), window),
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""


def _panel() -> dict[str, object]:
    return {
        "schema_version": 1,
        "primary_key": ["date", "code"],
        "columns": [{"name": "close", "dtype": "float64", "required": True}],
        "context": {
            "timezone": "Asia/Shanghai",
            "calendar": "XSHG",
            "frequency": "1d",
            "timestamp_semantics": "bar_close",
            "currency": "CNY",
            "price_adjustment": "raw",
            "data_version": "v1",
            "source": "literal-test",
        },
        "alignment": "preserve_input_order",
        "records": [
            {"date": "2026-08-24", "code": "000001.SZ", "close": 1.0},
            {"date": "2026-08-25", "code": "000001.SZ", "close": 2.0},
            {"date": "2026-08-26", "code": "000001.SZ", "close": 3.0},
        ],
    }


def _manifest(implementation_digest: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract_version": 1,
        "operator_id": "fixture.baseline.sma",
        "operator_version": "1.0.0",
        "semantic_name": "SMA",
        "distribution": "baseline-provider",
        "module": "baseline_provider",
        "callable": "sma",
        "execution_scope": "time_series",
        "lifecycle": "stateless",
        "causality": "past_only",
        "availability": "close_t",
        "mutates_input": False,
        "availability_depends_on_input": False,
        "input": {
            "required_columns": ["close"],
            "optional_columns": [],
            "supported_dtypes": ["float64"],
            "minimum_assets": 1,
            "minimum_time_length": 1,
            "requires_complete_cross_section": False,
            "requires_benchmark": False,
            "requires_industry_data": False,
            "requires_market_cap_data": False,
            "requires_fundamental_data": False,
            "requires_sorted_input": False,
            "required_context": [
                "timezone",
                "calendar",
                "frequency",
                "timestamp_semantics",
                "currency",
                "price_adjustment",
                "data_version",
                "source",
            ],
        },
        "parameters": {
            "window": {
                "type": "integer",
                "default": 3,
                "required": False,
                "constraints": {"minimum": 1},
                "unit": "sessions",
                "affects_warmup": True,
                "affects_output_fields": True,
                "affects_causality": False,
                "affects_availability": False,
            }
        },
        "output": {
            "fields": [
                {
                    "name_template": "sma_{window}",
                    "dtype": "float64",
                    "value_range": "finite_or_nan",
                }
            ],
            "alignment": "canonical_order",
            "warmup": "window - 1",
            "nan_policy": "propagate",
            "multiple_outputs": False,
        },
        "determinism": {
            "bitwise_deterministic": True,
            "random_seed_required": False,
            "tolerance": {"absolute": 0, "relative": 0},
            "tested_platforms": ["test-python3.12"],
        },
        "implementation": {
            "package_version": "1.0.0",
            "source_commit": "git-sha1:" + "a" * 40,
            "source_files": ["src/baseline_provider.py"],
            "source_tree_digest": "sha256:" + "b" * 64,
            "implementation_digest": implementation_digest,
            "build_identifier": "baseline-build-1",
        },
    }


def _artifact(
    path: Path,
    *,
    distribution: str,
    role: str,
    build_identifier: str,
) -> BuildArtifact:
    return BuildArtifact(
        distribution=distribution,
        version="1.0.0",
        filename=path.name,
        role=role,
        build_identifier=build_identifier,
        digest=_sha256(path.read_bytes()),
        wheel_path=path,
    )


def _contract(
    tmp_path: Path,
    provider_source: str = SUCCESS_SOURCE,
    *,
    input_panel: dict[str, object] | None = None,
    parameters: dict[str, object] | None = None,
    expected: dict[str, object] | None = None,
    module: str = "baseline_provider",
) -> ContractCertification:
    tmp_path.mkdir(parents=True, exist_ok=True)
    provider_path = tmp_path / "baseline_provider-1.0.0-py3-none-any.whl"
    dependency_path = tmp_path / "baseline_dependency-1.0.0-py3-none-any.whl"
    provider_path.write_bytes(_wheel_bytes("baseline-provider", "baseline_provider", provider_source))
    dependency_path.write_bytes(
        _wheel_bytes(
            "baseline-dependency",
            "baseline_dependency",
            DEPENDENCY_SOURCE,
        )
    )
    provider_artifact = _artifact(
        provider_path,
        distribution="baseline-provider",
        role="implementation",
        build_identifier="baseline-build-1",
    )
    dependency_artifact = _artifact(
        dependency_path,
        distribution="baseline-dependency",
        role="runtime-dependency",
        build_identifier="baseline-dependency-build-1",
    )
    manifest = _manifest(provider_artifact.digest)
    manifest["module"] = module
    manifest_path = tmp_path / "fixture.baseline.sma.operator.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = ContractCandidate(
        manifest=manifest,
        binding={"certification_state": "contract-valid"},
        manifest_path=manifest_path,
        implementation_artifact=provider_path,
    )
    case = BaselineCase(
        case_id="sma-3",
        operator_id="fixture.baseline.sma",
        operator_version="1.0.0",
        parameters={"window": 3} if parameters is None else parameters,
        input=_panel() if input_panel is None else input_panel,
        expected={"sma_3": [None, None, 2.0]} if expected is None else expected,
        tolerance={"absolute": 0.0, "relative": 0.0},
    )
    return ContractCertification(
        provider="baseline-provider",
        release="1.0.0",
        submission_commit="git-sha1:" + "c" * 40,
        source_commit="git-sha1:" + "a" * 40,
        source_root=tmp_path,
        operators=(candidate,),
        artifacts=(provider_artifact, dependency_artifact),
        baseline_cases=(case,),
    )


def _assert_failure(
    candidate: ContractCertification,
    code: str,
    *,
    timeout_seconds: float = 5,
) -> None:
    with pytest.raises(OperatorCertificationError) as caught:
        run_research_baselines(
            candidate,
            candidate.artifacts,
            timeout_seconds=timeout_seconds,
        )
    assert caught.value.code == code
    assert caught.value.stage == "baseline"
    assert caught.value.operator_id == "fixture.baseline.sma"


def _with_output_dtype(
    candidate: ContractCertification,
    dtype: str,
) -> ContractCertification:
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["output"]["fields"][0]["dtype"] = dtype
    return replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
    )


def _with_output_alignment(
    candidate: ContractCertification,
    alignment: str,
) -> ContractCertification:
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["output"]["alignment"] = alignment
    return replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
    )


def test_executes_exact_provider_and_dependency_wheels_without_parent_imports(
    tmp_path: Path,
) -> None:
    candidate = _contract(tmp_path)
    before_path = list(sys.path)
    before_modules = set(sys.modules)

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]
    assert sys.path == before_path
    assert set(sys.modules).difference(before_modules).isdisjoint({"baseline_provider", "baseline_dependency"})
    assert not any(name == "baseline_provider" or name.startswith("baseline_provider.") for name in sys.modules)


def test_executes_equant_style_groupby_code_with_key_columns_preserved(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd

def sma(frame, *, window):
    assert list(frame.index.names) == ["__oxq_date", "__oxq_code"]
    assert frame.index.get_level_values("__oxq_date").tolist() == frame["date"].tolist()
    assert frame.index.get_level_values("__oxq_code").tolist() == frame["code"].tolist()
    output = frame.groupby("code", sort=False)["close"].transform(
        lambda values: values.rolling(window).mean()
    )
    return pd.Series(output, index=frame.index, name=f"sma_{window}")
"""
    candidate = _contract(tmp_path, source)

    results = run_research_baselines(candidate, candidate.artifacts, timeout_seconds=5)

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_rejects_float_output_for_boolean_manifest_field(tmp_path: Path) -> None:
    source = """
import pandas as pd

def sma(frame, *, window):
    return pd.Series([0.0, 0.0, 1.0], index=frame.index, name=f"sma_{window}")
"""
    candidate = _with_output_dtype(
        _contract(tmp_path, source, expected={"sma_3": [False, False, True]}),
        "boolean",
    )

    _assert_failure(candidate, "baseline_mismatch")


def test_accepts_exact_string_output_for_string_manifest_field(tmp_path: Path) -> None:
    source = """
import pandas as pd

def sma(frame, *, window):
    return pd.Series(["low", "mid", "high"], index=frame.index, name=f"sma_{window}")
"""
    candidate = _with_output_dtype(
        _contract(tmp_path, source, expected={"sma_3": ["low", "mid", "high"]}),
        "string",
    )

    results = run_research_baselines(candidate, candidate.artifacts, timeout_seconds=5)

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_accepts_date_objects_for_date_manifest_field(tmp_path: Path) -> None:
    source = """
import datetime
import pandas as pd

def sma(frame, *, window):
    values = [datetime.date.fromisoformat(value) for value in frame["date"]]
    return pd.Series(values, index=frame.index, name=f"sma_{window}", dtype="object")
"""
    expected = ["2026-08-24", "2026-08-25", "2026-08-26"]
    candidate = _with_output_dtype(
        _contract(tmp_path, source, expected={"sma_3": expected}),
        "date",
    )

    results = run_research_baselines(candidate, candidate.artifacts, timeout_seconds=5)

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


@pytest.mark.parametrize(
    ("dtype", "source_values", "expected"),
    [
        ("float64", "[False, False, True]", [0.0, 0.0, 1.0]),
        ("int64", "[1.0, 2.0, 3.0]", [1, 2, 3]),
        ("boolean", "[0, 0, 1]", [False, False, True]),
        ("string", "[1, 2, 3]", ["1", "2", "3"]),
        (
            "date",
            "frame['date'].tolist()",
            ["2026-08-24", "2026-08-25", "2026-08-26"],
        ),
        (
            "datetime",
            "[value + 'T12:34:56' for value in frame['date']]",
            [
                "2026-08-24T12:34:56",
                "2026-08-25T12:34:56",
                "2026-08-26T12:34:56",
            ],
        ),
    ],
)
def test_rejects_wrong_actual_scalar_type_for_manifest_field(
    tmp_path: Path,
    dtype: str,
    source_values: str,
    expected: list[object],
) -> None:
    source = (
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        f"    values = {source_values}\n"
        "    return pd.Series(values, index=frame.index, name=f'sma_{window}')\n"
    )
    candidate = _with_output_dtype(
        _contract(tmp_path, source, expected={"sma_3": expected}),
        dtype,
    )

    _assert_failure(candidate, "baseline_mismatch")


@pytest.mark.parametrize(
    ("dtype", "source_values", "expected"),
    [
        ("int64", "[1, 2, 3]", [1, 2, 3]),
        ("boolean", "[False, False, True]", [False, False, True]),
        (
            "datetime",
            "[pd.Timestamp(value + 'T12:34:56') for value in frame['date']]",
            [
                "2026-08-24T12:34:56",
                "2026-08-25T12:34:56",
                "2026-08-26T12:34:56",
            ],
        ),
    ],
)
def test_accepts_manifest_typed_exact_outputs(
    tmp_path: Path,
    dtype: str,
    source_values: str,
    expected: list[object],
) -> None:
    source = (
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        f"    values = {source_values}\n"
        "    return pd.Series(values, index=frame.index, name=f'sma_{window}')\n"
    )
    candidate = _with_output_dtype(
        _contract(tmp_path, source, expected={"sma_3": expected}),
        dtype,
    )

    results = run_research_baselines(candidate, candidate.artifacts, timeout_seconds=5)

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


@pytest.mark.parametrize(
    ("value", "passes"),
    [
        (-(2**63), True),
        (2**63 - 1, True),
        (-(2**63) - 1, False),
        (2**63, False),
    ],
    ids=["minimum", "maximum", "below-minimum", "above-maximum"],
)
def test_enforces_signed_int64_scalar_bounds(
    tmp_path: Path,
    value: int,
    passes: bool,
) -> None:
    source = (
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        f"    return pd.Series([{value}, None, {value}], "
        "index=frame.index, name=f'sma_{window}', dtype='object')\n"
    )
    candidate = _with_output_dtype(
        _contract(tmp_path, source, expected={"sma_3": [value, None, value]}),
        "int64",
    )

    if passes:
        results = run_research_baselines(
            candidate,
            candidate.artifacts,
            timeout_seconds=5,
        )
        assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]
    else:
        _assert_failure(candidate, "baseline_mismatch")


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**63])
def test_parent_rejects_out_of_range_int64_child_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: int,
) -> None:
    candidate = _with_output_dtype(
        _contract(tmp_path / "candidate", expected={"sma_3": [value, None, value]}),
        "int64",
    )
    child = tmp_path / "out_of_range_int_child.py"
    child.write_text(
        "import hashlib, hmac, json, pathlib, sys\n"
        "secret = bytes.fromhex(sys.stdin.readline().strip())\n"
        f"values = {{'sma_3': [{value}, None, {value}]}}\n"
        "value = {'status': 'ok', 'outputs': values, 'repeated_outputs': values}\n"
        "payload = json.dumps(value, separators=(',', ':'), sort_keys=True).encode()\n"
        "value['auth'] = 'hmac-sha256:' + hmac.new(secret, payload, hashlib.sha256).hexdigest()\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(value))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_protocol, "_child_path", lambda: child)

    _assert_failure(candidate, "baseline_mismatch")


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_rejects_float64_integer_that_cannot_convert_to_finite_ieee_value(
    tmp_path: Path,
    value: int,
) -> None:
    source = (
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        f"    return pd.Series([{value}, None, {value}], "
        "index=frame.index, name=f'sma_{window}', dtype='object')\n"
    )
    candidate = _contract(
        tmp_path,
        source,
        expected={"sma_3": [value, None, value]},
    )

    _assert_failure(candidate, "baseline_mismatch")


def test_float_comparison_preserves_large_integer_distinctions(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd
def sma(frame, *, window):
    return pd.Series(
        [None, None, 9007199254740992.0],
        index=frame.index,
        name=f"sma_{window}",
    )
"""
    candidate = _contract(
        tmp_path,
        source,
        expected={"sma_3": [None, None, 9007199254740993]},
    )

    _assert_failure(candidate, "baseline_mismatch")


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_parent_rejects_unconvertible_float64_child_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: int,
) -> None:
    candidate = _contract(
        tmp_path / "candidate",
        expected={"sma_3": [value, None, value]},
    )
    child = tmp_path / "unconvertible_float_child.py"
    child.write_text(
        "import hashlib, hmac, json, pathlib, sys\n"
        "secret = bytes.fromhex(sys.stdin.readline().strip())\n"
        f"values = {{'sma_3': [{value}, None, {value}]}}\n"
        "value = {'status': 'ok', 'outputs': values, 'repeated_outputs': values}\n"
        "payload = json.dumps(value, separators=(',', ':'), sort_keys=True).encode()\n"
        "value['auth'] = 'hmac-sha256:' + hmac.new(secret, payload, hashlib.sha256).hexdigest()\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(value))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_protocol, "_child_path", lambda: child)

    _assert_failure(candidate, "baseline_mismatch")


@pytest.mark.parametrize("expression", ["float('inf')", "float('-inf')"])
def test_rejects_non_finite_float64_actual_value(
    tmp_path: Path,
    expression: str,
) -> None:
    source = (
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        f"    value = {expression}\n"
        "    return pd.Series([value, None, value], "
        "index=frame.index, name=f'sma_{window}', dtype='object')\n"
    )
    candidate = _contract(
        tmp_path,
        source,
        expected={"sma_3": [1.0, None, 1.0]},
    )

    _assert_failure(candidate, "baseline_mismatch")


@pytest.mark.parametrize(
    ("dtype", "source_values", "expected"),
    [
        ("float64", "[0.0, 0.0, 1.0]", [False, False, True]),
        ("int64", "[1, 2, 3]", [1.0, 2.0, 3.0]),
        ("boolean", "[False, False, True]", [0, 0, 1]),
        ("string", "['1', '2', '3']", [1, 2, 3]),
        ("date", "[pd.Timestamp(value).date() for value in frame['date']]", [1, 2, 3]),
        (
            "datetime",
            "[pd.Timestamp(value + 'T12:34:56') for value in frame['date']]",
            [1, 2, 3],
        ),
    ],
)
def test_rejects_wrong_expected_scalar_type_for_manifest_field(
    tmp_path: Path,
    dtype: str,
    source_values: str,
    expected: list[object],
) -> None:
    source = (
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        f"    values = {source_values}\n"
        "    return pd.Series(values, index=frame.index, name=f'sma_{window}')\n"
    )
    candidate = _with_output_dtype(
        _contract(tmp_path, source, expected={"sma_3": expected}),
        dtype,
    )

    _assert_failure(candidate, "baseline_mismatch")


def test_numeric_tolerance_applies_only_to_float64_outputs(tmp_path: Path) -> None:
    float_candidate = _contract(
        tmp_path / "float",
        SUCCESS_SOURCE.replace("[None, None, 2.0]", "[None, None, 2.001]"),
    )
    float_candidate = replace(
        float_candidate,
        baseline_cases=(
            replace(
                float_candidate.baseline_cases[0],
                tolerance={"absolute": 0.01, "relative": 0.0},
            ),
        ),
    )
    float_results = run_research_baselines(
        float_candidate,
        float_candidate.artifacts,
        timeout_seconds=5,
    )
    assert [(item.case_id, item.status) for item in float_results] == [("sma-3", "passed")]

    integer_source = """
import pandas as pd
def sma(frame, *, window):
    return pd.Series([1, 2, 4], index=frame.index, name=f"sma_{window}")
"""
    integer_candidate = _with_output_dtype(
        _contract(
            tmp_path / "integer",
            integer_source,
            expected={"sma_3": [1, 2, 3]},
        ),
        "int64",
    )
    integer_candidate = replace(
        integer_candidate,
        baseline_cases=(
            replace(
                integer_candidate.baseline_cases[0],
                tolerance={"absolute": 100.0, "relative": 100.0},
            ),
        ),
    )
    _assert_failure(integer_candidate, "baseline_mismatch")


@pytest.mark.parametrize("field", ["absolute", "relative"])
def test_rejects_nonfinite_float_tolerance_before_child_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    candidate = _contract(tmp_path)
    tolerance = {"absolute": 0.0, "relative": 0.0}
    tolerance[field] = float("inf")
    candidate = replace(
        candidate,
        baseline_cases=(replace(candidate.baseline_cases[0], tolerance=tolerance),),
    )

    def child_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"child executed: {args!r} {kwargs!r}")

    monkeypatch.setattr(baseline_runner.subprocess, "run", child_must_not_run)
    _assert_failure(candidate, "baseline_input_invalid")


def test_requires_baseline_coverage_for_every_declared_output(tmp_path: Path) -> None:
    candidate = _contract(tmp_path)
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["output"]["fields"].append(
        {
            "name_template": "ema_{window}",
            "dtype": "float64",
            "value_range": "finite_or_nan",
        }
    )
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
    )

    _assert_failure(candidate, "baseline_input_invalid")


def test_accepts_baseline_coverage_for_all_declared_outputs(tmp_path: Path) -> None:
    source = """
import pandas as pd
def sma(frame, *, window):
    return pd.DataFrame({
        f"sma_{window}": [None, None, 2.0],
        f"ema_{window}": [None, None, 2.0],
    }, index=frame.index)
"""
    candidate = _contract(tmp_path, source)
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["output"]["fields"].append(
        {
            "name_template": "ema_{window}",
            "dtype": "float64",
            "value_range": "finite_or_nan",
        }
    )
    complete_case = replace(
        candidate.baseline_cases[0],
        expected={
            "sma_3": [None, None, 2.0],
            "ema_3": [None, None, 2.0],
        },
    )
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
        baseline_cases=(complete_case,),
    )

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_rejects_provider_that_fabricates_one_requested_output_per_invocation(
    tmp_path: Path,
) -> None:
    source = """
import json
import pathlib
import pandas as pd

def sma(frame, *, window):
    request_path = pathlib.Path(__file__).parents[4] / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    requested = request.get("output_fields")
    field = request["output_field"] if requested is None else next(iter(requested))
    return pd.DataFrame({field: [None, None, 2.0]}, index=frame.index)
"""
    candidate = _contract(tmp_path, source)
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["output"]["fields"].append(
        {
            "name_template": "ema_{window}",
            "dtype": "float64",
            "value_range": "finite_or_nan",
        }
    )
    complete_case = replace(
        candidate.baseline_cases[0],
        expected={
            "sma_3": [None, None, 2.0],
            "ema_3": [None, None, 2.0],
        },
    )
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
        baseline_cases=(complete_case,),
    )

    _assert_failure(candidate, "baseline_mismatch")


def test_requires_every_output_on_each_parameterized_baseline_case(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd
def sma(frame, *, window):
    field = "alpha" if window == 3 else "beta"
    return pd.DataFrame({field: [None, None, 2.0]}, index=frame.index)
"""
    candidate = _contract(tmp_path, source, expected={"alpha": [None, None, 2.0]})
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["output"]["multiple_outputs"] = True
    manifest["output"]["fields"] = [
        {
            "name_template": "alpha",
            "dtype": "float64",
            "value_range": "finite_or_nan",
        },
        {
            "name_template": "beta",
            "dtype": "float64",
            "value_range": "finite_or_nan",
        },
    ]
    beta_case = replace(
        candidate.baseline_cases[0],
        case_id="beta-4",
        parameters={"window": 4},
        expected={"beta": [None, None, 2.0]},
    )
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
        baseline_cases=(*candidate.baseline_cases, beta_case),
    )

    _assert_failure(candidate, "baseline_input_invalid")


def test_float_tolerance_does_not_relax_missing_value_positions(tmp_path: Path) -> None:
    source = SUCCESS_SOURCE.replace(
        'rolling_mean(frame["close"].tolist(), window)',
        "[None, 0.0, 2.0]",
    )
    candidate = _contract(
        tmp_path,
        source,
        expected={"sma_3": [0.0, None, 2.0]},
    )
    candidate = replace(
        candidate,
        baseline_cases=(
            replace(
                candidate.baseline_cases[0],
                tolerance={"absolute": 100.0, "relative": 100.0},
            ),
        ),
    )

    _assert_failure(candidate, "baseline_mismatch")


def test_rejects_non_iso_date_child_output_even_when_expected_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_dates = ["2026-08-24", "2026-08-25", "2026-08-99"]
    candidate = _with_output_dtype(
        _contract(tmp_path / "candidate", expected={"sma_3": invalid_dates}),
        "date",
    )
    child = tmp_path / "invalid_date_child.py"
    child.write_text(
        "import hashlib, hmac, json, pathlib, sys\n"
        "secret = bytes.fromhex(sys.stdin.readline().strip())\n"
        f"values = {{'sma_3': {invalid_dates!r}}}\n"
        "value = {'status': 'ok', 'outputs': values, 'repeated_outputs': values}\n"
        "payload = json.dumps(value, separators=(',', ':'), sort_keys=True).encode()\n"
        "value['auth'] = 'hmac-sha256:' + hmac.new(secret, payload, hashlib.sha256).hexdigest()\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(value))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_protocol, "_child_path", lambda: child)

    _assert_failure(candidate, "baseline_mismatch")


def test_rejects_shadow_dependency_instead_of_skipping_bound_implementation(
    tmp_path: Path,
) -> None:
    wrong_implementation = """
import pandas as pd
def sma(frame, *, window):
    return pd.Series([0.0, 0.0, 0.0], index=frame.index, name=f"sma_{window}")
"""
    shadow_implementation = """
import pandas as pd
def sma(frame, *, window):
    return pd.Series([None, None, 2.0], index=frame.index, name=f"sma_{window}")
"""
    candidate = _contract(tmp_path, wrong_implementation)
    implementation, dependency = candidate.artifacts
    dependency.wheel_path.write_bytes(
        _wheel_bytes(
            "baseline-dependency",
            "baseline_provider",
            shadow_implementation,
        )
    )
    shadow_dependency = replace(
        dependency,
        digest=_sha256(dependency.wheel_path.read_bytes()),
    )
    candidate = replace(
        candidate,
        artifacts=(shadow_dependency, implementation),
    )

    _assert_failure(candidate, "provider_import_failed")


def test_rejects_dependency_callable_reexported_with_spoofed_module(
    tmp_path: Path,
) -> None:
    provider_source = """
from baseline_dependency import sma
sma.__module__ = "baseline_provider"
"""
    dependency_source = """
import pandas as pd
def sma(frame, *, window):
    return pd.Series([None, None, 2.0], index=frame.index, name=f"sma_{window}")
"""
    candidate = _contract(tmp_path, provider_source)
    implementation, dependency = candidate.artifacts
    dependency.wheel_path.write_bytes(
        _wheel_bytes(
            "baseline-dependency",
            "baseline_dependency",
            dependency_source,
        )
    )
    dependency = replace(
        dependency,
        digest=_sha256(dependency.wheel_path.read_bytes()),
    )
    candidate = replace(candidate, artifacts=(implementation, dependency))

    _assert_failure(candidate, "provider_import_failed")


def test_rejects_provider_that_mutates_traceback_reachable_input_original(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd

def sma(frame, *, window):
    try:
        raise RuntimeError("traceback access")
    except RuntimeError as exc:
        cursor = exc.__traceback__.tb_frame
        while cursor is not None:
            original = cursor.f_locals.get("invocation_original")
            if original is not None:
                original.loc[original.index[0], "close"] = 99.0
            cursor = cursor.f_back
    frame.loc[frame.index[0], "close"] = 99.0
    return pd.Series([None, None, 2.0], index=frame.index, name=f"sma_{window}")
"""

    _assert_failure(_contract(tmp_path, source), "provider_mutated_input")


@pytest.mark.parametrize("child_module", [_baseline_child, _exact_wheel_child])
def test_provider_call_cannot_reach_verifier_stack_locals(
    child_module: object,
) -> None:
    verifier_secret = object()
    assert verifier_secret is not None

    def provider(frame: object, **parameters: object) -> str:
        del frame, parameters
        try:
            raise RuntimeError("traceback access")
        except RuntimeError as exc:
            cursor = exc.__traceback__.tb_frame
            while cursor is not None:
                if "verifier_secret" in cursor.f_locals:
                    return "leaked"
                cursor = cursor.f_back
        return "isolated"

    assert child_module._invoke_provider_outside_verifier_stack(provider, object(), {}) == "isolated"


def test_rejects_provider_that_spoofs_alignment_keys_with_mutated_pandas_methods(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd
from baseline_dependency import rolling_mean

class SpoofedKeyFrame(pd.DataFrame):
    _metadata = ["_trusted_keys"]

    def to_dict(self, *args, **kwargs):
        del args, kwargs
        return [
            {"date": date, "code": code} for date, code in self._trusted_keys
        ]

def sma(frame, *, window):
    output = SpoofedKeyFrame({
        "date": list(reversed(frame["date"].tolist())),
        "code": frame["code"].tolist(),
        f"sma_{window}": rolling_mean(frame["close"].tolist(), window),
    })
    output._trusted_keys = list(frame.index)
    return output
"""

    candidate = _contract(tmp_path, source)
    with pytest.raises(OperatorCertificationError):
        run_research_baselines(
            candidate,
            candidate.artifacts,
            timeout_seconds=5,
        )


def test_rejects_undeclared_ambient_dependency(tmp_path: Path) -> None:
    candidate = _contract(tmp_path, "import jsonschema\n" + SUCCESS_SOURCE)

    _assert_failure(candidate, "provider_import_failed")


def test_rejects_ambient_dependency_imported_during_output_extraction(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd
from baseline_dependency import rolling_mean

class LazyImportSeries(pd.Series):
    def tolist(self):
        import jsonschema
        return super().tolist()

def sma(frame, *, window):
    return LazyImportSeries(
        rolling_mean(frame["close"].tolist(), window),
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    _assert_failure(_contract(tmp_path, source), "provider_execution_failed")


@pytest.mark.parametrize("dependency", ["dateutil", "pytz"])
def test_rejects_provider_direct_import_of_preloaded_environment_dependency(
    tmp_path: Path,
    dependency: str,
) -> None:
    candidate = _contract(tmp_path, f"import {dependency}\n" + SUCCESS_SOURCE)

    _assert_failure(candidate, "provider_import_failed")


@pytest.mark.parametrize("dependency", ["dateutil", "pytz"])
def test_rejects_provider_dynamic_import_of_preloaded_environment_dependency(
    tmp_path: Path,
    dependency: str,
) -> None:
    source = SUCCESS_SOURCE.replace(
        "def sma(frame, *, window):",
        f"import importlib\ndef sma(frame, *, window):\n    importlib.import_module({dependency!r})",
    )
    candidate = _contract(tmp_path, source)

    _assert_failure(candidate, "provider_execution_failed")


def test_normalizes_provider_process_creation_as_import_failure(
    tmp_path: Path,
) -> None:
    source = """
import subprocess
import sys
import pandas as pd

def sma(frame, *, window):
    completed = subprocess.run(
        [sys.executable, "-c", "print(2.0)"],
        check=True,
        capture_output=True,
        text=True,
    )
    return pd.Series(
        [None, None, float(completed.stdout)],
        index=frame.index,
        name=f"sma_{window}",
    )
"""

    _assert_failure(_contract(tmp_path, source), "provider_import_failed")


@pytest.mark.parametrize(
    "violation_statement",
    ["import jsonschema", "importlib.import_module('dateutil')"],
    ids=["ordinary-import", "dynamic-import"],
)
@pytest.mark.parametrize("after_violation", ["mutation", "alignment"])
def test_handled_unavailable_import_preserves_later_provider_failures(
    tmp_path: Path,
    after_violation: str,
    violation_statement: str,
) -> None:
    if after_violation == "mutation":
        consequence = (
            "    frame.loc[frame.index[0], 'close'] = 99.0\n"
            "    return pd.Series([None, None, 2.0], index=frame.index, name=f'sma_{window}')"
        )
    else:
        consequence = (
            "    return pd.DataFrame({\n"
            "        'date': list(reversed(frame['date'].tolist())),\n"
            "        'code': frame['code'].tolist(),\n"
            "        f'sma_{window}': [None, None, 2.0],\n"
            "    })"
        )
    source = (
        "import importlib\n"
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        "    try:\n"
        f"        {violation_statement}\n"
        "    except ImportError:\n"
        "        pass\n"
        f"{consequence}\n"
    )

    expected_code = "provider_mutated_input" if after_violation == "mutation" else "provider_alignment_failed"
    _assert_failure(_contract(tmp_path, source), expected_code)


def test_import_gate_restores_both_import_entry_points() -> None:
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    gate = _baseline_child._ProviderImportGate([])

    gate.install()
    try:
        assert builtins.__import__ != original_import
        assert importlib.import_module != original_import_module
    finally:
        gate.restore()

    assert builtins.__import__ is original_import
    assert importlib.import_module is original_import_module


def test_allows_caught_unavailable_dynamic_import_fallback(
    tmp_path: Path,
) -> None:
    source = (
        "import importlib\n"
        "import pandas as pd\n"
        "def sma(frame, *, window):\n"
        "    try:\n"
        "        importlib.import_module('unverified_package_that_does_not_exist')\n"
        "    except ImportError:\n"
        "        pass\n"
        "    return pd.Series([None, None, 2.0], index=frame.index, name=f'sma_{window}')\n"
    )

    candidate = _contract(tmp_path, source)

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_blocks_loader_execution_from_ambient_sys_path(tmp_path: Path) -> None:
    source = """
import importlib.util
import pathlib
import sys
import pandas as pd

def sma(frame, *, window):
    loaded = False
    for root in sys.path:
        candidate = pathlib.Path(root) / "six.py"
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("ambient_six", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded = module.PY3
        break
    value = 2.0 if loaded else 99.0
    return pd.Series([None, None, value], index=frame.index, name=f"sma_{window}")
"""

    _assert_failure(_contract(tmp_path, source), "baseline_mismatch")


def test_allows_pandas_runtime_to_use_its_platform_dependencies(
    tmp_path: Path,
) -> None:
    source = SUCCESS_SOURCE.replace(
        "def sma(frame, *, window):",
        "def sma(frame, *, window):\n    pandas_dates = pd.to_datetime(frame['date'])\n    assert len(pandas_dates) == len(frame)",
    )
    candidate = _contract(tmp_path, source)

    results = run_research_baselines(candidate, candidate.artifacts, timeout_seconds=5)

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_rejects_preimported_module_name_not_owned_by_implementation(
    tmp_path: Path,
) -> None:
    candidate = _contract(
        tmp_path,
        "def sma(frame, *, window):\n    return None\n",
        module="json",
    )

    _assert_failure(candidate, "provider_import_failed")


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            SUCCESS_SOURCE.replace(
                'rolling_mean(frame["close"].tolist(), window)',
                "[0.0, 0.0, 0.0]",
            ),
            "baseline_mismatch",
        ),
        (SUCCESS_SOURCE.replace('name=f"sma_{window}"', 'name="wrong"'), "baseline_mismatch"),
        (
            SUCCESS_SOURCE.replace(
                "def sma(frame, *, window):",
                'def sma(frame, *, window):\n    frame.loc[0, "close"] = 99.0',
            ),
            "provider_mutated_input",
        ),
        (
            """
import pandas as pd
def sma(frame, *, window):
    return pd.DataFrame({
        "date": list(reversed(frame["date"].tolist())),
        "code": frame["code"].tolist(),
        f"sma_{window}": [None, None, 2.0],
    })
""",
            "provider_alignment_failed",
        ),
        ("raise ImportError('provider import exploded')", "provider_import_failed"),
        (
            "def sma(frame, *, window):\n    raise RuntimeError('provider exploded')\n",
            "provider_execution_failed",
        ),
    ],
    ids=[
        "wrong-output",
        "missing-output",
        "input-mutation",
        "changed-key-row-alignment",
        "import-failure",
        "provider-exception",
    ],
)
def test_normalizes_provider_baseline_failures(
    tmp_path: Path,
    source: str,
    code: str,
) -> None:
    _assert_failure(_contract(tmp_path, source), code)


def test_normalizes_missing_provider_module(tmp_path: Path) -> None:
    _assert_failure(
        _contract(tmp_path, module="package_that_does_not_exist"),
        "provider_import_failed",
    )


def test_rejects_a_series_that_cannot_prove_original_key_alignment(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd
def sma(frame, *, window):
    return pd.Series([None, None, 2.0], name=f"sma_{window}", dtype="float64")
"""
    _assert_failure(_contract(tmp_path, source), "provider_alignment_failed")


def test_accepts_canonical_output_for_noncanonical_input(tmp_path: Path) -> None:
    panel = _panel()
    records = cast(list[dict[str, object]], panel["records"])
    panel["records"] = [records[2], records[0], records[1]]
    source = """
def sma(frame, *, window):
    result = frame.sort_values(["date", "code"])[["date", "code", "close"]].copy()
    result[f"sma_{window}"] = result["close"]
    return result[["date", "code", f"sma_{window}"]]
"""
    candidate = _with_output_alignment(
        _contract(
            tmp_path,
            source,
            input_panel=panel,
            expected={"sma_3": [1.0, 2.0, 3.0]},
        ),
        "canonical_order",
    )

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_accepts_reordered_explicit_keyed_output(tmp_path: Path) -> None:
    source = """
def sma(frame, *, window):
    result = frame.iloc[[1, 2, 0]][["date", "code", "close"]].copy()
    result[f"sma_{window}"] = result["close"]
    return result[["date", "code", f"sma_{window}"]]
"""
    candidate = _with_output_alignment(
        _contract(
            tmp_path,
            source,
            expected={"sma_3": [1.0, 2.0, 3.0]},
        ),
        "explicit_keyed_output",
    )

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_delivers_quant_panel_context_to_provider_frame(tmp_path: Path) -> None:
    source = """
import pandas as pd
def sma(frame, *, window):
    context = frame.attrs["open_xquant_context"]
    value = 2.0 if context["timezone"] == "Asia/Shanghai" else 0.0
    return pd.Series([None, None, value], index=frame.index, name=f"sma_{window}")
"""
    candidate = _contract(tmp_path, source)

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_rejects_provider_mutation_of_quant_panel_context(tmp_path: Path) -> None:
    source = """
import pandas as pd
def sma(frame, *, window):
    frame.attrs["open_xquant_context"]["timezone"] = "UTC"
    return pd.Series([None, None, 2.0], index=frame.index, name=f"sma_{window}")
"""

    _assert_failure(_contract(tmp_path, source), "provider_mutated_input")


def test_blocks_provider_access_to_preloaded_ambient_modules(tmp_path: Path) -> None:
    source = """
import sys
AMBIENT_PARSER = sys.modules["dateutil.parser"]
import pandas as pd
def sma(frame, *, window):
    return pd.Series([None, None, 2.0], index=frame.index, name=f"sma_{window}")
"""

    _assert_failure(_contract(tmp_path, source), "provider_import_failed")


def test_blocks_indirect_access_to_preloaded_ambient_modules(tmp_path: Path) -> None:
    source = """
import os
AMBIENT_PARSER = os.sys.modules["dateutil.parser"]
import pandas as pd
def sma(frame, *, window):
    return pd.Series([None, None, 2.0], index=frame.index, name=f"sma_{window}")
"""

    _assert_failure(_contract(tmp_path, source), "provider_import_failed")


def test_executes_module_from_wheel_purelib_data_layout(tmp_path: Path) -> None:
    candidate = _contract(tmp_path)
    operator = candidate.operators[0]
    implementation, dependency = candidate.artifacts
    wheel_bytes = _wheel_bytes(
        "baseline-provider",
        "baseline_provider",
        SUCCESS_SOURCE,
        purelib_data=True,
    )
    implementation.wheel_path.write_bytes(wheel_bytes)
    implementation = replace(implementation, digest=_sha256(wheel_bytes))
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["implementation"]["implementation_digest"] = implementation.digest
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
        artifacts=(implementation, dependency),
    )

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_installs_wheel_data_scheme_under_isolated_prefix(tmp_path: Path) -> None:
    source = """
from pathlib import Path
import sys
import pandas as pd
def sma(frame, *, window):
    marker = Path(sys.prefix, "share", "baseline-marker.txt").read_text()
    value = 2.0 if marker == "verified-data" else 0.0
    return pd.Series([None, None, value], index=frame.index, name=f"sma_{window}")
"""
    candidate = _contract(tmp_path, source)
    operator = candidate.operators[0]
    implementation, dependency = candidate.artifacts
    wheel_bytes = _wheel_bytes(
        "baseline-provider",
        "baseline_provider",
        source,
        prefix_data={"share/baseline-marker.txt": "verified-data"},
    )
    implementation.wheel_path.write_bytes(wheel_bytes)
    implementation = replace(implementation, digest=_sha256(wheel_bytes))
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["implementation"]["implementation_digest"] = implementation.digest
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
        artifacts=(implementation, dependency),
    )

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_preserves_declared_nullable_int64_input_dtype(tmp_path: Path) -> None:
    panel = _panel()
    panel["columns"] = [{"name": "close", "dtype": "int64", "required": True}]
    panel["records"] = [
        {"date": "2026-08-24", "code": "000001.SZ", "close": 9007199254740993},
        {"date": "2026-08-25", "code": "000001.SZ", "close": None},
        {"date": "2026-08-26", "code": "000001.SZ", "close": 3},
    ]
    source = """
def sma(frame, *, window):
    if str(frame["close"].dtype) != "Int64":
        raise TypeError("declared nullable int64 dtype was not preserved")
    result = frame["close"].copy()
    result.name = f"sma_{window}"
    return result
"""
    candidate = _with_output_dtype(
        _contract(
            tmp_path,
            source,
            input_panel=panel,
            expected={"sma_3": [9007199254740993, None, 3]},
        ),
        "int64",
    )
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["input"]["supported_dtypes"] = ["int64"]
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
    )

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_rejects_malformed_child_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "malformed_child.py"
    child.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[2]).write_bytes(b'{broken')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_protocol, "_child_path", lambda: child)

    _assert_failure(_contract(tmp_path / "candidate"), "provider_execution_failed")


def test_rejects_child_json_with_undeclared_response_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "extra_field_child.py"
    child.write_text(
        "import hashlib, hmac, json, pathlib, sys\n"
        "secret = bytes.fromhex(sys.stdin.readline().strip())\n"
        "value = {'status': 'ok', 'outputs': {'sma_3': [None, None, 2.0]}, "
        "'repeated_outputs': {'sma_3': [None, None, 2.0]}, 'extra': 1}\n"
        "payload = json.dumps(value, separators=(',', ':'), sort_keys=True).encode()\n"
        "value['auth'] = 'hmac-sha256:' + hmac.new(secret, payload, hashlib.sha256).hexdigest()\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(value))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_protocol, "_child_path", lambda: child)

    _assert_failure(_contract(tmp_path / "candidate"), "provider_execution_failed")


def test_provider_cannot_forge_discovered_verifier_response_file(
    tmp_path: Path,
) -> None:
    source = """
import json
import os
import pathlib
import tempfile

def sma(frame, *, window):
    values = {f"sma_{window}": [None, None, 2.0]}
    forged = {
        "status": "ok",
        "outputs": values,
        "repeated_outputs": values,
    }
    roots = pathlib.Path(tempfile.gettempdir()).glob("oxq-baseline-response-*")
    for root in roots:
        (root / "response.json").write_text(json.dumps(forged))
    os._exit(0)
"""

    _assert_failure(
        _contract(tmp_path / "candidate", source),
        "provider_execution_failed",
    )


def test_normalizes_child_timeout(tmp_path: Path) -> None:
    source = "import time\ndef sma(frame, *, window):\n    time.sleep(60)\n"
    _assert_failure(
        _contract(tmp_path, source),
        "provider_execution_timeout",
        timeout_seconds=0.05,
    )


@pytest.mark.parametrize(
    "timeout_seconds",
    [float("nan"), float("inf"), 10**400],
    ids=["nan", "infinity", "huge-integer"],
)
def test_rejects_nonfinite_timeout_before_child_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: float | int,
) -> None:
    candidate = _contract(tmp_path)

    def child_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"child executed: {args!r} {kwargs!r}")

    monkeypatch.setattr(baseline_runner, "run_exact_wheel_request", child_must_not_run)

    _assert_failure(
        candidate,
        "provider_execution_timeout",
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda panel: panel.pop("context"),
        lambda panel: panel["records"].append(dict(panel["records"][0])),  # type: ignore[union-attr,index]
    ],
    ids=["schema", "semantics"],
)
def test_rejects_invalid_frozen_quant_panel_before_child_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    panel = _panel()
    mutate(panel)
    candidate = _contract(tmp_path, input_panel=panel)

    def child_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"child executed: {args!r} {kwargs!r}")

    monkeypatch.setattr(baseline_runner.subprocess, "run", child_must_not_run)
    _assert_failure(candidate, "baseline_input_invalid")


@pytest.mark.parametrize(
    "requirement",
    [
        "required-column",
        "required-flag",
        "supported-dtype",
        "minimum-assets",
        "minimum-time-length",
        "required-sort-order",
    ],
)
def test_rejects_baseline_that_does_not_meet_manifest_input_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requirement: str,
) -> None:
    panel = _panel()
    candidate = _contract(tmp_path, input_panel=panel)
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    input_contract = manifest["input"]
    if requirement == "required-column":
        input_contract["required_columns"] = ["volume"]
    elif requirement == "required-flag":
        panel["columns"][0]["required"] = False
        for record in panel["records"]:
            record.pop("close")
    elif requirement == "supported-dtype":
        input_contract["supported_dtypes"] = ["int64"]
    elif requirement == "minimum-assets":
        input_contract["minimum_assets"] = 2
    elif requirement == "minimum-time-length":
        input_contract["minimum_time_length"] = 4
    else:
        input_contract["requires_sorted_input"] = True
        input_contract["required_sort_order"] = ["date", "code"]
        records = cast(list[dict[str, object]], panel["records"])
        panel["records"] = list(reversed(records))
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
    )

    def child_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"child executed: {args!r} {kwargs!r}")

    monkeypatch.setattr(baseline_runner.subprocess, "run", child_must_not_run)
    _assert_failure(candidate, "baseline_input_invalid")


def test_rejects_baseline_columns_not_declared_by_manifest(tmp_path: Path) -> None:
    panel = _panel()
    columns = cast(list[dict[str, object]], panel["columns"])
    columns.append({"name": "answer", "dtype": "float64", "required": True})
    records = cast(list[dict[str, object]], panel["records"])
    for record, answer in zip(records, [None, None, 2.0], strict=True):
        record["answer"] = answer
    source = """
import pandas as pd
def sma(frame, *, window):
    result = frame["answer"].copy()
    result.name = f"sma_{window}"
    return result
"""

    _assert_failure(
        _contract(tmp_path, source, input_panel=panel),
        "baseline_input_invalid",
    )


def test_rejects_optional_column_with_unsupported_dtype(tmp_path: Path) -> None:
    panel = _panel()
    columns = cast(list[dict[str, object]], panel["columns"])
    columns.append({"name": "label", "dtype": "string", "required": False})
    records = cast(list[dict[str, object]], panel["records"])
    for record in records:
        record["label"] = "observed"
    candidate = _contract(tmp_path, input_panel=panel)
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["input"]["optional_columns"] = ["label"]
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
    )

    _assert_failure(candidate, "baseline_input_invalid")


def test_executes_verified_immutable_artifact_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _contract(tmp_path / "candidate")
    second_case = replace(candidate.baseline_cases[0], case_id="sma-3-repeat")
    candidate = replace(
        candidate,
        baseline_cases=(*candidate.baseline_cases, second_case),
    )
    provider_path = candidate.artifacts[0].wheel_path
    original_bytes = provider_path.read_bytes()
    real_run_child = baseline_runner._run_child
    calls = 0

    def mutate_original_after_first_call(
        **kwargs: object,
    ) -> tuple[dict[str, list[object]], dict[str, list[object]]]:
        nonlocal calls
        output = real_run_child(**kwargs)  # type: ignore[arg-type]
        calls += 1
        if calls == 1:
            provider_path.write_bytes(b"mutated after verification")
        return output

    monkeypatch.setattr(baseline_runner, "_run_child", mutate_original_after_first_call)
    try:
        results = run_research_baselines(
            candidate,
            candidate.artifacts,
            timeout_seconds=5,
        )
    finally:
        provider_path.write_bytes(original_bytes)

    assert [item.case_id for item in results] == ["sma-3", "sma-3-repeat"]


def test_recreates_verified_wheel_snapshot_for_each_child_invocation(
    tmp_path: Path,
) -> None:
    replacement_source = """
import pandas as pd

def sma(frame, *, window):
    return pd.Series([None, None, 3.0], index=frame.index, name=f"sma_{window}")
"""
    replacement_wheel = _wheel_bytes(
        "baseline-provider",
        "baseline_provider",
        replacement_source,
    )
    encoded_replacement = base64.b64encode(replacement_wheel).decode("ascii")
    mutating_source = f"""
import base64
import json
import pathlib
import pandas as pd

def sma(frame, *, window):
    request_path = pathlib.Path(__file__).parents[4] / "request.json"
    request = json.loads(request_path.read_text())
    artifact = pathlib.Path(request["implementation_artifact"])
    artifact.chmod(0o600)
    artifact.write_bytes(base64.b64decode({encoded_replacement!r}))
    return pd.Series([None, None, 2.0], index=frame.index, name=f"sma_{{window}}")
"""
    candidate = _contract(tmp_path, mutating_source)

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


def test_large_verified_artifacts_are_snapshotted_with_bounded_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _contract(tmp_path)
    implementation, dependency = candidate.artifacts
    provider_bytes = _wheel_bytes(
        "baseline-provider",
        "baseline_provider",
        SUCCESS_SOURCE,
        prefix_data={"share/padding.bin": "x" * (6 * 1024 * 1024)},
    )
    implementation.wheel_path.write_bytes(provider_bytes)
    implementation = replace(
        implementation,
        digest=_sha256(provider_bytes),
    )
    del provider_bytes
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["implementation"]["implementation_digest"] = implementation.digest
    candidate = replace(
        candidate,
        artifacts=(implementation, dependency),
        operators=(replace(operator, manifest=manifest),),
    )
    artifact_paths = {
        implementation.wheel_path.resolve(),
        dependency.wheel_path.resolve(),
    }
    read_sizes: list[int] = []
    real_open = Path.open

    class BoundedReader:
        def __init__(self, file: object) -> None:
            self._file = file

        def __enter__(self) -> BoundedReader:
            return self

        def __exit__(self, *args: object) -> None:
            self._file.close()  # type: ignore[attr-defined]

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            assert 0 < size <= 1024 * 1024
            return cast(bytes, self._file.read(size))  # type: ignore[attr-defined]

    def bounded_open(path: Path, *args: object, **kwargs: object) -> object:
        file = real_open(path, *args, **kwargs)  # type: ignore[arg-type]
        if path.resolve() in artifact_paths and "r" in cast(str, args[0] if args else kwargs.get("mode", "r")):
            return BoundedReader(file)
        return file

    monkeypatch.setattr(Path, "open", bounded_open)

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]
    assert read_sizes


def test_rejects_nondeterministic_provider_output(tmp_path: Path) -> None:
    counter_path = tmp_path / "executions.txt"
    source = f"""
import pathlib
import pandas as pd

def sma(frame, *, window):
    path = pathlib.Path({str(counter_path)!r})
    count = int(path.read_text()) if path.exists() else 0
    path.write_text(str(count + 1))
    value = 2.0 if count == 0 else 3.0
    return pd.Series([None, None, value], index=frame.index, name=f"sma_{{window}}")
"""

    _assert_failure(_contract(tmp_path / "candidate", source), "baseline_mismatch")


def test_rejects_nondeterministic_callable_state_within_one_provider_process(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd

calls = 0

def sma(frame, *, window):
    global calls
    calls += 1
    value = 2.0 if calls == 1 else 3.0
    return pd.Series(
        [None, None, value],
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    _assert_failure(_contract(tmp_path / "candidate", source), "baseline_mismatch")


@pytest.mark.parametrize(
    "requirement",
    [
        "requires_benchmark",
        "requires_industry_data",
        "requires_market_cap_data",
        "requires_fundamental_data",
    ],
)
def test_rejects_unsupported_auxiliary_input_requirements(
    tmp_path: Path,
    requirement: str,
) -> None:
    candidate = _contract(tmp_path)
    operator = candidate.operators[0]
    manifest = json.loads(json.dumps(operator.manifest))
    manifest["input"][requirement] = True
    candidate = replace(
        candidate,
        operators=(replace(operator, manifest=manifest),),
    )

    _assert_failure(candidate, "baseline_input_invalid")


def test_applies_manifest_parameter_defaults_to_provider_call(tmp_path: Path) -> None:
    source = """
import pandas as pd
def sma(frame, *, window):
    return pd.Series([None, None, window - 1.0], index=frame.index, name=f"sma_{window}")
"""
    candidate = _contract(
        tmp_path,
        source,
        parameters={},
        expected={"sma_3": [None, None, 2.0]},
    )

    results = run_research_baselines(
        candidate,
        candidate.artifacts,
        timeout_seconds=5,
    )

    assert [(item.case_id, item.status) for item in results] == [("sma-3", "passed")]


@pytest.mark.parametrize("parameters", [{"unknown": 1}, {"window": 0}])
def test_rejects_invalid_parameters_before_child_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, object],
) -> None:
    candidate = _contract(tmp_path, parameters=parameters)

    def child_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"child executed: {args!r} {kwargs!r}")

    monkeypatch.setattr(baseline_runner.subprocess, "run", child_must_not_run)
    _assert_failure(candidate, "baseline_parameters_invalid")


def test_rejects_undeclared_expected_output_before_child_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _contract(
        tmp_path,
        expected={"not_declared": [None, None, 2.0]},
    )

    def child_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"child executed: {args!r} {kwargs!r}")

    monkeypatch.setattr(baseline_runner.subprocess, "run", child_must_not_run)
    _assert_failure(candidate, "baseline_input_invalid")


def _write_certifiable_provider(
    tmp_path: Path,
    *,
    expected: list[object],
    provider_source: str = SUCCESS_SOURCE,
    output_dtype: str = "float64",
) -> object:
    provider_bytes = _wheel_bytes("equant-ttr", "equant_ttr", provider_source)
    dependency_bytes = _wheel_bytes(
        "baseline-dependency",
        "baseline_dependency",
        DEPENDENCY_SOURCE,
    )
    provider_digest = _sha256(provider_bytes)
    dependency_digest = _sha256(dependency_bytes)
    dependency_name = "baseline_dependency-1.0.0-py3-none-any.whl"

    def mutate(repository: Path) -> None:
        rewrite_json(
            repository / COMPATIBILITY_ROOT / "candidate-build-v1.json",
            lambda build: (
                build["artifacts"][0].update({"digest": provider_digest}),  # type: ignore[index,union-attr]
                build["artifacts"].append(  # type: ignore[union-attr]
                    {
                        "distribution": "baseline-dependency",
                        "version": "1.0.0",
                        "filename": dependency_name,
                        "role": "runtime-dependency",
                        "build_identifier": "baseline-dependency-build-1",
                        "digest": dependency_digest,
                    }
                ),
            ),
        )

        def update_manifest(manifest: dict[str, object]) -> None:
            manifest["implementation"].update(  # type: ignore[union-attr]
                {"implementation_digest": provider_digest}
            )
            manifest["output"]["fields"][0]["dtype"] = output_dtype  # type: ignore[index]

        rewrite_json(
            repository / COMPATIBILITY_ROOT / "manifests" / "equant.ttr.sma.operator.json",
            update_manifest,
        )

        def replace_case(baseline: dict[str, object]) -> None:
            case = baseline["cases"][0]  # type: ignore[index]
            case["input"] = _panel()
            case["expected"] = {"sma_3": expected}

        rewrite_json(
            repository / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-v1.json",
            replace_case,
        )

    fixture = write_provider_repository(tmp_path, mutate=mutate)
    (fixture.artifact_dir / fixture.wheel_name).write_bytes(provider_bytes)
    (fixture.artifact_dir / dependency_name).write_bytes(dependency_bytes)
    return fixture


def _write_two_operator_provider(tmp_path: Path, executions_path: Path) -> object:
    provider_source = f"""
import pathlib
import pandas as pd
from baseline_dependency import rolling_mean

def _record(value):
    with pathlib.Path({str(executions_path)!r}).open("a") as stream:
        stream.write(value)

def sma(frame, *, window):
    _record("S")
    return pd.Series(
        rolling_mean(frame["close"].tolist(), window),
        index=frame.index,
        name=f"sma_{{window}}",
        dtype="float64",
    )

def zzz(frame, *, window):
    _record("Z")
    return pd.Series(
        [0.0, 0.0, 0.0],
        index=frame.index,
        name=f"zzz_{{window}}",
        dtype="float64",
    )
"""
    provider_bytes = _wheel_bytes("equant-ttr", "equant_ttr", provider_source)
    dependency_bytes = _wheel_bytes(
        "baseline-dependency",
        "baseline_dependency",
        DEPENDENCY_SOURCE,
    )
    provider_digest = _sha256(provider_bytes)
    dependency_digest = _sha256(dependency_bytes)
    dependency_name = "baseline_dependency-1.0.0-py3-none-any.whl"

    def mutate(repository: Path) -> None:
        rewrite_json(
            repository / COMPATIBILITY_ROOT / "candidate-build-v1.json",
            lambda build: (
                build["artifacts"][0].update({"digest": provider_digest}),  # type: ignore[index,union-attr]
                build["artifacts"].append(  # type: ignore[union-attr]
                    {
                        "distribution": "baseline-dependency",
                        "version": "1.0.0",
                        "filename": dependency_name,
                        "role": "runtime-dependency",
                        "build_identifier": "baseline-dependency-build-1",
                        "digest": dependency_digest,
                    }
                ),
            ),
        )
        sma_manifest_path = repository / COMPATIBILITY_ROOT / "manifests" / "equant.ttr.sma.operator.json"
        rewrite_json(
            sma_manifest_path,
            lambda manifest: manifest["implementation"].update(  # type: ignore[union-attr]
                {"implementation_digest": provider_digest}
            ),
        )
        sma_manifest = json.loads(sma_manifest_path.read_text(encoding="utf-8"))
        zzz_manifest = json.loads(json.dumps(sma_manifest))
        zzz_manifest.update(
            {
                "operator_id": "equant.ttr.zzz",
                "semantic_name": "ZZZ",
                "callable": "zzz",
            }
        )
        zzz_manifest["output"]["fields"][0]["name_template"] = "zzz_{window}"
        zzz_manifest_path = repository / COMPATIBILITY_ROOT / "manifests" / "equant.ttr.zzz.operator.json"
        zzz_manifest_path.write_text(
            json.dumps(zzz_manifest, sort_keys=True),
            encoding="utf-8",
        )

        def replace_sma_case(baseline: dict[str, object]) -> None:
            case = baseline["cases"][0]  # type: ignore[index]
            case["input"] = _panel()
            case["expected"] = {"sma_3": [None, None, 2.0]}

        sma_baseline_path = repository / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-v1.json"
        rewrite_json(sma_baseline_path, replace_sma_case)
        zzz_baseline = json.loads(sma_baseline_path.read_text(encoding="utf-8"))
        zzz_case = zzz_baseline["cases"][0]
        zzz_case.update(
            {
                "operator_id": "equant.ttr.zzz",
                "expected": {"zzz_3": [None, None, 2.0]},
            }
        )
        zzz_baseline_path = repository / COMPATIBILITY_ROOT / "numerical_baselines" / "technical-zzz-v1.json"
        zzz_baseline_path.write_text(
            json.dumps(zzz_baseline, sort_keys=True),
            encoding="utf-8",
        )
        rewrite_json(
            repository / COMPATIBILITY_ROOT / CATALOG_NAME,
            lambda catalog: catalog["operators"].update(  # type: ignore[union-attr]
                {
                    "equant.ttr.zzz@1.0.0": {
                        "manifest": "manifests/equant.ttr.zzz.operator.json",
                        "baseline": "numerical_baselines/technical-zzz-v1.json",
                    }
                }
            ),
        )

    fixture = write_provider_repository(tmp_path, mutate=mutate)
    (fixture.artifact_dir / fixture.wheel_name).write_bytes(provider_bytes)
    (fixture.artifact_dir / dependency_name).write_bytes(dependency_bytes)
    return fixture


def test_certify_provider_promotes_only_revalidated_passing_bindings(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(tmp_path, expected=[None, None, 2.0])
    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        result = certify_provider(submission)

    assert [item.binding["certification_state"] for item in result.operators] == ["research-certified"]
    assert [(item.case_id, item.status) for item in result.baseline_results] == [("sma-window-3", "passed")]

    with pytest.raises(TypeError):
        result.operators[0].binding["certification_state"] = "contract-valid"  # type: ignore[index]
    implementation = cast(
        dict[str, object],
        result.operators[0].manifest["implementation"],
    )
    with pytest.raises(TypeError):
        implementation["package_version"] = "9.9.9"
    with pytest.raises(TypeError):
        result.baseline_cases[0].parameters["window"] = 99  # type: ignore[index]
    expected = cast(dict[str, object], result.baseline_cases[0].expected)
    with pytest.raises(TypeError):
        expected["sma_3"] = [2.0]
    expected_values = cast(tuple[object, ...], result.baseline_cases[0].expected["sma_3"])
    with pytest.raises(TypeError):
        expected_values[0] = 0.0  # type: ignore[index]
    records = cast(
        tuple[dict[str, object], ...],
        result.baseline_cases[0].input["records"],
    )
    with pytest.raises(TypeError):
        records[0]["close"] = 99.0


def test_late_case_failure_does_not_mutate_contract_valid_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions_path = tmp_path / "executions.txt"
    source = SUCCESS_SOURCE.replace(
        "def sma(frame, *, window):",
        "import pathlib\n"
        "def sma(frame, *, window):\n"
        f"    with pathlib.Path({str(executions_path)!r}).open('a') as stream:\n"
        "        stream.write('x')",
    )
    contract = _contract(tmp_path, source)
    passing = contract.baseline_cases[0]
    failing = replace(passing, expected={"sma_3": [None, None, 99.0]})
    contract = replace(contract, baseline_cases=(passing, failing))
    monkeypatch.setattr(
        certification,
        "validate_provider_contract",
        lambda submission: contract,
    )

    with pytest.raises(OperatorCertificationError) as caught:
        certify_provider(cast(ProviderSubmission, object()))

    assert caught.value.code == "baseline_mismatch"
    assert executions_path.read_text(encoding="utf-8") == "xxxx"
    assert contract.operators[0].binding["certification_state"] == "contract-valid"
    assert contract.operators[0].manifest["operator_version"] == "1.0.0"


def test_second_operator_failure_in_one_submission_preserves_all_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions_path = tmp_path / "operator-executions.txt"
    fixture = _write_two_operator_provider(tmp_path, executions_path)
    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        contract = validate_provider_contract(submission)
        assert [item.manifest["operator_id"] for item in contract.operators] == [
            "equant.ttr.sma",
            "equant.ttr.zzz",
        ]
        monkeypatch.setattr(
            certification,
            "validate_provider_contract",
            lambda loaded: contract,
        )

        with pytest.raises(OperatorCertificationError) as caught:
            certify_provider(submission)

    assert caught.value.code == "baseline_mismatch"
    assert executions_path.read_text(encoding="utf-8") == "SSZZ"
    assert [item.binding["certification_state"] for item in contract.operators] == [
        "contract-valid",
        "contract-valid",
    ]


def test_certify_provider_never_returns_research_certified_after_any_failure(
    tmp_path: Path,
) -> None:
    fixture = _write_certifiable_provider(tmp_path, expected=[None, None, 99.0])
    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        with pytest.raises(OperatorCertificationError) as caught:
            certify_provider(submission)

    assert caught.value.code == "baseline_mismatch"


def test_certify_provider_normalizes_recursive_promotion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_certifiable_provider(
        tmp_path,
        expected=[None, None, 2.0],
    )

    def fail_recursive_freeze(value: Mapping[str, object]) -> Mapping[str, object]:
        del value
        raise RecursionError("manifest nesting exceeds recursive freezer")

    monkeypatch.setattr(
        certification,
        "_freeze_json_mapping",
        fail_recursive_freeze,
    )

    with load_provider_submission(
        fixture.path,
        fixture.submission_commit,
        fixture.artifact_dir,
    ) as submission:
        with pytest.raises(OperatorCertificationError) as caught:
            certify_provider(submission)

    assert caught.value.code == "binding_validation_failed"
    assert caught.value.stage == "binding"
