"""Exact-wheel numerical baseline execution and promotion tests."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import io
import json
import os
import sys
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import oxq.operators.baseline_runner as baseline_runner
import oxq.operators.certification as certification
from oxq.operators import _baseline_child
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


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _wheel_bytes(distribution: str, package: str, source: str) -> bytes:
    buffer = io.BytesIO()
    dist_info = distribution.replace("-", "_")
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{package}/__init__.py", source)
        archive.writestr(
            f"{dist_info}-1.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n",
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
        "columns": [
            {"name": "close", "dtype": "float64", "required": True}
        ],
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
    provider_path.write_bytes(
        _wheel_bytes("baseline-provider", "baseline_provider", provider_source)
    )
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
        expected={"sma_3": [None, None, 2.0]}
        if expected is None
        else expected,
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

    assert [(item.case_id, item.status) for item in results] == [
        ("sma-3", "passed")
    ]
    assert sys.path == before_path
    assert set(sys.modules).difference(before_modules).isdisjoint(
        {"baseline_provider", "baseline_dependency"}
    )
    assert not any(
        name == "baseline_provider" or name.startswith("baseline_provider.")
        for name in sys.modules
    )


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
    assert [(item.case_id, item.status) for item in float_results] == [
        ("sma-3", "passed")
    ]

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


def test_float_tolerance_does_not_relax_missing_value_positions(tmp_path: Path) -> None:
    source = SUCCESS_SOURCE.replace(
        "rolling_mean(frame[\"close\"].tolist(), window)",
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
        "import json, pathlib, sys\n"
        f"value = {{'status': 'ok', 'output': {invalid_dates!r}}}\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(value))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline_runner, "_child_script_path", lambda: child)

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

    _assert_failure(candidate, "baseline_mismatch")


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

    _assert_failure(_contract(tmp_path, source), "provider_import_failed")


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
        "import importlib\n"
        "def sma(frame, *, window):\n"
        f"    importlib.import_module({dependency!r})",
    )
    candidate = _contract(tmp_path, source)

    _assert_failure(candidate, "provider_import_failed")


@pytest.mark.parametrize(
    "violation_statement",
    ["import jsonschema", "importlib.import_module('dateutil')"],
    ids=["ordinary-import", "dynamic-import"],
)
@pytest.mark.parametrize("after_violation", ["mutation", "alignment"])
def test_import_violation_dominates_later_provider_result_failures(
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

    _assert_failure(_contract(tmp_path, source), "provider_import_failed")


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


def test_caught_unavailable_dynamic_import_remains_a_sticky_violation(
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
        "    frame.loc[frame.index[0], 'close'] = 99.0\n"
        "    return pd.Series([None, None, 2.0], index=frame.index, name=f'sma_{window}')\n"
    )

    _assert_failure(_contract(tmp_path, source), "provider_import_failed")


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
                "rolling_mean(frame[\"close\"].tolist(), window)",
                "[0.0, 0.0, 0.0]",
            ),
            "baseline_mismatch",
        ),
        (SUCCESS_SOURCE.replace('name=f"sma_{window}"', 'name="wrong"'), "baseline_mismatch"),
        (
            SUCCESS_SOURCE.replace(
                "def sma(frame, *, window):",
                "def sma(frame, *, window):\n    frame.loc[0, \"close\"] = 99.0",
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


def test_rejects_malformed_child_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "malformed_child.py"
    child.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[2]).write_bytes(b'{broken')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline_runner, "_child_script_path", lambda: child)

    _assert_failure(_contract(tmp_path / "candidate"), "provider_execution_failed")


def test_rejects_child_json_with_undeclared_response_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "extra_field_child.py"
    child.write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[2]).write_text("
        "json.dumps({'status': 'ok', 'output': [None, None, 2.0], 'extra': 1}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline_runner, "_child_script_path", lambda: child)

    _assert_failure(_contract(tmp_path / "candidate"), "provider_execution_failed")


def test_normalizes_child_timeout(tmp_path: Path) -> None:
    source = "import time\ndef sma(frame, *, window):\n    time.sleep(60)\n"
    _assert_failure(
        _contract(tmp_path, source),
        "provider_execution_timeout",
        timeout_seconds=0.05,
    )


def test_timeout_reaps_the_direct_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    source = (
        "import os, pathlib, time\n"
        "def sma(frame, *, window):\n"
        f"    pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
    )
    _assert_failure(
        _contract(tmp_path / "candidate", source),
        "provider_execution_timeout",
        timeout_seconds=2,
    )
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


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
    expected: list[float | None],
) -> object:
    provider_bytes = _wheel_bytes("equant-ttr", "equant_ttr", SUCCESS_SOURCE)
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
        rewrite_json(
            repository
            / COMPATIBILITY_ROOT
            / "manifests"
            / "equant.ttr.sma.operator.json",
            lambda manifest: manifest["implementation"].update(  # type: ignore[union-attr]
                {"implementation_digest": provider_digest}
            ),
        )

        def replace_case(baseline: dict[str, object]) -> None:
            case = baseline["cases"][0]  # type: ignore[index]
            case["input"] = _panel()
            case["expected"] = {"sma_3": expected}

        rewrite_json(
            repository
            / COMPATIBILITY_ROOT
            / "numerical_baselines"
            / "technical-v1.json",
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
        sma_manifest_path = (
            repository
            / COMPATIBILITY_ROOT
            / "manifests"
            / "equant.ttr.sma.operator.json"
        )
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
        zzz_manifest_path = (
            repository
            / COMPATIBILITY_ROOT
            / "manifests"
            / "equant.ttr.zzz.operator.json"
        )
        zzz_manifest_path.write_text(
            json.dumps(zzz_manifest, sort_keys=True),
            encoding="utf-8",
        )

        def replace_sma_case(baseline: dict[str, object]) -> None:
            case = baseline["cases"][0]  # type: ignore[index]
            case["input"] = _panel()
            case["expected"] = {"sma_3": [None, None, 2.0]}

        sma_baseline_path = (
            repository
            / COMPATIBILITY_ROOT
            / "numerical_baselines"
            / "technical-v1.json"
        )
        rewrite_json(sma_baseline_path, replace_sma_case)
        zzz_baseline = json.loads(sma_baseline_path.read_text(encoding="utf-8"))
        zzz_case = zzz_baseline["cases"][0]
        zzz_case.update(
            {
                "operator_id": "equant.ttr.zzz",
                "expected": {"zzz_3": [None, None, 2.0]},
            }
        )
        zzz_baseline_path = (
            repository
            / COMPATIBILITY_ROOT
            / "numerical_baselines"
            / "technical-zzz-v1.json"
        )
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

    assert [item.binding["certification_state"] for item in result.operators] == [
        "research-certified"
    ]
    assert [(item.case_id, item.status) for item in result.baseline_results] == [
        ("sma-window-3", "passed")
    ]

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
    assert executions_path.read_text(encoding="utf-8") == "xx"
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
    assert executions_path.read_text(encoding="utf-8") == "SZ"
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
