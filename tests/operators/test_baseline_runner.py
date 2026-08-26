"""Exact-wheel numerical baseline execution and promotion tests."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

import oxq.operators.baseline_runner as baseline_runner
from oxq.operators.baseline_runner import run_research_baselines
from oxq.operators.certification import certify_provider
from oxq.operators.errors import OperatorCertificationError
from oxq.operators.models import (
    BaselineCase,
    BuildArtifact,
    ContractCandidate,
    ContractCertification,
)
from oxq.operators.submission import load_provider_submission
from tests.operators.helpers import rewrite_json, write_provider_repository


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
        submission_commit="c" * 40,
        source_commit="a" * 40,
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
            repository / "candidate-build-v1.json",
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
            repository / "manifests" / "equant.ttr.sma.operator.json",
            lambda manifest: manifest["implementation"].update(  # type: ignore[union-attr]
                {"implementation_digest": provider_digest}
            ),
        )

        def replace_case(baseline: dict[str, object]) -> None:
            case = baseline["cases"][0]  # type: ignore[index]
            case["input"] = _panel()
            case["expected"] = {"sma_3": expected}

        rewrite_json(
            repository / "numerical_baselines" / "technical-v1.json",
            replace_case,
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
        ("sma-3", "passed")
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
