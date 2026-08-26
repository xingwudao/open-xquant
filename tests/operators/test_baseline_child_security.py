"""Security regression tests for the isolated baseline child."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import cast

import pytest

from oxq.operators import _baseline_child


def _write_provider_wheel(path: Path, source: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("baseline_provider/__init__.py", source)
        archive.writestr(
            "baseline_provider-1.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "baseline_provider-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: baseline-provider\nVersion: 1.0.0\n",
        )


def _run_child(tmp_path: Path, provider_source: str) -> dict[str, object]:
    wheel_path = tmp_path / "baseline_provider-1.0.0-py3-none-any.whl"
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    _write_provider_wheel(wheel_path, provider_source)
    request_path.write_text(
        json.dumps(
            {
                "implementation_artifact": str(wheel_path),
                "dependency_artifacts": [],
                "module": "baseline_provider",
                "callable": "sma",
                "parameters": {"window": 3},
                "input": {
                    "columns": [{"name": "close", "dtype": "float64", "required": True}],
                    "context": {"timezone": "Asia/Shanghai"},
                    "records": [
                        {
                            "date": "2026-08-24",
                            "code": "000001.SZ",
                            "close": 1.0,
                        },
                        {
                            "date": "2026-08-25",
                            "code": "000001.SZ",
                            "close": 2.0,
                        },
                        {
                            "date": "2026-08-26",
                            "code": "000001.SZ",
                            "close": 3.0,
                        },
                    ],
                },
                "output_fields": [
                    {"name": "sma_3", "dtype": "float64"},
                ],
                "output_alignment": "preserve_input_order",
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    child_path = Path(_baseline_child.__file__).resolve()

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(child_path),
            str(request_path),
            str(response_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    return cast(dict[str, object], json.loads(response_path.read_text(encoding="utf-8")))


def test_dynamic_code_gate_allows_trusted_callers_and_restores_builtins() -> None:
    original_compile = builtins.compile
    original_exec = builtins.exec
    original_eval = builtins.eval
    gate = _baseline_child._ProviderImportGate([])

    gate.install()
    try:
        code = compile("40 + 2", "<trusted-test>", "eval")
        assert eval(code) == 42
        namespace: dict[str, object] = {}
        exec("value = 42", namespace)
        assert namespace["value"] == 42
    finally:
        gate.restore()

    assert builtins.compile is original_compile
    assert builtins.exec is original_exec
    assert builtins.eval is original_eval


def test_importlib_can_still_load_and_execute_verified_provider(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd

def sma(frame, *, window):
    return pd.Series(
        [None, None, 2.0],
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "ok",
        "outputs": {"sma_3": [None, None, 2.0]},
    }


def test_restricted_sys_path_does_not_alias_real_sys_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    original_path = list(sys.path)
    restricted = _baseline_child._restricted_sys_module([])
    restricted_path = cast(list[str], getattr(restricted, "path"))

    restricted_path.append("/provider-controlled")

    assert restricted_path != original_path
    assert sys.path == original_path


def test_provider_cannot_bypass_input_mutation_check_by_replacing_pandas_assertion(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd

def sma(frame, *, window):
    pd.testing.assert_frame_equal = lambda *args, **kwargs: None
    frame.loc[frame.index[0], "close"] = 99.0
    return pd.Series(
        [None, None, 2.0],
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_mutated_input",
    }


def _assert_provider_cannot_execute_ambient_source(
    tmp_path: Path,
    operation: str,
) -> None:
    ambient_root = tmp_path / "ambient" / "site-packages"
    ambient_root.mkdir(parents=True)
    ambient_source = ambient_root / "ambient_payload.py"
    if operation == "eval":
        ambient_source.write_text("2.0\n", encoding="utf-8")
        dynamic_execution = "value = eval(source)"
    elif operation == "exec":
        ambient_source.write_text("VALUE = 2.0\n", encoding="utf-8")
        dynamic_execution = "exec(source, namespace)\n    value = namespace['VALUE']"
    else:
        ambient_source.write_text("VALUE = 2.0\n", encoding="utf-8")
        dynamic_execution = "code = compile(source, str(ambient_path), 'exec')\n    exec(code, namespace)\n    value = namespace['VALUE']"
    source = f"""
import pathlib
import pandas as pd

def sma(frame, *, window):
    ambient_path = pathlib.Path({str(ambient_source)!r})
    source = ambient_path.read_text(encoding="utf-8")
    namespace = {{}}
    {dynamic_execution}
    return pd.Series(
        [None, None, value],
        index=frame.index,
        name=f"sma_{{window}}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_import_failed",
    }


def test_provider_cannot_compile_ambient_site_packages_source(
    tmp_path: Path,
) -> None:
    _assert_provider_cannot_execute_ambient_source(tmp_path, "compile")


def test_provider_cannot_exec_ambient_site_packages_source(tmp_path: Path) -> None:
    _assert_provider_cannot_execute_ambient_source(tmp_path, "exec")


def test_provider_cannot_eval_ambient_site_packages_source(tmp_path: Path) -> None:
    _assert_provider_cannot_execute_ambient_source(tmp_path, "eval")
