"""Security regression tests for the isolated baseline child."""

from __future__ import annotations

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
                "output_field": "sma_3",
                "output_dtype": "float64",
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
