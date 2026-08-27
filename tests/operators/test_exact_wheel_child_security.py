"""Security regressions for the stdlib-first exact-wheel child."""

from __future__ import annotations

import importlib.metadata
import zipfile
from pathlib import Path

from oxq.operators.runtime_protocol import run_exact_wheel_request


def _snapshot_installed_distribution(tmp_path: Path, name: str) -> Path:
    distribution = importlib.metadata.distribution(name)
    installed_root = Path(distribution.locate_file("")).resolve(strict=True)
    wheel_path = tmp_path / f"{name}-fixture.whl"
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for entry in distribution.files or ():
            source = Path(distribution.locate_file(entry))
            relative = source.relative_to(installed_root)
            if ".." not in relative.parts and source.is_file():
                wheel.write(source, relative.as_posix())
    return wheel_path


def _write_provider_wheel(path: Path) -> None:
    source = """
import os

import numpy as np
import pandas as pd

def runtime_origins(frame):
    closure_root = os.path.commonpath([np.__file__, pd.__file__])
    return pd.DataFrame(
        {
            "closure_root": [closure_root] * len(frame),
            "numpy_origin": [np.__file__] * len(frame),
            "pandas_origin": [pd.__file__] * len(frame),
        },
        index=frame.index,
    )
"""
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("origin_provider/__init__.py", source)


def test_child_imports_platform_runtime_only_from_exact_closure(tmp_path: Path) -> None:
    ambient_root = tmp_path / "poisoned-ambient" / "site-packages"
    for package in ("numpy", "pandas"):
        package_root = ambient_root / package
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text(
            f'raise RuntimeError("loaded poisoned ambient {package}")\n',
            encoding="utf-8",
        )

    provider_wheel = tmp_path / "origin-provider.whl"
    _write_provider_wheel(provider_wheel)
    runtime_wheels = [
        _snapshot_installed_distribution(tmp_path, name)
        for name in ("numpy", "pandas", "python-dateutil", "pytz", "six")
    ]
    response = run_exact_wheel_request(
        {
            "module": "origin_provider",
            "callable": "runtime_origins",
            "parameters": {},
            "input": {
                "columns": [
                    {"name": "close", "dtype": "float64", "required": True},
                ],
                "context": {"timezone": "Asia/Shanghai"},
                "records": [
                    {
                        "date": "2026-08-27",
                        "code": "000001.SZ",
                        "close": 1.0,
                    },
                ],
            },
            "output_fields": [
                {"name": "closure_root", "dtype": "string"},
                {"name": "numpy_origin", "dtype": "string"},
                {"name": "pandas_origin", "dtype": "string"},
            ],
            "output_alignment": "preserve_input_order",
        },
        [provider_wheel, *runtime_wheels],
        timeout_seconds=60,
        _test_runtime_paths=[ambient_root],
    )

    assert response["status"] == "ok"
    outputs = response["outputs"]
    assert isinstance(outputs, dict)
    closure_root = Path(outputs["closure_root"][0])
    numpy_origin = Path(outputs["numpy_origin"][0])
    pandas_origin = Path(outputs["pandas_origin"][0])
    assert numpy_origin.is_relative_to(closure_root)
    assert pandas_origin.is_relative_to(closure_root)
    assert not numpy_origin.is_relative_to(ambient_root)
    assert not pandas_origin.is_relative_to(ambient_root)


def test_child_preserves_platform_runtime_from_exact_closure_without_test_paths(
    tmp_path: Path,
) -> None:
    provider_wheel = tmp_path / "origin-provider.whl"
    _write_provider_wheel(provider_wheel)
    runtime_wheels = [
        _snapshot_installed_distribution(tmp_path, name)
        for name in ("numpy", "pandas", "python-dateutil", "pytz", "six")
    ]
    response = run_exact_wheel_request(
        {
            "module": "origin_provider",
            "callable": "runtime_origins",
            "parameters": {},
            "input": {
                "columns": [
                    {"name": "close", "dtype": "float64", "required": True},
                ],
                "context": {"timezone": "Asia/Shanghai"},
                "records": [
                    {
                        "date": "2026-08-27",
                        "code": "000001.SZ",
                        "close": 1.0,
                    },
                ],
            },
            "output_fields": [
                {"name": "closure_root", "dtype": "string"},
                {"name": "numpy_origin", "dtype": "string"},
                {"name": "pandas_origin", "dtype": "string"},
            ],
            "output_alignment": "preserve_input_order",
        },
        [provider_wheel, *runtime_wheels],
        timeout_seconds=60,
    )

    assert response["status"] == "ok"
    outputs = response["outputs"]
    assert isinstance(outputs, dict)
    closure_root = Path(outputs["closure_root"][0])
    assert Path(outputs["numpy_origin"][0]).is_relative_to(closure_root)
    assert Path(outputs["pandas_origin"][0]).is_relative_to(closure_root)
