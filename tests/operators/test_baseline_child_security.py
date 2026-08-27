"""Security regression tests for the isolated baseline child."""

from __future__ import annotations

import builtins
import importlib.util
import marshal
import operator
import os
import sys
import types
import zipfile
from pathlib import Path
from typing import cast

import pytest

from oxq.operators import _exact_wheel_child as _baseline_child
from oxq.operators.runtime_protocol import run_exact_wheel_request


def _module_from_file(name: str, origin: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(origin)
    module.__spec__ = importlib.util.spec_from_file_location(name, origin)
    return module


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


def _write_dependency_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("baseline_dependency/__init__.py", "")
        archive.writestr("baseline_dependency/late.py", "VALUE = 1.0\n")
        archive.writestr(
            "baseline_dependency-1.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "baseline_dependency-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: baseline-dependency\nVersion: 1.0.0\n",
        )


def _run_child(
    tmp_path: Path,
    provider_source: str,
    *,
    with_dependency: bool = False,
) -> dict[str, object]:
    wheel_path = tmp_path / "baseline_provider-1.0.0-py3-none-any.whl"
    dependency_path = tmp_path / "baseline_dependency-1.0.0-py3-none-any.whl"
    _write_provider_wheel(wheel_path, provider_source)
    if with_dependency:
        _write_dependency_wheel(dependency_path)
    request = {
                "implementation_artifact": str(wheel_path),
                "dependency_artifacts": ([str(dependency_path)] if with_dependency else []),
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
            }
    return run_exact_wheel_request(
        request,
        [wheel_path, *([dependency_path] if with_dependency else [])],
        timeout_seconds=10,
    )


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


def test_verified_package_can_alias_static_runtime_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified_root = tmp_path / "verified"
    package_root = verified_root / "verified_pkg"
    package_root.mkdir(parents=True)
    package_init = package_root / "__init__.py"
    package_init.write_text("", encoding="utf-8")

    before = set(sys.modules)
    monkeypatch.setitem(
        sys.modules,
        "verified_pkg",
        _module_from_file("verified_pkg", package_init),
    )
    monkeypatch.setitem(
        sys.modules,
        "verified_pkg.operator",
        _module_from_file("verified_pkg.operator", Path(operator.__file__)),
    )

    assert _baseline_child._new_modules_are_allowed(
        before,
        [str(verified_root)],
    )


def test_provider_import_allows_static_runtime_namedtuple_eval(
    tmp_path: Path,
) -> None:
    source = """
from collections import namedtuple

import pandas as pd

VersionInfo = namedtuple("VersionInfo", ["major", "minor"])
VERSION = VersionInfo(1, 0)

def sma(frame, *, window):
    return pd.Series(
        [None, None, float(VERSION.major + VERSION.minor + 1)],
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "ok",
        "outputs": {"sma_3": [None, None, 2.0]},
        "repeated_outputs": {"sma_3": [None, None, 2.0]},
    }


def test_provider_import_allows_static_runtime_forward_ref_compile(
    tmp_path: Path,
) -> None:
    source = """
from typing import Dict, Type

import pandas as pd

class Node:
    children: Dict[type, Type["Node"]] = {}

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
        "repeated_outputs": {"sma_3": [None, None, 2.0]},
    }


def test_provider_import_allows_platform_runtime_lazy_compile(
    tmp_path: Path,
) -> None:
    source = """
import numpy as np
import pandas as pd

POLYNOMIAL = np.polynomial.polynomial.Polynomial

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
        "repeated_outputs": {"sma_3": [None, None, 2.0]},
    }


def test_provider_import_allows_static_runtime_dataclass_exec(
    tmp_path: Path,
) -> None:
    source = """
from dataclasses import make_dataclass

import pandas as pd

Point = make_dataclass("Point", [("x", float)])

def sma(frame, *, window):
    point = Point(2.0)
    return pd.Series(
        [None, None, point.x],
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "ok",
        "outputs": {"sma_3": [None, None, 2.0]},
        "repeated_outputs": {"sma_3": [None, None, 2.0]},
    }


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
        "repeated_outputs": {"sma_3": [None, None, 2.0]},
    }


def test_provider_can_import_unpreloaded_standard_library_module(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd

def sma(frame, *, window):
    from fractions import Fraction
    return pd.Series(
        [None, None, float(Fraction(2, 1))],
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "ok",
        "outputs": {"sma_3": [None, None, 2.0]},
        "repeated_outputs": {"sma_3": [None, None, 2.0]},
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


def test_restricted_sys_hides_process_arguments_and_frame_introspection() -> None:
    restricted = _baseline_child._restricted_sys_module([])

    assert not hasattr(restricted, "argv")
    assert not hasattr(restricted, "orig_argv")
    assert not hasattr(restricted, "_getframe")
    assert not hasattr(restricted, "_current_frames")


def test_verified_root_snapshot_streams_large_files_with_bounded_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "verified"
    root.mkdir()
    target = root / "large.bin"
    target.write_bytes(b"x" * (2 * 1024 * 1024))
    real_open = Path.open
    read_sizes: list[int] = []

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
        if path == target:
            return BoundedReader(file)
        return file

    monkeypatch.setattr(Path, "open", bounded_open)

    snapshot = _baseline_child._snapshot_verified_roots([str(root)])

    assert snapshot[f"0/{target.name}"].startswith("sha256:")
    assert read_sizes


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


def test_provider_cannot_bypass_input_snapshot_by_replacing_dataframe_copy(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd

pd.DataFrame.copy = lambda self, deep=True: self

def sma(frame, *, window):
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


@pytest.mark.parametrize("output_type", ["series", "dataframe"])
def test_provider_cannot_replace_pandas_output_types(
    tmp_path: Path,
    output_type: str,
) -> None:
    if output_type == "series":
        fake_output = """
class FakeSeries:
    def __init__(self, frame, window):
        self.name = f"sma_{window}"
        self.index = frame.index

    def tolist(self):
        return [None, None, 2.0]

def sma(frame, *, window):
    pd.Series = FakeSeries
    return FakeSeries(frame, window)
"""
    else:
        fake_output = """
class FakeColumn:
    def tolist(self):
        return [None, None, 2.0]

class FakeDataFrame:
    def __init__(self, frame, window):
        self.columns = [f"sma_{window}"]
        self.index = frame.index

    def __getitem__(self, name):
        return FakeColumn()

def sma(frame, *, window):
    pd.DataFrame = FakeDataFrame
    return FakeDataFrame(frame, window)
"""
    source = f"""
import pandas as pd

{fake_output}
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_alignment_failed",
    }


@pytest.mark.parametrize("output_type", ["series", "dataframe"])
def test_provider_cannot_forge_pandas_type_through_class_property(
    tmp_path: Path,
    output_type: str,
) -> None:
    if output_type == "series":
        fake_output = """
TrustedSeries = pd.Series

class FakeValues:
    def tolist(self):
        return [None, None, 2.0]

class FakeSeries:
    @property
    def __class__(self):
        return TrustedSeries

    def __init__(self, frame, window):
        self.name = f"sma_{window}"
        self.index = frame.index
        self._values = FakeValues()

    def tolist(self):
        return [None, None, 2.0]

def sma(frame, *, window):
    pd.Series = FakeSeries
    return FakeSeries(frame, window)
"""
    else:
        fake_output = """
TrustedDataFrame = pd.DataFrame

class FakeValues:
    def tolist(self):
        return [None, None, 2.0]

class FakeColumn:
    def __init__(self):
        self._values = FakeValues()

    def tolist(self):
        return [None, None, 2.0]

class FakeDataFrame:
    @property
    def __class__(self):
        return TrustedDataFrame

    def __init__(self, frame, window):
        self.columns = pd.Index([f"sma_{window}"])
        self.index = frame.index

    def _get_item_cache(self, name):
        return FakeColumn()

def sma(frame, *, window):
    pd.DataFrame = FakeDataFrame
    return FakeDataFrame(frame, window)
"""
    source = f"""
import pandas as pd

{fake_output}
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_alignment_failed",
    }


def test_provider_cannot_replace_pandas_scalar_conversion_helpers(
    tmp_path: Path,
) -> None:
    source = """
import pandas as pd

class FakeScalar:
    def item(self):
        return 2.0

def sma(frame, *, window):
    result = pd.Series(
        [None, None, FakeScalar()],
        index=frame.index,
        name=f"sma_{window}",
        dtype="object",
    )
    pd.isna = lambda value: False
    return result
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "baseline_mismatch",
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


def test_provider_cannot_execute_ambient_source_through_runpy(
    tmp_path: Path,
) -> None:
    ambient_root = tmp_path / "ambient" / "site-packages"
    ambient_root.mkdir(parents=True)
    ambient_source = ambient_root / "ambient_payload.py"
    ambient_source.write_text("VALUE = 2.0\n", encoding="utf-8")
    source = f"""
import runpy
import pandas as pd

def sma(frame, *, window):
    value = runpy.run_path({str(ambient_source)!r})["VALUE"]
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


def test_provider_cannot_execute_ambient_source_in_threaded_runpy(
    tmp_path: Path,
) -> None:
    ambient_root = tmp_path / "ambient" / "site-packages"
    ambient_root.mkdir(parents=True)
    ambient_source = ambient_root / "ambient_payload.py"
    ambient_source.write_text("VALUE = 2.0\n", encoding="utf-8")
    source = f"""
import runpy
import threading
import pandas as pd

def sma(frame, *, window):
    values = {{}}
    worker = threading.Thread(
        target=lambda: values.update(runpy.run_path({str(ambient_source)!r})),
    )
    worker.start()
    worker.join()
    return pd.Series(
        [None, None, values["VALUE"]],
        index=frame.index,
        name=f"sma_{{window}}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_import_failed",
    }


def test_provider_cannot_forge_verifier_response_through_sys_argv(
    tmp_path: Path,
) -> None:
    source = """
import json
import os
import pathlib
import sys

def sma(frame, *, window):
    forged = {
        "status": "ok",
        "outputs": {f"sma_{window}": [None, None, 2.0]},
    }
    pathlib.Path(sys.argv[2]).write_text(json.dumps(forged))
    os._exit(0)
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_execution_failed",
    }


def test_provider_cannot_find_verifier_response_through_frame_locals(
    tmp_path: Path,
) -> None:
    source = """
import inspect
import json
import os
import pathlib

def sma(frame, *, window):
    current = inspect.currentframe()
    while current is not None:
        response_path = current.f_locals.get("response_path")
        if isinstance(response_path, pathlib.Path):
            forged = {
                "status": "ok",
                "outputs": {f"sma_{window}": [None, None, 2.0]},
            }
            response_path.write_text(json.dumps(forged))
            os._exit(0)
        current = current.f_back
    raise RuntimeError("response path was not visible")
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_execution_failed",
    }


def test_provider_traceback_cannot_reach_verifier_response_state(
    tmp_path: Path,
) -> None:
    source = """
import hashlib
import hmac
import json
import os
import pathlib

def sma(frame, *, window):
    try:
        raise RuntimeError("capture provider traceback")
    except RuntimeError as error:
        current = error.__traceback__.tb_frame
    response_path = None
    response_secret = None
    while current is not None:
        candidate_path = current.f_locals.get("response_path")
        candidate_secret = current.f_locals.get("response_secret")
        if isinstance(candidate_path, pathlib.Path):
            response_path = candidate_path
        if isinstance(candidate_secret, bytes) and len(candidate_secret) == 32:
            response_secret = candidate_secret
        current = current.f_back
    if response_path is None or response_secret is None:
        raise RuntimeError("verifier response state was not visible")
    forged = {
        "status": "ok",
        "outputs": {f"sma_{window}": [None, None, 2.0]},
        "repeated_outputs": {f"sma_{window}": [None, None, 2.0]},
    }
    payload = json.dumps(
        forged,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    forged["auth"] = "hmac-sha256:" + hmac.new(
        response_secret,
        payload,
        hashlib.sha256,
    ).hexdigest()
    response_path.write_text(json.dumps(forged), encoding="utf-8")
    os._exit(0)
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_execution_failed",
    }


@pytest.mark.parametrize(
    "operation",
    [
        "subprocess-run",
        "subprocess-popen",
        "os-system",
        "os-popen",
        "os-fork",
        "os-posix-spawn",
        "os-spawnv",
    ],
)
def test_provider_cannot_create_processes(
    tmp_path: Path,
    operation: str,
) -> None:
    required_os_function = {
        "os-fork": "fork",
        "os-posix-spawn": "posix_spawn",
        "os-spawnv": "spawnv",
    }.get(operation)
    if required_os_function is not None and not hasattr(os, required_os_function):
        pytest.skip(f"os.{required_os_function} is unavailable")
    if operation == "subprocess-run":
        process_creation = (
            "value = float(subprocess.run([sys.executable, '-c', 'print(2.0)'], check=True, capture_output=True, text=True).stdout)"
        )
    elif operation == "subprocess-popen":
        process_creation = (
            "value = float(subprocess.Popen([sys.executable, '-c', 'print(2.0)'], stdout=subprocess.PIPE, text=True).communicate()[0])"
        )
    elif operation == "os-system":
        process_creation = "value = 2.0 if os.system(sys.executable + \" -c 'raise SystemExit(0)'\") == 0 else 99.0"
    elif operation == "os-popen":
        process_creation = "value = float(os.popen(sys.executable + \" -c 'print(2.0)'\").read())"
    elif operation == "os-fork":
        process_creation = "pid = os.fork()\n    if pid == 0:\n        os._exit(0)\n    os.waitpid(pid, 0)\n    value = 2.0"
    elif operation == "os-posix-spawn":
        process_creation = (
            "pid = os.posix_spawn(sys.executable, [sys.executable, '-c', 'pass'], os.environ)\n    os.waitpid(pid, 0)\n    value = 2.0"
        )
    else:
        process_creation = (
            "pid = os.spawnv(os.P_NOWAIT, sys.executable, [sys.executable, '-c', 'pass'])\n    os.waitpid(pid, 0)\n    value = 2.0"
        )
    source = f"""
import os
import subprocess
import sys
import pandas as pd

def sma(frame, *, window):
    {process_creation}
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


@pytest.mark.skipif(sys.platform != "linux", reason="Linux seccomp assertion")
def test_linux_filter_blocks_native_fork_during_provider_import(
    tmp_path: Path,
) -> None:
    source = """
import ctypes
import os
import pandas as pd

libc = ctypes.CDLL(None, use_errno=True)
pid = libc.fork()
if pid < 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
if pid == 0:
    os._exit(0)
os.waitpid(pid, 0)

def sma(frame, *, window):
    return pd.Series(
        [None, None, 2.0],
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_import_failed",
    }


@pytest.mark.parametrize(
    "filename_expression",
    ["__file__", "fractions.__file__"],
    ids=["provider-file", "stdlib-file"],
)
def test_provider_cannot_launder_ambient_code_filename(
    tmp_path: Path,
    filename_expression: str,
) -> None:
    ambient_code_path = tmp_path / "ambient-code.bin"
    ambient_code_path.write_bytes(
        marshal.dumps(
            compile(
                "VALUE = 2.0\n",
                str(tmp_path / "ambient.py"),
                "exec",
            )
        )
    )
    source = f"""
import fractions
import marshal
import pathlib
import pandas as pd

def sma(frame, *, window):
    ambient_code = marshal.loads(pathlib.Path({str(ambient_code_path)!r}).read_bytes())
    laundered_code = ambient_code.replace(co_filename={filename_expression})
    namespace = {{}}
    exec(laundered_code, namespace)
    return pd.Series(
        [None, None, namespace["VALUE"]],
        index=frame.index,
        name=f"sma_{{window}}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source) == {
        "status": "error",
        "code": "provider_import_failed",
    }


def test_provider_cannot_modify_unimported_verified_dependency_module(
    tmp_path: Path,
) -> None:
    source = """
import pathlib
import pandas as pd
import baseline_dependency

def sma(frame, *, window):
    dependency_root = pathlib.Path(baseline_dependency.__file__).parent
    (dependency_root / "late.py").write_text("VALUE = 2.0\\n")
    from baseline_dependency.late import VALUE
    return pd.Series(
        [None, None, VALUE],
        index=frame.index,
        name=f"sma_{window}",
        dtype="float64",
    )
"""

    assert _run_child(tmp_path, source, with_dependency=True) == {
        "status": "error",
        "code": "provider_import_failed",
    }
