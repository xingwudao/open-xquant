"""Self-contained strict-JSON child for exact-wheel baseline execution."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import json
import math
import os
import stat
import sys
import zipfile
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, cast

_PLATFORM_RUNTIME_ROOTS = {"numpy", "pandas"}
_OUTPUT_DTYPES = {"boolean", "int64", "float64", "string", "date", "datetime"}
_OUTPUT_ALIGNMENTS = {
    "preserve_input_order",
    "canonical_order",
    "explicit_keyed_output",
}
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


class _OutputTypeError(TypeError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate request key")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    del value
    raise ValueError("non-standard JSON number")


def _read_request(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_bytes().decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonstandard_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("request is not an object")
    if set(value) != {
        "implementation_artifact",
        "dependency_artifacts",
        "module",
        "callable",
        "parameters",
        "input",
        "output_field",
        "output_dtype",
        "output_alignment",
    }:
        raise ValueError("request fields are invalid")
    return value


def _write_response(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(_encode_response(value))


def _encode_response(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _error(path: Path, code: str) -> int:
    _write_response(path, {"status": "error", "code": code})
    return 0


def _json_value(value: object, output_dtype: str) -> object:
    import pandas as pd

    missing = pd.isna(value)  # type: ignore[call-overload]
    if isinstance(missing, bool) and missing:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if value is None:
        return value
    if output_dtype == "float64" and _valid_float64_scalar(value):
        return value
    if output_dtype == "int64" and _valid_int64_scalar(value):
        return value
    if output_dtype == "boolean" and type(value) is bool:
        return value
    if output_dtype == "string" and type(value) is str:
        return value
    if (
        output_dtype == "date"
        and isinstance(value, date)
        and not isinstance(
            value,
            datetime,
        )
    ):
        return value.isoformat()
    if output_dtype == "datetime" and isinstance(value, datetime):
        return value.isoformat()
    raise _OutputTypeError("provider output scalar does not match manifest dtype")


def _valid_float64_scalar(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(float(cast(int | float, value)))
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_int64_scalar(value: object) -> bool:
    return type(value) is int and _INT64_MIN <= value <= _INT64_MAX


def _extract_output(
    result: object,
    frame: Any,
    output_field: str,
    output_dtype: str,
    output_alignment: str,
) -> list[object]:
    import pandas as pd

    input_keys = _frame_keys(frame)
    if isinstance(result, pd.Series):
        if result.name != output_field:
            raise KeyError("declared output field is missing")
        output_keys = _index_keys(result.index)
        values = [_json_value(value, output_dtype) for value in result.tolist()]
        return _align_output(values, output_keys, input_keys, output_alignment)
    if isinstance(result, pd.DataFrame):
        if output_field not in result.columns:
            raise KeyError("declared output field is missing")
        key_columns = {"date", "code"}.intersection(result.columns)
        if key_columns and key_columns != {"date", "code"}:
            raise IndexError("partial key alignment cannot be proven")
        if key_columns:
            output_keys = _frame_keys(result)
        else:
            output_keys = _index_keys(result.index)
        values = [_json_value(value, output_dtype) for value in result[output_field].tolist()]
        return _align_output(values, output_keys, input_keys, output_alignment)
    raise TypeError("provider output must be a Series or DataFrame")


def _frame_keys(frame: Any) -> list[tuple[object, object]]:
    return [(record["date"], record["code"]) for record in frame[["date", "code"]].to_dict("records")]


def _index_keys(index: Any) -> list[tuple[object, object]]:
    keys = list(index.tolist())
    if not all(isinstance(key, tuple) and len(key) == 2 for key in keys):
        raise IndexError("output index does not identify panel keys")
    return [cast(tuple[object, object], key) for key in keys]


def _align_output(
    values: list[object],
    output_keys: list[tuple[object, object]],
    input_keys: list[tuple[object, object]],
    output_alignment: str,
) -> list[object]:
    if (
        len(values) != len(input_keys)
        or len(output_keys) != len(input_keys)
        or len(set(output_keys)) != len(output_keys)
        or set(output_keys) != set(input_keys)
    ):
        raise IndexError("output keys do not match input keys")
    if output_alignment == "preserve_input_order":
        if output_keys != input_keys:
            raise IndexError("output does not preserve input order")
        return values
    if output_alignment == "canonical_order":
        if output_keys != sorted(input_keys):
            raise IndexError("output is not in canonical order")
        return values
    if output_alignment == "explicit_keyed_output":
        by_key = dict(zip(output_keys, values, strict=True))
        return [by_key[key] for key in input_keys]
    raise ValueError("output alignment is unsupported")


def _module_locations(module: ModuleType) -> tuple[str, ...]:
    locations: list[str] = []
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        locations.append(origin)
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        locations.append(module_file)
    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations is not None:
        locations.extend(location for location in search_locations if isinstance(location, str))
    return tuple(dict.fromkeys(locations))


def _location_is_in_archive(location: str, archive: str) -> bool:
    normalized_location = os.path.normcase(os.path.abspath(location))
    normalized_archive = os.path.normcase(os.path.abspath(archive))
    return normalized_location.startswith(normalized_archive + os.sep)


def _module_is_from_archives(
    module: ModuleType,
    archives: list[str],
) -> bool:
    locations = _module_locations(module)
    return bool(locations) and any(_location_is_in_archive(location, archive) for location in locations for archive in archives)


def _new_modules_are_allowed(
    modules_before_provider: set[str],
    verified_archives: list[str],
) -> bool:
    for name in set(sys.modules).difference(modules_before_provider):
        module = sys.modules.get(name)
        if not isinstance(module, ModuleType):
            return False
        root = name.partition(".")[0]
        if root in sys.stdlib_module_names or root in _PLATFORM_RUNTIME_ROOTS:
            continue
        if not _module_is_from_archives(module, verified_archives):
            return False
    return True


def _visible_modules(verified_roots: list[str]) -> Mapping[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name.partition(".")[0] in sys.stdlib_module_names
        or name.partition(".")[0] in _PLATFORM_RUNTIME_ROOTS
        or (isinstance(module, ModuleType) and _module_is_from_archives(module, verified_roots))
    }


def _restricted_sys_module(verified_roots: list[str]) -> ModuleType:
    restricted = ModuleType("sys")
    restricted.__dict__.update(sys.__dict__)
    restricted.modules = _visible_modules(verified_roots)  # type: ignore[attr-defined]
    return restricted


def _globals_are_from_archives(
    globals_value: Mapping[str, object] | None,
    archives: list[str],
) -> bool:
    if globals_value is None:
        return False
    locations: list[str] = []
    spec = globals_value.get("__spec__")
    origin = getattr(spec, "origin", None)
    if isinstance(origin, str) and origin not in {"built-in", "frozen"}:
        locations.append(origin)
    module_file = globals_value.get("__file__")
    if isinstance(module_file, str):
        locations.append(module_file)
    return any(_location_is_in_archive(location, archive) for location in locations for archive in archives)


class _ProviderImportGate:
    def __init__(self, verified_archives: list[str]) -> None:
        self._verified_archives = verified_archives
        self._original_import = builtins.__import__
        self._original_import_module = importlib.import_module
        self.violation = False

    def install(self) -> None:
        setattr(builtins, "__import__", self._guarded_import)
        setattr(importlib, "import_module", self._guarded_import_module)

    def restore(self) -> None:
        setattr(builtins, "__import__", self._original_import)
        setattr(importlib, "import_module", self._original_import_module)

    def _guarded_import(
        self,
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        provider_call = globals is None or _globals_are_from_archives(
            globals,
            self._verified_archives,
        )
        try:
            imported = self._original_import(name, globals, locals, fromlist, level)
        except BaseException:
            if provider_call:
                self.violation = True
            raise
        if not provider_call:
            return imported
        absolute_name = self._absolute_name(name, globals, level)
        if not self._provider_import_is_allowed(absolute_name):
            self.violation = True
            raise ImportError(f"provider import is outside the verified closure: {absolute_name}")
        if absolute_name == "sys":
            return _restricted_sys_module(self._verified_archives)
        return imported

    def _guarded_import_module(
        self,
        name: str,
        package: str | None = None,
    ) -> ModuleType:
        caller_globals = sys._getframe(1).f_globals
        provider_call = _globals_are_from_archives(
            caller_globals,
            self._verified_archives,
        )
        absolute_name = name
        if provider_call and name.startswith("."):
            if not isinstance(package, str) or not package:
                self.violation = True
                raise ImportError("provider relative import has no package")
            try:
                absolute_name = importlib.util.resolve_name(name, package)
            except (ImportError, ValueError) as error:
                self.violation = True
                raise ImportError("provider relative import is invalid") from error
        try:
            imported = self._original_import_module(name, package)
        except BaseException:
            if provider_call:
                self.violation = True
            raise
        if provider_call and not self._provider_import_is_allowed(absolute_name):
            self.violation = True
            raise ImportError(f"provider import is outside the verified closure: {absolute_name}")
        if provider_call and absolute_name == "sys":
            return _restricted_sys_module(self._verified_archives)
        return imported

    def _absolute_name(
        self,
        name: str,
        globals_value: Mapping[str, object] | None,
        level: int,
    ) -> str:
        if level == 0:
            return name
        package = None if globals_value is None else globals_value.get("__package__")
        if not isinstance(package, str) or not package:
            self.violation = True
            raise ImportError("provider relative import has no package")
        return importlib.util.resolve_name("." * level + name, package)

    def _provider_import_is_allowed(self, absolute_name: str) -> bool:
        root = absolute_name.partition(".")[0]
        if root in sys.stdlib_module_names or root in _PLATFORM_RUNTIME_ROOTS:
            return True
        module = sys.modules.get(absolute_name)
        if not isinstance(module, ModuleType):
            module = sys.modules.get(root)
        return isinstance(module, ModuleType) and _module_is_from_archives(
            module,
            self._verified_archives,
        )


def _provider_error(
    path: Path,
    import_gate: _ProviderImportGate,
    fallback_code: str,
) -> int:
    code = "provider_import_failed" if import_gate.violation else fallback_code
    return _error(path, code)


def _materialize_wheels(
    wheel_paths: list[str],
    root: Path,
) -> tuple[list[str], str]:
    root.mkdir()
    libraries_root = root / "site-packages"
    libraries_root.mkdir()
    prefix = root / "prefix"
    prefix.mkdir()
    roots: list[str] = []
    written: set[Path] = set()
    for index, wheel_path in enumerate(wheel_paths):
        destination = libraries_root / str(index)
        destination.mkdir()
        _extract_wheel(Path(wheel_path), destination, prefix, written)
        roots.append(str(destination))
    return roots, str(prefix)


def _extract_wheel(
    wheel_path: Path,
    library_destination: Path,
    prefix_destination: Path,
    written: set[Path],
) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        for member in wheel.infolist():
            mapped = _wheel_member_destination(member)
            if mapped is None:
                continue
            scheme, relative = mapped
            destination = library_destination if scheme == "library" else prefix_destination
            target = destination.joinpath(*relative.parts)
            if target in written:
                raise ValueError("wheel members map to the same destination")
            written.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wheel.read(member))
            mode = (member.external_attr >> 16) & 0o777
            target.chmod(mode or 0o644)


def _wheel_member_destination(
    member: zipfile.ZipInfo,
) -> tuple[str, PurePosixPath] | None:
    raw_name = member.filename
    path = PurePosixPath(raw_name)
    mode = member.external_attr >> 16
    if not raw_name or "\\" in raw_name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or stat.S_ISLNK(mode):
        raise ValueError("wheel member path is unsafe")
    if member.is_dir():
        return None
    parts = path.parts
    if parts[0].endswith(".data"):
        if len(parts) < 3:
            raise ValueError("wheel data member is invalid")
        scheme = parts[1]
        relative = parts[2:]
        if scheme in {"purelib", "platlib"}:
            return "library", PurePosixPath(*relative)
        if scheme == "data":
            return "prefix", PurePosixPath(*relative)
        if scheme == "scripts":
            return "prefix", PurePosixPath("bin", *relative)
        if scheme == "headers":
            return "prefix", PurePosixPath("include", *relative)
        raise ValueError("wheel data scheme is invalid")
    return "library", PurePosixPath(*parts)


def _hide_ambient_modules() -> None:
    for name, module in tuple(sys.modules.items()):
        root = name.partition(".")[0]
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if name == "__main__" or root in sys.stdlib_module_names or root in _PLATFORM_RUNTIME_ROOTS or origin in {"built-in", "frozen"}:
            continue
        sys.modules.pop(name, None)


def _frame_from_quant_panel(pd: Any, panel: dict[str, object]) -> Any:
    records = panel["records"]
    columns = panel["columns"]
    if not isinstance(records, list) or not isinstance(columns, list):
        raise TypeError("panel records or columns are invalid")
    frame = pd.DataFrame.from_records(records)
    nullable_dtypes = {
        "boolean": "boolean",
        "int64": "Int64",
        "float64": "Float64",
        "string": "string",
    }
    for descriptor in columns:
        if not isinstance(descriptor, dict):
            raise TypeError("panel column descriptor is invalid")
        name = descriptor.get("name")
        dtype = descriptor.get("dtype")
        if not isinstance(name, str) or not isinstance(dtype, str):
            raise TypeError("panel column descriptor is invalid")
        if name not in frame:
            frame[name] = pd.Series([None] * len(frame))
        if dtype in nullable_dtypes:
            raw_values = [record.get(name) if isinstance(record, dict) else None for record in records]
            frame[name] = pd.array(raw_values, dtype=nullable_dtypes[dtype])
    return frame


def _execute(request: dict[str, object], response_path: Path) -> int:
    implementation_artifact = request["implementation_artifact"]
    dependency_artifacts = request["dependency_artifacts"]
    module_name = request["module"]
    callable_name = request["callable"]
    parameters = request["parameters"]
    panel = request["input"]
    output_field = request["output_field"]
    output_dtype = request["output_dtype"]
    output_alignment = request["output_alignment"]
    if (
        not isinstance(implementation_artifact, str)
        or not Path(implementation_artifact).is_absolute()
        or not isinstance(dependency_artifacts, list)
        or not all(isinstance(path, str) and Path(path).is_absolute() for path in dependency_artifacts)
        or not isinstance(module_name, str)
        or not isinstance(callable_name, str)
        or not isinstance(parameters, dict)
        or not isinstance(panel, dict)
        or not isinstance(output_field, str)
        or not isinstance(output_dtype, str)
        or output_dtype not in _OUTPUT_DTYPES
        or not isinstance(output_alignment, str)
        or output_alignment not in _OUTPUT_ALIGNMENTS
    ):
        return _error(response_path, "provider_execution_failed")
    try:
        import numpy as np  # noqa: F401
        import pandas as pd
    except BaseException:
        return _error(response_path, "provider_execution_failed")

    try:
        verified_archives, install_prefix = _materialize_wheels(
            [implementation_artifact, *dependency_artifacts],
            response_path.parent / "wheel-layout",
        )
    except BaseException:
        return _error(response_path, "provider_import_failed")
    sys.prefix = install_prefix
    sys.exec_prefix = install_prefix
    _hide_ambient_modules()
    modules_before_provider = set(sys.modules)
    sys.path[:0] = verified_archives
    import_gate = _ProviderImportGate(verified_archives)
    import_gate.install()
    try:
        try:
            module = importlib.import_module(module_name)
            if not _module_is_from_archives(module, [verified_archives[0]]):
                raise ImportError("manifest module is not from implementation artifact")
            implementation = getattr(module, callable_name)
            if not callable(implementation):
                raise ImportError("manifest callable is not callable")
            if import_gate.violation or not _new_modules_are_allowed(
                modules_before_provider,
                verified_archives,
            ):
                raise ImportError("provider imported an undeclared ambient dependency")
        except BaseException:
            return _provider_error(
                response_path,
                import_gate,
                "provider_import_failed",
            )

        try:
            frame = _frame_from_quant_panel(pd, panel)
            context = panel.get("context")
            if not isinstance(context, dict):
                raise TypeError("panel context is not an object")
            frame.attrs["open_xquant_context"] = dict(context)
            frame.index = pd.MultiIndex.from_frame(
                frame[["date", "code"]],
                names=["__oxq_date", "__oxq_code"],
            )
            original = frame.copy(deep=True)
            result = implementation(frame, **parameters)
            try:
                pd.testing.assert_frame_equal(
                    frame,
                    original,
                    check_exact=True,
                    check_like=False,
                )
                if frame.attrs != original.attrs:
                    raise AssertionError("provider mutated frame metadata")
            except (AssertionError, TypeError, ValueError):
                return _provider_error(
                    response_path,
                    import_gate,
                    "provider_mutated_input",
                )
            try:
                output = _extract_output(
                    result,
                    original,
                    output_field,
                    output_dtype,
                    output_alignment,
                )
            except KeyError:
                return _provider_error(
                    response_path,
                    import_gate,
                    "baseline_mismatch",
                )
            except _OutputTypeError:
                return _provider_error(
                    response_path,
                    import_gate,
                    "baseline_mismatch",
                )
            except (IndexError, TypeError, ValueError):
                return _provider_error(
                    response_path,
                    import_gate,
                    "provider_alignment_failed",
                )
            response_bytes = _encode_response({"status": "ok", "output": output})
            if import_gate.violation or not _new_modules_are_allowed(
                modules_before_provider,
                verified_archives,
            ):
                return _error(response_path, "provider_import_failed")
        except BaseException:
            return _provider_error(
                response_path,
                import_gate,
                "provider_execution_failed",
            )
    finally:
        import_gate.restore()
    response_path.write_bytes(response_bytes)
    return 0


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    try:
        request = _read_request(request_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        return _error(response_path, "provider_execution_failed")
    return _execute(request, response_path)


if __name__ == "__main__":
    raise SystemExit(main())
