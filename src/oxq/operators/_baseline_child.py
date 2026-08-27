"""Self-contained strict-JSON child for exact-wheel baseline execution."""

from __future__ import annotations

import _thread
import builtins
import hashlib
import hmac
import importlib
import importlib.util
import json
import math
import os
import platform
import stat
import subprocess
import sys
import sysconfig
import threading
import time
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from types import CodeType, ModuleType
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
_FILE_HASH_CHUNK_BYTES = 1024 * 1024
_PROVIDER_POLICY_VIOLATION_EXIT_CODE = 86
_STATIC_RUNTIME_SOURCE_ROOTS = tuple(
    dict.fromkeys(
        root
        for root in (
            sysconfig.get_path("stdlib"),
            sysconfig.get_path("platstdlib"),
        )
        if isinstance(root, str) and root
    )
)
_HIDDEN_PROVIDER_SYS_ATTRIBUTES = {
    "_current_frames",
    "_getframe",
    "argv",
    "orig_argv",
}


class _OutputTypeError(TypeError):
    pass


@dataclass(frozen=True)
class _PandasValidationPrimitives:
    series_type: type
    dataframe_type: type
    isna: Callable[[object], object]
    numpy_scalar_type: type
    numpy_scalar_item: Callable[[object], object]
    series_tolist: Callable[[Any], list[object]]
    dataframe_getitem: Callable[[Any, object], Any]


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
        "output_fields",
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


def _authenticated_response(
    value: dict[str, object],
    secret: bytes,
) -> dict[str, object]:
    digest = hmac.new(secret, _encode_response(value), hashlib.sha256).hexdigest()
    return {**value, "auth": f"hmac-sha256:{digest}"}


def _json_value(
    value: object,
    output_dtype: str,
    primitives: _PandasValidationPrimitives,
) -> object:
    missing = primitives.isna(value)
    if isinstance(missing, bool) and missing:
        return None
    if issubclass(type(value), primitives.numpy_scalar_type):
        value = primitives.numpy_scalar_item(value)
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
        converted = float(cast(int | float, value))
        return math.isfinite(converted) and (type(value) is float or int(converted) == value)
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_int64_scalar(value: object) -> bool:
    return type(value) is int and _INT64_MIN <= value <= _INT64_MAX


def _trusted_series_values(
    series: Any,
    primitives: _PandasValidationPrimitives,
) -> list[object]:
    trusted_values = primitives.series_tolist(series)
    current_tolist = getattr(type(series), "tolist", None)
    if current_tolist is primitives.series_tolist:
        return trusted_values
    observed_values = series.tolist()
    if type(observed_values) is not list or observed_values != trusted_values:
        raise TypeError("provider output scalar extraction is not trustworthy")
    return trusted_values


def _extract_output(
    result: object,
    frame: Any,
    output_field: str,
    output_dtype: str,
    output_alignment: str,
    primitives: _PandasValidationPrimitives,
) -> list[object]:
    input_keys = _frame_keys(frame)
    result_type = type(result)
    if issubclass(result_type, primitives.series_type):
        series = cast(Any, result)
        if series.name != output_field:
            raise KeyError("declared output field is missing")
        output_keys = _index_keys(series.index)
        values = [_json_value(value, output_dtype, primitives) for value in _trusted_series_values(series, primitives)]
        return _align_output(values, output_keys, input_keys, output_alignment)
    if issubclass(result_type, primitives.dataframe_type):
        output_frame = cast(Any, result)
        if output_field not in output_frame.columns:
            raise KeyError("declared output field is missing")
        key_columns = {"date", "code"}.intersection(output_frame.columns)
        if key_columns and key_columns != {"date", "code"}:
            raise IndexError("partial key alignment cannot be proven")
        if key_columns:
            output_keys = _frame_keys(output_frame)
        else:
            output_keys = _index_keys(output_frame.index)
        output_series = primitives.dataframe_getitem(output_frame, output_field)
        values = [_json_value(value, output_dtype, primitives) for value in _trusted_series_values(output_series, primitives)]
        return _align_output(values, output_keys, input_keys, output_alignment)
    raise TypeError("provider output must be a Series or DataFrame")


def _extract_outputs(
    result: object,
    frame: Any,
    output_fields: list[tuple[str, str]],
    output_alignment: str,
    primitives: _PandasValidationPrimitives,
) -> dict[str, list[object]]:
    return {
        output_field: _extract_output(
            result,
            frame,
            output_field,
            output_dtype,
            output_alignment,
            primitives,
        )
        for output_field, output_dtype in output_fields
    }


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
    normalized_location = os.path.normcase(os.path.realpath(location))
    normalized_archive = os.path.normcase(os.path.realpath(archive))
    return normalized_location.startswith(normalized_archive + os.sep)


def _location_is_in_static_runtime_source(location: str) -> bool:
    if not location or any(part in {"site-packages", "dist-packages"} for part in Path(location).parts):
        return False
    return any(_location_is_in_archive(location, root) for root in _STATIC_RUNTIME_SOURCE_ROOTS)


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


def _restrict_sys_path(verified_archives: list[str]) -> None:
    runtime_paths = [
        entry
        for entry in sys.path
        if entry
        and not {"site-packages", "dist-packages"}.intersection(
            Path(entry).parts,
        )
    ]
    sys.path[:] = [*verified_archives, *runtime_paths]


def _restricted_sys_module(verified_roots: list[str]) -> ModuleType:
    restricted = ModuleType("sys")
    restricted.__dict__.update({name: value for name, value in sys.__dict__.items() if name not in _HIDDEN_PROVIDER_SYS_ATTRIBUTES})
    restricted.path = list(sys.path)  # type: ignore[attr-defined]
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


def _snapshot_verified_roots(verified_roots: list[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for root_index, root_value in enumerate(verified_roots):
        root = Path(root_value)
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            key = f"{root_index}/{relative}"
            if path.is_symlink():
                raise OSError("verified root contains a symbolic link")
            if path.is_dir():
                snapshot[key] = "directory"
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(_FILE_HASH_CHUNK_BYTES):
                        digest.update(chunk)
                snapshot[key] = f"sha256:{digest.hexdigest()}"
            else:
                raise OSError("verified root contains an unsupported entry")
    return snapshot


def _verified_roots_are_unchanged(
    verified_roots: list[str],
    expected: Mapping[str, str],
) -> bool:
    try:
        return _snapshot_verified_roots(verified_roots) == expected
    except OSError:
        return False


class _ProviderImportGate:
    def __init__(
        self,
        verified_archives: list[str],
        verified_root_snapshot: Mapping[str, str] | None = None,
    ) -> None:
        self._verified_archives = verified_archives
        self._verified_root_snapshot = dict(verified_root_snapshot or {})
        self._original_import = builtins.__import__
        self._original_import_module = importlib.import_module
        self._original_compile: Callable[..., object] = builtins.compile
        self._original_exec: Callable[..., None] = builtins.exec
        self._original_eval: Callable[..., object] = builtins.eval
        self._original_getframe = sys._getframe
        self._original_thread_start = threading.Thread.start
        self._original_start_new_thread = _thread.start_new_thread
        self._trusted_code_objects: dict[int, CodeType] = {}
        self._process_entry_points = self._capture_process_entry_points()
        self._hidden_sys_attributes: dict[str, object] = {}
        self.violation = False

    def install(self) -> None:
        setattr(builtins, "__import__", self._guarded_import)
        setattr(builtins, "compile", self._guarded_compile)
        setattr(builtins, "exec", self._guarded_exec)
        setattr(builtins, "eval", self._guarded_eval)
        setattr(importlib, "import_module", self._guarded_import_module)
        setattr(threading.Thread, "start", self._guarded_thread_start)
        setattr(_thread, "start_new_thread", self._guarded_start_new_thread)
        for owner, name, original in self._process_entry_points:
            setattr(owner, name, self._process_guard(name, original))
        for name in _HIDDEN_PROVIDER_SYS_ATTRIBUTES:
            if name in sys.__dict__:
                self._hidden_sys_attributes[name] = sys.__dict__.pop(name)

    def restore(self) -> None:
        setattr(builtins, "__import__", self._original_import)
        setattr(builtins, "compile", self._original_compile)
        setattr(builtins, "exec", self._original_exec)
        setattr(builtins, "eval", self._original_eval)
        setattr(importlib, "import_module", self._original_import_module)
        setattr(threading.Thread, "start", self._original_thread_start)
        setattr(_thread, "start_new_thread", self._original_start_new_thread)
        for owner, name, original in self._process_entry_points:
            setattr(owner, name, original)
        sys.__dict__.update(self._hidden_sys_attributes)

    def _guarded_compile(self, *args: object, **kwargs: object) -> object:
        if not self._compile_matches_verified_source(args):
            self._reject_provider_dynamic_code("compile")
        compiled = self._original_compile(*args, **kwargs)
        if isinstance(compiled, CodeType) and self._compile_matches_verified_source(args):
            self._trust_code_graph(compiled)
        return compiled

    def _guarded_exec(self, *args: object, **kwargs: object) -> None:
        if not self._code_is_from_verified_source(args):
            self._reject_provider_dynamic_code("exec")
        self._original_exec(*args, **kwargs)

    def _guarded_eval(self, *args: object, **kwargs: object) -> object:
        self._reject_provider_dynamic_code("eval")
        return self._original_eval(*args, **kwargs)

    def _guarded_thread_start(
        self,
        thread: threading.Thread,
        *args: object,
        **kwargs: object,
    ) -> object:
        self._reject_provider_dynamic_code("thread")
        return self._original_thread_start(thread, *args, **kwargs)

    def _guarded_start_new_thread(
        self,
        function: Callable[..., object],
        args: tuple[object, ...],
        kwargs: Mapping[str, object] | None = None,
    ) -> int:
        self._reject_provider_dynamic_code("thread")
        return self._original_start_new_thread(
            function,
            args,
            {} if kwargs is None else dict(kwargs),
        )

    def _reject_provider_dynamic_code(self, operation: str) -> None:
        caller = self._original_getframe(2)
        try:
            provider_call = False
            while True:
                if any(
                    _location_is_in_archive(
                        caller.f_code.co_filename,
                        archive,
                    )
                    for archive in self._verified_archives
                ) or _globals_are_from_archives(
                    caller.f_globals,
                    self._verified_archives,
                ):
                    provider_call = True
                    break
                parent = caller.f_back
                if parent is None:
                    break
                caller = parent
        finally:
            del caller
        if provider_call:
            self.violation = True
            raise ImportError(f"provider dynamic code execution is outside the verified closure: {operation}")

    def _compile_matches_verified_source(self, args: tuple[object, ...]) -> bool:
        if len(args) < 2 or not isinstance(args[1], str):
            return False
        source = args[0]
        if isinstance(source, str):
            source_bytes = source.encode("utf-8")
        elif isinstance(source, (bytes, bytearray)):
            source_bytes = bytes(source)
        else:
            return False
        expected = self._verified_source_digest(args[1])
        return expected == f"sha256:{hashlib.sha256(source_bytes).hexdigest()}"

    def _code_is_from_verified_source(self, args: tuple[object, ...]) -> bool:
        if not args or not isinstance(args[0], CodeType):
            return False
        code = args[0]
        return self._trusted_code_objects.get(id(code)) is code

    def _trust_code_graph(self, code: CodeType) -> None:
        self._trusted_code_objects[id(code)] = code
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                self._trust_code_graph(constant)

    def _capture_process_entry_points(
        self,
    ) -> list[tuple[object, str, Callable[..., object]]]:
        names_by_owner: list[tuple[object, tuple[str, ...]]] = [
            (
                subprocess,
                (
                    "Popen",
                    "run",
                    "call",
                    "check_call",
                    "check_output",
                    "getoutput",
                    "getstatusoutput",
                    "_fork_exec",
                ),
            ),
            (
                os,
                (
                    "system",
                    "popen",
                    "fork",
                    "forkpty",
                    "posix_spawn",
                    "posix_spawnp",
                    "spawnl",
                    "spawnle",
                    "spawnlp",
                    "spawnlpe",
                    "spawnv",
                    "spawnve",
                    "spawnvp",
                    "spawnvpe",
                    "execl",
                    "execle",
                    "execlp",
                    "execlpe",
                    "execv",
                    "execve",
                    "execvp",
                    "execvpe",
                ),
            ),
        ]
        entry_points: list[tuple[object, str, Callable[..., object]]] = []
        for owner, names in names_by_owner:
            for name in names:
                value = getattr(owner, name, None)
                if callable(value):
                    entry_points.append((owner, name, value))
        return entry_points

    def _process_guard(
        self,
        operation: str,
        original: Callable[..., object],
    ) -> Callable[..., object]:
        def guarded(*args: object, **kwargs: object) -> object:
            if self._provider_call_in_stack():
                self.violation = True
                os._exit(_PROVIDER_POLICY_VIOLATION_EXIT_CODE)
            return original(*args, **kwargs)

        guarded.__name__ = f"blocked_provider_{operation}"
        return guarded

    def _verified_source_digest(self, filename: str) -> str | None:
        location = os.path.abspath(filename)
        for root_index, root_value in enumerate(self._verified_archives):
            root = os.path.abspath(root_value)
            if not _location_is_in_archive(location, root):
                continue
            relative = Path(location).relative_to(root).as_posix()
            value = self._verified_root_snapshot.get(f"{root_index}/{relative}")
            if isinstance(value, str) and value.startswith("sha256:"):
                return value
        return None

    def _verified_roots_are_unchanged(self) -> bool:
        return _verified_roots_are_unchanged(
            self._verified_archives,
            self._verified_root_snapshot,
        )

    def _reject_modified_verified_roots(self) -> None:
        if not self._verified_roots_are_unchanged():
            self.violation = True
            raise ImportError("provider modified the verified import closure")

    def _provider_call_in_stack(self) -> bool:
        caller = self._original_getframe(2)
        try:
            while True:
                if _globals_are_from_archives(
                    caller.f_globals,
                    self._verified_archives,
                ) or any(
                    _location_is_in_archive(
                        caller.f_code.co_filename,
                        archive,
                    )
                    for archive in self._verified_archives
                ):
                    return True
                parent = caller.f_back
                if parent is None:
                    break
                caller = parent
        finally:
            del caller
        return False

    def _guarded_import(
        self,
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        provider_call = globals is None or _globals_are_from_archives(globals, self._verified_archives) or self._provider_call_in_stack()
        if provider_call:
            self._reject_modified_verified_roots()
        try:
            imported = self._original_import(name, globals, locals, fromlist, level)
        except ModuleNotFoundError:
            raise
        except BaseException:
            if provider_call:
                self.violation = True
            raise
        if not provider_call:
            return imported
        self._reject_modified_verified_roots()
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
        provider_call = self._provider_call_in_stack()
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
        if provider_call:
            self._reject_modified_verified_roots()
        try:
            imported = self._original_import_module(name, package)
        except ModuleNotFoundError:
            raise
        except BaseException:
            if provider_call:
                self.violation = True
            raise
        if provider_call:
            self._reject_modified_verified_roots()
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
        if not isinstance(module, ModuleType):
            return _root_exists_in_archives(root, self._verified_archives)
        return isinstance(module, ModuleType) and _module_is_from_archives(
            module,
            self._verified_archives,
        )


def _root_exists_in_archives(root: str, archives: list[str]) -> bool:
    if not root.isidentifier():
        return False
    for archive in archives:
        archive_root = Path(archive)
        if (archive_root / root).exists() or (archive_root / f"{root}.py").exists():
            return True
    return False


def _provider_error(
    import_gate: _ProviderImportGate,
    fallback_code: str,
) -> dict[str, object]:
    code = "provider_import_failed" if import_gate.violation else fallback_code
    return {"status": "error", "code": code}


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
                content = wheel.read(member)
                if target.read_bytes() == content:
                    continue
                raise ValueError(f"wheel members map to the same destination: {target}")
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


def _install_linux_process_creation_filter() -> None:
    if sys.platform != "linux":
        return
    import ctypes
    import errno

    syscall_numbers = {
        "x86_64": (56, (57, 58, 59, 322), 435),
        "amd64": (56, (57, 58, 59, 322), 435),
        "aarch64": (220, (221, 281), 435),
        "arm64": (220, (221, 281), 435),
    }.get(platform.machine().lower())
    if syscall_numbers is None:
        raise OSError("unsupported Linux process-filter architecture")

    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class SockFprog(ctypes.Structure):
        _fields_ = [
            ("len", ctypes.c_ushort),
            ("filter", ctypes.POINTER(SockFilter)),
        ]

    clone_number, blocked_numbers, clone3_number = syscall_numbers
    deny = 0x00050000 | errno.EPERM
    instructions = [
        SockFilter(0x20, 0, 0, 0),
        SockFilter(0x15, 0, 3, clone_number),
        SockFilter(0x20, 0, 0, 16),
        SockFilter(0x45, 1, 0, 0x00010000),
        SockFilter(0x06, 0, 0, deny),
        SockFilter(0x20, 0, 0, 0),
    ]
    for number in blocked_numbers:
        instructions.extend(
            (
                SockFilter(0x15, 0, 1, number),
                SockFilter(0x06, 0, 0, deny),
            )
        )
    instructions.extend(
        (
            SockFilter(0x15, 0, 1, clone3_number),
            SockFilter(0x06, 0, 0, 0x00050000 | errno.ENOSYS),
        )
    )
    instructions.append(SockFilter(0x06, 0, 0, 0x7FFF0000))
    filter_array = (SockFilter * len(instructions))(*instructions)
    program = SockFprog(len(instructions), filter_array)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(22, 2, ctypes.byref(program)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _execute(
    request: dict[str, object],
    execution_root: Path,
) -> dict[str, object]:
    implementation_artifact = request["implementation_artifact"]
    dependency_artifacts = request["dependency_artifacts"]
    module_name = request["module"]
    callable_name = request["callable"]
    parameters = request["parameters"]
    panel = request["input"]
    output_descriptors = request["output_fields"]
    output_alignment = request["output_alignment"]
    try:
        if not isinstance(output_descriptors, list) or not output_descriptors:
            raise TypeError("output fields are invalid")
        output_fields = [
            (descriptor["name"], descriptor["dtype"])
            for descriptor in output_descriptors
            if isinstance(descriptor, dict)
            and set(descriptor) == {"name", "dtype"}
            and isinstance(descriptor["name"], str)
            and isinstance(descriptor["dtype"], str)
            and descriptor["dtype"] in _OUTPUT_DTYPES
        ]
        if len(output_fields) != len(output_descriptors) or len({name for name, _ in output_fields}) != len(output_fields):
            raise TypeError("output fields are invalid")
    except (KeyError, TypeError):
        return {"status": "error", "code": "provider_execution_failed"}
    if (
        not isinstance(implementation_artifact, str)
        or not Path(implementation_artifact).is_absolute()
        or not isinstance(dependency_artifacts, list)
        or not all(isinstance(path, str) and Path(path).is_absolute() for path in dependency_artifacts)
        or not isinstance(module_name, str)
        or not isinstance(callable_name, str)
        or not isinstance(parameters, dict)
        or not isinstance(panel, dict)
        or not isinstance(output_alignment, str)
        or output_alignment not in _OUTPUT_ALIGNMENTS
    ):
        return {"status": "error", "code": "provider_execution_failed"}
    try:
        import numpy as np
        import pandas as pd
    except BaseException:
        return {"status": "error", "code": "provider_execution_failed"}
    trusted_assert_frame_equal = pd.testing.assert_frame_equal
    trusted_frame_copy = pd.DataFrame.copy
    trusted_pandas_primitives = _PandasValidationPrimitives(
        series_type=pd.Series,
        dataframe_type=pd.DataFrame,
        isna=cast(Callable[[object], object], pd.isna),
        numpy_scalar_type=np.generic,
        numpy_scalar_item=cast(Callable[[object], object], np.generic.item),
        series_tolist=pd.Series.tolist,
        dataframe_getitem=pd.DataFrame.__getitem__,
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
        original = trusted_frame_copy(frame, deep=True)
        repeated_frame = trusted_frame_copy(original, deep=True)
        repeated_original = trusted_frame_copy(original, deep=True)
    except BaseException:
        return {"status": "error", "code": "provider_execution_failed"}

    try:
        verified_archives, install_prefix = _materialize_wheels(
            [implementation_artifact, *dependency_artifacts],
            execution_root,
        )
    except BaseException:
        return {"status": "error", "code": "provider_import_failed"}
    sys.prefix = install_prefix
    sys.exec_prefix = install_prefix
    sys.dont_write_bytecode = True
    _hide_ambient_modules()
    modules_before_provider = set(sys.modules)
    _restrict_sys_path(verified_archives)
    try:
        verified_root_snapshot = _snapshot_verified_roots(verified_archives)
    except OSError:
        return {"status": "error", "code": "provider_import_failed"}
    import_gate = _ProviderImportGate(
        verified_archives,
        verified_root_snapshot,
    )
    try:
        _install_linux_process_creation_filter()
    except BaseException:
        return {"status": "error", "code": "provider_import_failed"}
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
                import_gate,
                "provider_import_failed",
            )

        try:
            output_runs: list[dict[str, list[object]]] = []
            for invocation_frame, invocation_original in (
                (frame, original),
                (repeated_frame, repeated_original),
            ):
                result = implementation(invocation_frame, **parameters)
                try:
                    trusted_assert_frame_equal(
                        invocation_frame,
                        invocation_original,
                        check_exact=True,
                        check_like=False,
                    )
                    if invocation_frame.attrs != invocation_original.attrs:
                        raise AssertionError("provider mutated frame metadata")
                except (AssertionError, TypeError, ValueError):
                    return _provider_error(
                        import_gate,
                        "provider_mutated_input",
                    )
                try:
                    outputs = _extract_outputs(
                        result,
                        invocation_original,
                        output_fields,
                        output_alignment,
                        trusted_pandas_primitives,
                    )
                except (KeyError, _OutputTypeError):
                    return _provider_error(
                        import_gate,
                        "baseline_mismatch",
                    )
                except (IndexError, TypeError, ValueError):
                    return _provider_error(
                        import_gate,
                        "provider_alignment_failed",
                    )
                if not import_gate._verified_roots_are_unchanged():
                    import_gate.violation = True
                    return _provider_error(
                        import_gate,
                        "provider_import_failed",
                    )
                output_runs.append(outputs)
            if (
                import_gate.violation
                or not _new_modules_are_allowed(
                    modules_before_provider,
                    verified_archives,
                )
                or not import_gate._verified_roots_are_unchanged()
            ):
                return {"status": "error", "code": "provider_import_failed"}
        except BaseException:
            return _provider_error(
                import_gate,
                "provider_execution_failed",
            )
    finally:
        import_gate.restore()
    return {
        "status": "ok",
        "outputs": output_runs[0],
        "repeated_outputs": output_runs[1],
    }


def _execute_outside_verifier_stack(
    request: dict[str, object],
    execution_root: Path,
) -> dict[str, object]:
    responses: list[dict[str, object]] = []

    def run_provider_verification() -> None:
        try:
            responses.append(_execute(request, execution_root))
        except BaseException:
            responses.append({"status": "error", "code": "provider_execution_failed"})

    execution_thread = threading.Thread(
        target=run_provider_verification,
        name="oxq-provider-verification",
    )
    execution_thread.start()
    execution_thread.join()
    if len(responses) != 1:
        return {"status": "error", "code": "provider_execution_failed"}
    return responses[0]


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    try:
        secret_value = sys.stdin.buffer.readline(65).strip()
        if len(secret_value) != 64:
            raise ValueError("response authentication secret is invalid")
        response_secret = bytes.fromhex(secret_value.decode("ascii"))
    except (AttributeError, UnicodeError, ValueError):
        return 2
    gate_path_value = os.environ.pop("OXQ_BASELINE_WINDOWS_JOB_GATE", None)
    if gate_path_value is not None:
        gate_path = Path(gate_path_value)
        deadline = time.monotonic() + 30
        while not gate_path.is_file():
            if time.monotonic() >= deadline:
                return 2
            time.sleep(0.005)
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    sys.argv[:] = [sys.argv[0]]
    if hasattr(sys, "orig_argv"):
        sys.orig_argv = [sys.orig_argv[0]]
    response: dict[str, object]
    try:
        request = _read_request(request_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        response = {"status": "error", "code": "provider_execution_failed"}
        _write_response(
            response_path,
            _authenticated_response(response, response_secret),
        )
        return 0
    response = _execute_outside_verifier_stack(
        request,
        request_path.parent / "wheel-layout",
    )
    _write_response(
        response_path,
        _authenticated_response(response, response_secret),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
