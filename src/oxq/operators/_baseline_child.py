"""Self-contained strict-JSON child for exact-wheel baseline execution."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_PLATFORM_RUNTIME_ROOTS = {"numpy", "pandas"}


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
    }:
        raise ValueError("request fields are invalid")
    return value


def _write_response(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _error(path: Path, code: str) -> int:
    _write_response(path, {"status": "error", "code": code})
    return 0


def _json_value(value: object) -> object:
    import pandas as pd

    missing = pd.isna(value)  # type: ignore[call-overload]
    if isinstance(missing, bool) and missing:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("provider output is not JSON scalar data")


def _extract_output(
    result: object,
    frame: Any,
    output_field: str,
) -> list[object]:
    import pandas as pd

    if isinstance(result, pd.Series):
        if result.name != output_field:
            raise KeyError("declared output field is missing")
        if len(result) != len(frame) or not result.index.equals(frame.index):
            raise IndexError("series alignment changed")
        return [_json_value(value) for value in result.tolist()]
    if isinstance(result, pd.DataFrame):
        if output_field not in result.columns:
            raise KeyError("declared output field is missing")
        if len(result) != len(frame):
            raise IndexError("dataframe length changed")
        key_columns = {"date", "code"}.intersection(result.columns)
        if key_columns and key_columns != {"date", "code"}:
            raise IndexError("partial key alignment cannot be proven")
        if key_columns:
            if result[["date", "code"]].to_dict("records") != frame[
                ["date", "code"]
            ].to_dict("records"):
                raise IndexError("dataframe keys changed")
        elif not result.index.equals(frame.index):
            raise IndexError("dataframe index changed")
        return [_json_value(value) for value in result[output_field].tolist()]
    raise TypeError("provider output must be a Series or DataFrame")


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
        locations.extend(
            location for location in search_locations if isinstance(location, str)
        )
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
    return bool(locations) and any(
        _location_is_in_archive(location, archive)
        for location in locations
        for archive in archives
    )


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


def _execute(request: dict[str, object], response_path: Path) -> int:
    implementation_artifact = request["implementation_artifact"]
    dependency_artifacts = request["dependency_artifacts"]
    module_name = request["module"]
    callable_name = request["callable"]
    parameters = request["parameters"]
    panel = request["input"]
    output_field = request["output_field"]
    if (
        not isinstance(implementation_artifact, str)
        or not Path(implementation_artifact).is_absolute()
        or not isinstance(dependency_artifacts, list)
        or not all(
            isinstance(path, str) and Path(path).is_absolute()
            for path in dependency_artifacts
        )
        or not isinstance(module_name, str)
        or not isinstance(callable_name, str)
        or not isinstance(parameters, dict)
        or not isinstance(panel, dict)
        or not isinstance(output_field, str)
    ):
        return _error(response_path, "provider_execution_failed")
    try:
        import numpy as np  # noqa: F401
        import pandas as pd
    except BaseException:
        return _error(response_path, "provider_execution_failed")

    modules_before_provider = set(sys.modules)
    verified_archives = [implementation_artifact, *dependency_artifacts]
    sys.path[:0] = verified_archives
    try:
        module = importlib.import_module(module_name)
        if not _module_is_from_archives(module, [implementation_artifact]):
            raise ImportError("manifest module is not from implementation artifact")
        implementation = getattr(module, callable_name)
        if not callable(implementation):
            raise ImportError("manifest callable is not callable")
        if not _new_modules_are_allowed(modules_before_provider, verified_archives):
            raise ImportError("provider imported an undeclared ambient dependency")
    except BaseException:
        return _error(response_path, "provider_import_failed")

    try:
        records = panel["records"]
        if not isinstance(records, list):
            raise TypeError("panel records are not a list")
        frame = pd.DataFrame.from_records(records)
        frame.index = pd.MultiIndex.from_frame(frame[["date", "code"]])
        original = frame.copy(deep=True)
        result = implementation(frame, **parameters)
        if not _new_modules_are_allowed(modules_before_provider, verified_archives):
            return _error(response_path, "provider_import_failed")
        try:
            pd.testing.assert_frame_equal(
                frame,
                original,
                check_exact=True,
                check_like=False,
            )
        except AssertionError:
            return _error(response_path, "provider_mutated_input")
        try:
            output = _extract_output(result, original, output_field)
        except KeyError:
            return _error(response_path, "baseline_mismatch")
        except (IndexError, TypeError, ValueError):
            return _error(response_path, "provider_alignment_failed")
    except BaseException:
        return _error(response_path, "provider_execution_failed")
    _write_response(response_path, {"status": "ok", "output": output})
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
