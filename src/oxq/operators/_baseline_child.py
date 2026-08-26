"""Self-contained strict-JSON child for exact-wheel baseline execution."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any


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
        "wheel_paths",
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


def _execute(request: dict[str, object], response_path: Path) -> int:
    wheel_paths = request["wheel_paths"]
    module_name = request["module"]
    callable_name = request["callable"]
    parameters = request["parameters"]
    panel = request["input"]
    output_field = request["output_field"]
    if (
        not isinstance(wheel_paths, list)
        or not wheel_paths
        or not all(isinstance(path, str) and Path(path).is_absolute() for path in wheel_paths)
        or not isinstance(module_name, str)
        or not isinstance(callable_name, str)
        or not isinstance(parameters, dict)
        or not isinstance(panel, dict)
        or not isinstance(output_field, str)
    ):
        return _error(response_path, "provider_execution_failed")
    sys.path[:0] = wheel_paths
    try:
        module = importlib.import_module(module_name)
        implementation = getattr(module, callable_name)
        if not callable(implementation):
            raise ImportError("manifest callable is not callable")
    except BaseException:
        return _error(response_path, "provider_import_failed")

    try:
        import pandas as pd

        records = panel["records"]
        if not isinstance(records, list):
            raise TypeError("panel records are not a list")
        frame = pd.DataFrame.from_records(records)
        frame.index = pd.MultiIndex.from_frame(frame[["date", "code"]])
        original = frame.copy(deep=True)
        result = implementation(frame, **parameters)
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
