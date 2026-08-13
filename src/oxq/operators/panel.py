"""Conversions between engine symbol frames and the QuantPanel contract."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any, Literal

import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype

from oxq.operators._schema import load_contract_schema
from oxq.operators.errors import DuplicateKeyError, InvalidPanelError, MissingColumnError
from oxq.operators.types import OperatorContext, TimestampSemantics

Alignment = Literal["preserve_input_order", "canonical_order", "explicit_keyed_output"]
_KEY_COLUMNS = ["date", "code"]


class QuantPanelAdapter:
    """Strict, non-mutating QuantPanel conversion and validation."""

    @staticmethod
    def to_panel(frames: Mapping[str, pd.DataFrame], context: OperatorContext) -> pd.DataFrame:
        if not isinstance(frames, Mapping):
            raise InvalidPanelError("symbol frames must be a mapping")
        columns: list[str] | None = None
        parts: list[pd.DataFrame] = []
        for code, frame in frames.items():
            if not isinstance(code, str) or not code:
                raise InvalidPanelError("symbol codes must be non-empty strings")
            if not isinstance(frame, pd.DataFrame):
                raise InvalidPanelError(f"symbol frame must be a DataFrame: {code}")
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise InvalidPanelError(f"symbol frame index must be a DatetimeIndex: {code}")
            if frame.columns.has_duplicates:
                raise InvalidPanelError(f"symbol frame columns must be unique: {code}")
            if any(not isinstance(column, str) or not column for column in frame.columns):
                raise InvalidPanelError(f"symbol frame columns must be non-empty strings: {code}")
            frame_columns = [str(column) for column in frame.columns]
            reserved = sorted(set(frame_columns) & set(_KEY_COLUMNS))
            if reserved:
                raise InvalidPanelError(f"symbol frame uses reserved QuantPanel columns: {', '.join(reserved)}")
            if columns is None:
                columns = frame_columns
            elif frame_columns != columns:
                raise InvalidPanelError("all symbol frames must have identical ordered columns")
            dates = QuantPanelAdapter._panel_dates(frame.index, context, code)
            part = frame.copy(deep=True).reset_index(drop=True)
            part.insert(0, "code", code)
            part.insert(0, "date", dates)
            parts.append(part)
        if not parts:
            panel = pd.DataFrame(columns=[*_KEY_COLUMNS, *(columns or [])])
            date_dtype = "datetime64[ns]" if context.timestamp_semantics is TimestampSemantics.SESSION_DATE else "datetime64[ns, UTC]"
            panel["date"] = pd.Series(dtype=date_dtype)
            panel["code"] = panel["code"].astype("string")
            QuantPanelAdapter.validate_panel(panel, context, require_canonical_order=True)
            return panel
        panel = pd.concat(parts, ignore_index=True)
        panel["code"] = panel["code"].astype("string")
        panel = panel.sort_values(_KEY_COLUMNS, kind="stable", ignore_index=True)
        QuantPanelAdapter.validate_panel(panel, context, require_canonical_order=True)
        return panel

    @staticmethod
    def to_symbol_frames(panel: pd.DataFrame, context: OperatorContext) -> dict[str, pd.DataFrame]:
        QuantPanelAdapter.validate_panel(panel, context)
        result: dict[str, pd.DataFrame] = {}
        for code, rows in panel.groupby("code", sort=True, observed=True):
            frame = rows.drop(columns="code").set_index("date")
            frame.index = pd.DatetimeIndex(frame.index)
            if context.timestamp_semantics is TimestampSemantics.SESSION_DATE:
                frame.index = frame.index.tz_localize(context.timezone)
            else:
                frame.index = frame.index.tz_convert(context.timezone)
            frame.index.name = None
            result[str(code)] = frame
        return result

    @staticmethod
    def validate_panel(
        panel: pd.DataFrame,
        context: OperatorContext,
        *,
        require_canonical_order: bool = False,
    ) -> None:
        if not isinstance(panel, pd.DataFrame):
            raise InvalidPanelError("QuantPanel must be a pandas DataFrame")
        for column in _KEY_COLUMNS:
            if column not in panel.columns:
                raise MissingColumnError(f"QuantPanel requires {column}", details={"column": column})
        if panel.columns.has_duplicates:
            raise InvalidPanelError("QuantPanel columns must be unique")
        if not is_datetime64_any_dtype(panel["date"]):
            raise InvalidPanelError("QuantPanel date must have datetime dtype")
        if panel["date"].isna().any():
            raise InvalidPanelError("QuantPanel date must not be missing")
        valid_codes = all(isinstance(value, str) and bool(value) for value in panel["code"].tolist())
        if panel["code"].isna().any() or not valid_codes:
            raise InvalidPanelError("QuantPanel code must contain non-empty strings")
        if not (is_string_dtype(panel["code"]) or panel["code"].dtype == object):
            raise InvalidPanelError("QuantPanel code must have string-compatible dtype")
        if panel.duplicated(_KEY_COLUMNS).any():
            duplicates = panel.loc[panel.duplicated(_KEY_COLUMNS, keep=False), _KEY_COLUMNS]
            raise DuplicateKeyError(
                "QuantPanel contains duplicate (date, code) keys",
                details={"count": len(duplicates)},
            )
        date_is_aware = isinstance(panel["date"].dtype, pd.DatetimeTZDtype)
        if context.timestamp_semantics is TimestampSemantics.SESSION_DATE:
            if date_is_aware:
                raise InvalidPanelError("daily session-date QuantPanel must use timezone-naive dates")
            if not panel["date"].eq(panel["date"].dt.normalize()).all():
                raise InvalidPanelError("daily session-date QuantPanel must contain normalized session dates")
        elif not date_is_aware:
            raise InvalidPanelError("intraday/event QuantPanel dates must be timezone-aware")
        if require_canonical_order:
            keys = panel[_KEY_COLUMNS].reset_index(drop=True)
            canonical = panel.sort_values(_KEY_COLUMNS, kind="stable", ignore_index=True)[_KEY_COLUMNS]
            if not keys.equals(canonical):
                raise InvalidPanelError("QuantPanel must use canonical date/code order")

    @staticmethod
    def validate_output(
        input_panel: pd.DataFrame,
        output_panel: pd.DataFrame,
        context: OperatorContext,
        *,
        alignment: Alignment,
    ) -> None:
        QuantPanelAdapter.validate_panel(input_panel, context)
        QuantPanelAdapter.validate_panel(output_panel, context)
        input_keys = pd.MultiIndex.from_frame(input_panel[_KEY_COLUMNS])
        output_keys = pd.MultiIndex.from_frame(output_panel[_KEY_COLUMNS])
        if alignment == "explicit_keyed_output":
            if not output_keys.isin(input_keys).all():
                raise InvalidPanelError("explicit_keyed_output contains keys outside the input panel")
            return
        if len(input_keys) != len(output_keys) or set(input_keys) != set(output_keys):
            raise InvalidPanelError(f"{alignment} output must contain the same keys as input")
        if alignment == "preserve_input_order" and not input_keys.equals(output_keys):
            raise InvalidPanelError("preserve_input_order output must preserve input key order")
        if alignment == "canonical_order":
            canonical = output_panel.sort_values(_KEY_COLUMNS, kind="stable", ignore_index=True)
            if not output_panel[_KEY_COLUMNS].reset_index(drop=True).equals(canonical[_KEY_COLUMNS]):
                raise InvalidPanelError("canonical_order output must be sorted by date and code")

    @staticmethod
    def _panel_dates(index: pd.DatetimeIndex, context: OperatorContext, code: str) -> pd.DatetimeIndex:
        if context.timestamp_semantics is TimestampSemantics.SESSION_DATE:
            if index.tz is None:
                dates = index
            else:
                dates = index.tz_convert(context.timezone).tz_localize(None)
            if not dates.equals(dates.normalize()):
                raise InvalidPanelError(f"daily index must contain normalized session dates: {code}")
            return dates
        if index.tz is None:
            raise InvalidPanelError(f"intraday/event index must be timezone-aware: {code}")
        return index.tz_convert("UTC")


def validate_serialized_quant_panel(payload: Mapping[str, Any]) -> None:
    """Validate the JSON Schema and composite-key semantics of a serialized QuantPanel."""

    schema = load_contract_schema("quant-panel-v1.schema.json")
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path)
        raise InvalidPanelError(f"serialized QuantPanel {path or 'payload'}: {error.message}")
    rows = payload["rows"]
    timestamp_semantics = payload["context"]["timestamp_semantics"]
    seen: set[tuple[str | datetime, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for row in rows:
        parsed_date = _validate_serialized_date(row["date"], timestamp_semantics)
        key = (parsed_date, row["code"])
        if key in seen:
            duplicates.append((row["date"], row["code"]))
        seen.add(key)
    if duplicates:
        raise DuplicateKeyError(
            "serialized QuantPanel contains duplicate (date, code) keys",
            details={"keys": [list(key) for key in duplicates]},
        )


def _validate_serialized_date(value: str, timestamp_semantics: str) -> str | datetime:
    try:
        if timestamp_semantics == TimestampSemantics.SESSION_DATE:
            parsed = date.fromisoformat(value)
            if parsed.isoformat() != value:
                raise ValueError
            return value
        else:
            normalized_value = f"{value[:-1]}+00:00" if value.endswith(("Z", "z")) else value
            parsed_datetime = datetime.fromisoformat(normalized_value)
            if parsed_datetime.tzinfo is None:
                raise ValueError
            return parsed_datetime.astimezone(UTC)
    except ValueError as exc:
        raise InvalidPanelError(f"serialized QuantPanel date is invalid for timestamp_semantics={timestamp_semantics}: {value}") from exc
