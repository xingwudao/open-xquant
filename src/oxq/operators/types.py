"""Stable request and result types for quant operator execution."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ISO_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_ISO_DATETIME_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_IMMUTABLE_LEAF_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _parse_training_boundary(value: str) -> tuple[str, date | datetime]:
    try:
        if _ISO_DATE_RE.fullmatch(value):
            return "date", date.fromisoformat(value)
        if _ISO_DATETIME_RE.fullmatch(value):
            return "datetime", datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    raise ValueError(
        "training boundaries must be ISO 8601 dates (YYYY-MM-DD) or timezone-aware datetimes (YYYY-MM-DDTHH:MM:SS[.ffffff](Z|+HH:MM))"
    )


def _freeze(
    value: Any,
    *,
    permanent_arrays: bool = False,
    string_keys_only: bool = False,
    immutable_leaves_only: bool = False,
    path: str = "value",
    _active_container_ids: set[int] | None = None,
) -> Any:
    active = set() if _active_container_ids is None else _active_container_ids
    is_container = isinstance(value, (np.ndarray, Mapping, list, tuple, set, frozenset))
    if is_container:
        container_id = id(value)
        if container_id in active:
            raise TypeError(f"{path} contains a cyclic container")
        active.add(container_id)
    else:
        container_id = None
    try:
        if isinstance(value, np.ndarray):
            if permanent_arrays:
                if value.dtype.hasobject:
                    raise TypeError("object-dtype arrays cannot be frozen")
                contiguous = np.ascontiguousarray(value)
                return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
            frozen = copy.deepcopy(value)
            frozen.setflags(write=False)
            return frozen
        if immutable_leaves_only and isinstance(value, np.generic):
            return _freeze(
                value.item(),
                permanent_arrays=permanent_arrays,
                string_keys_only=string_keys_only,
                immutable_leaves_only=True,
                path=path,
                _active_container_ids=active,
            )
        if isinstance(value, Mapping):
            frozen_items = {}
            for key, item in value.items():
                if string_keys_only and not isinstance(key, str):
                    raise TypeError(f"{path} mapping keys must be strings; got {type(key).__name__} key {key!r}")
                frozen_items[key] = _freeze(
                    item,
                    permanent_arrays=permanent_arrays,
                    string_keys_only=string_keys_only,
                    immutable_leaves_only=immutable_leaves_only,
                    path=f"{path}[{key!r}]",
                    _active_container_ids=active,
                )
            return MappingProxyType(frozen_items)
        if isinstance(value, (list, tuple)):
            return tuple(
                _freeze(
                    item,
                    permanent_arrays=permanent_arrays,
                    string_keys_only=string_keys_only,
                    immutable_leaves_only=immutable_leaves_only,
                    path=f"{path}[{index}]",
                    _active_container_ids=active,
                )
                for index, item in enumerate(value)
            )
        if isinstance(value, (set, frozenset)):
            return frozenset(
                _freeze(
                    item,
                    permanent_arrays=permanent_arrays,
                    string_keys_only=string_keys_only,
                    immutable_leaves_only=immutable_leaves_only,
                    path=f"{path} set item",
                    _active_container_ids=active,
                )
                for item in value
            )
        if immutable_leaves_only:
            if type(value) in _IMMUTABLE_LEAF_TYPES:
                return value
            raise TypeError(f"{path} has unsupported leaf type {type(value).__name__}")
        return copy.deepcopy(value)
    finally:
        if container_id is not None:
            active.remove(container_id)


class OperatorScope(StrEnum):
    TIME_SERIES = "time_series"
    CROSS_SECTION = "cross_section"
    PANEL = "panel"
    RESEARCH_ONLY = "research_only"


class OperatorLifecycle(StrEnum):
    STATELESS = "stateless"
    FIT_TRANSFORM = "fit_transform"
    EVALUATION = "evaluation"
    DATA_ACCESS = "data_access"
    VISUALIZATION = "visualization"


class OperatorCausality(StrEnum):
    PAST_ONLY = "past_only"
    LABEL_DEPENDENT = "label_dependent"
    FUTURE_USING = "future_using"


class OperatorAvailability(StrEnum):
    PRE_OPEN = "pre_open_t"
    OPEN = "open_t"
    INTRADAY = "intraday_t"
    CLOSE = "close_t"
    AFTER_CLOSE = "after_close_t"
    PUBLICATION_TIME = "publication_time"


class TimestampSemantics(StrEnum):
    SESSION_DATE = "session_date"
    BAR_OPEN = "bar_open"
    BAR_CLOSE = "bar_close"
    EVENT_TIME = "event_time"
    PUBLICATION_TIME = "publication_time"


class PriceAdjustment(StrEnum):
    RAW = "raw"
    FORWARD = "forward_adjusted"
    BACKWARD = "backward_adjusted"
    TOTAL_RETURN = "total_return_adjusted"


@dataclass(frozen=True, slots=True)
class OperatorContext:
    timezone: str
    calendar: str
    frequency: str
    timestamp_semantics: TimestampSemantics | str
    currency: str
    price_adjustment: PriceAdjustment | str
    data_version: str
    source: str
    evaluation_time: OperatorAvailability | str

    def __post_init__(self) -> None:
        for name in ("timezone", "calendar", "frequency", "currency", "data_version", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        try:
            ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError(f"timezone {self.timezone!r} is not supported by the zoneinfo database") from exc
        object.__setattr__(self, "timestamp_semantics", TimestampSemantics(self.timestamp_semantics))
        object.__setattr__(self, "price_adjustment", PriceAdjustment(self.price_adjustment))
        object.__setattr__(self, "evaluation_time", OperatorAvailability(self.evaluation_time))


@dataclass(frozen=True, slots=True)
class OperatorRequest:
    operator_id: str
    parameters: Mapping[str, Any]
    input_panel: pd.DataFrame
    context: OperatorContext

    def __post_init__(self) -> None:
        if not isinstance(self.operator_id, str):
            raise TypeError("operator_id must be a string")
        if not self.operator_id:
            raise ValueError("operator_id must be a non-empty string")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        if not isinstance(self.input_panel, pd.DataFrame):
            raise TypeError("input_panel must be a pandas DataFrame")
        if not isinstance(self.context, OperatorContext):
            raise TypeError("context must be an OperatorContext")
        object.__setattr__(
            self,
            "parameters",
            _freeze(self.parameters, permanent_arrays=True, string_keys_only=True, path="parameters"),
        )


@dataclass(frozen=True, slots=True)
class OperatorDiagnostics:
    input_rows: int
    output_rows: int
    warmup_rows: int = 0
    dropped_rows: int = 0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = (self.input_rows, self.output_rows, self.warmup_rows, self.dropped_rows)
        if any(type(value) is not int for value in counts):
            raise TypeError("diagnostic row counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("diagnostic row counts must be non-negative")
        if self.output_rows > self.input_rows:
            raise ValueError("output_rows cannot exceed input_rows")
        if self.warmup_rows > self.input_rows:
            raise ValueError("warmup_rows cannot exceed input_rows")
        if self.dropped_rows > self.input_rows:
            raise ValueError("dropped_rows cannot exceed input_rows")
        if self.output_rows + self.dropped_rows != self.input_rows:
            raise ValueError("output_rows + dropped_rows must match input_rows")
        if (
            isinstance(self.warnings, (str, bytes))
            or not isinstance(self.warnings, Sequence)
            or any(not isinstance(warning, str) for warning in self.warnings)
        ):
            raise TypeError("warnings must be a sequence of strings")
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class OperatorProvenance:
    operator_id: str
    operator_version: str
    implementation_digest: str

    def __post_init__(self) -> None:
        for name in ("operator_id", "operator_version"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not _DIGEST_RE.fullmatch(self.implementation_digest):
            raise ValueError("implementation_digest must be a sha256 digest")


@dataclass(frozen=True, slots=True)
class FittedOperatorState:
    operator_id: str
    operator_version: str
    training_start: str
    training_end: str
    training_data_digest: str
    training_data_summary: Mapping[str, Any]
    feature_order: tuple[str, ...]
    parameters: Mapping[str, Any]
    learned_state: Mapping[str, Any]
    random_seed: int | None
    dependency_versions: Mapping[str, str]
    state_digest: str

    def __post_init__(self) -> None:
        for name in ("operator_id", "operator_version", "training_start", "training_end"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value:
                raise ValueError(f"{name} must be a non-empty string")
        start_kind, training_start = _parse_training_boundary(self.training_start)
        end_kind, training_end = _parse_training_boundary(self.training_end)
        if start_kind != end_kind:
            raise ValueError("training boundaries must be ISO 8601 values of the same kind")
        if training_start > training_end:
            raise ValueError("training_start must not be after training_end")
        for name in ("training_data_digest", "state_digest"):
            if not _DIGEST_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a sha256 digest")
        if isinstance(self.feature_order, (str, bytes)) or not isinstance(self.feature_order, Sequence):
            raise TypeError("feature_order must be a non-string sequence")
        if not self.feature_order or any(not isinstance(feature, str) or not feature for feature in self.feature_order):
            raise ValueError("feature_order must contain non-empty feature names")
        object.__setattr__(self, "feature_order", tuple(self.feature_order))
        if self.random_seed is not None and type(self.random_seed) is not int:
            raise TypeError("random_seed must be an integer or None")
        for name in ("training_data_summary", "parameters", "learned_state"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(
                self,
                name,
                _freeze(
                    value,
                    permanent_arrays=True,
                    string_keys_only=True,
                    immutable_leaves_only=True,
                    path=name,
                ),
            )
        if not isinstance(self.dependency_versions, Mapping) or any(
            not isinstance(name, str) or not isinstance(version, str) for name, version in self.dependency_versions.items()
        ):
            raise TypeError("dependency_versions must be a mapping of strings to strings")
        if any(not name or not version for name, version in self.dependency_versions.items()):
            raise ValueError("dependency_versions must contain non-empty strings")
        object.__setattr__(
            self,
            "dependency_versions",
            _freeze(
                self.dependency_versions,
                permanent_arrays=True,
                string_keys_only=True,
                immutable_leaves_only=True,
                path="dependency_versions",
            ),
        )


@dataclass(frozen=True, slots=True)
class OperatorResult:
    data: pd.DataFrame
    diagnostics: OperatorDiagnostics
    provenance: OperatorProvenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("operator result data must be a pandas DataFrame")
        if not isinstance(self.diagnostics, OperatorDiagnostics):
            raise TypeError("diagnostics must be an OperatorDiagnostics")
        if not isinstance(self.provenance, OperatorProvenance):
            raise TypeError("provenance must be an OperatorProvenance")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if len(self.data) != self.diagnostics.output_rows:
            raise ValueError("diagnostics.output_rows must match result data rows")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @classmethod
    def for_request(
        cls,
        request: OperatorRequest,
        *,
        data: pd.DataFrame,
        diagnostics: OperatorDiagnostics,
        provenance: OperatorProvenance,
        metadata: Mapping[str, Any] | None = None,
    ) -> OperatorResult:
        if provenance.operator_id != request.operator_id:
            raise ValueError("provenance operator_id must match request operator_id")
        if diagnostics.input_rows != len(request.input_panel):
            raise ValueError("diagnostics.input_rows must match request input_panel rows")
        if diagnostics.output_rows + diagnostics.dropped_rows != diagnostics.input_rows:
            raise ValueError("diagnostics.output_rows + diagnostics.dropped_rows must match diagnostics.input_rows")
        return cls(
            data=data,
            diagnostics=diagnostics,
            provenance=provenance,
            metadata={} if metadata is None else metadata,
        )


class QuantOperatorExecutor(Protocol):
    def execute(self, request: OperatorRequest) -> OperatorResult: ...
