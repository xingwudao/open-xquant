"""Stable request and result types for quant operator execution."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import pandas as pd

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMMUTABLE_LEAF_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _freeze(
    value: Any,
    *,
    permanent_arrays: bool = False,
    string_keys_only: bool = False,
    immutable_leaves_only: bool = False,
    path: str = "value",
) -> Any:
    if isinstance(value, np.ndarray):
        if permanent_arrays:
            if value.dtype.hasobject:
                raise TypeError("object-dtype arrays cannot be frozen")
            contiguous = np.ascontiguousarray(value)
            return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
        frozen = copy.deepcopy(value)
        frozen.setflags(write=False)
        return frozen
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
            )
            for item in value
        )
    if immutable_leaves_only:
        if type(value) in _IMMUTABLE_LEAF_TYPES:
            return value
        raise TypeError(f"{path} has unsupported leaf type {type(value).__name__}")
    return copy.deepcopy(value)


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
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
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
        if not self.operator_id:
            raise ValueError("operator_id must be a non-empty string")
        if not isinstance(self.input_panel, pd.DataFrame):
            raise TypeError("input_panel must be a pandas DataFrame")
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
        if self.dropped_rows > self.input_rows:
            raise ValueError("dropped_rows cannot exceed input_rows")
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True, slots=True)
class OperatorProvenance:
    operator_id: str
    operator_version: str
    implementation_digest: str

    def __post_init__(self) -> None:
        if not self.operator_id or not self.operator_version:
            raise ValueError("operator provenance identity must be non-empty")
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
            if not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("training_data_digest", "state_digest"):
            if not _DIGEST_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a sha256 digest")
        if not self.feature_order or any(not isinstance(feature, str) or not feature for feature in self.feature_order):
            raise ValueError("feature_order must contain non-empty feature names")
        object.__setattr__(self, "feature_order", tuple(self.feature_order))
        for name in ("training_data_summary", "parameters", "learned_state"):
            object.__setattr__(
                self,
                name,
                _freeze(
                    getattr(self, name),
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
        return cls(data=data, diagnostics=diagnostics, provenance=provenance, metadata=metadata or {})


class QuantOperatorExecutor(Protocol):
    def execute(self, request: OperatorRequest) -> OperatorResult: ...
