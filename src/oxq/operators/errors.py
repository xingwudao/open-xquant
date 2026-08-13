"""Structured quant operator errors."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np


def _freeze_details(value: Any, *, path: str = "details", active: set[int] | None = None) -> Any:
    if active is None:
        active = set()
    if isinstance(value, np.ndarray):
        identity = id(value)
        if identity in active:
            raise TypeError(f"{path} must form a finite JSON-compatible tree")
        active.add(identity)
        try:
            if value.dtype.hasobject:
                return _freeze_details(value.tolist(), path=path, active=active)
            _freeze_details(value.tolist(), path=path, active=active)
            contiguous = np.ascontiguousarray(value)
            return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
        finally:
            active.remove(identity)
    if isinstance(value, np.generic):
        return _freeze_details(value.item(), path=path, active=active)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise TypeError(f"{path} must form a finite JSON-compatible tree")
        active.add(identity)
        try:
            frozen = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} mapping keys must be strings for JSON compatibility; got {type(key).__name__} key {key!r}")
                frozen[key] = _freeze_details(item, path=f"{path}[{key!r}]", active=active)
            return MappingProxyType(frozen)
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise TypeError(f"{path} must form a finite JSON-compatible tree")
        active.add(identity)
        try:
            return tuple(_freeze_details(item, path=f"{path}[{index}]", active=active) for index, item in enumerate(value))
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        identity = id(value)
        if identity in active:
            raise TypeError(f"{path} must form a finite JSON-compatible tree")
        active.add(identity)
        try:
            return frozenset(_freeze_details(item, path=f"{path} set item", active=active) for item in value)
        finally:
            active.remove(identity)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        raise TypeError(f"{path} contains a non-finite number that is not strict JSON-compatible")
    raise TypeError(f"{path} contains unsupported JSON leaf type {type(value).__name__}")


def _materialize_details(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _materialize_details(value.tolist())
    if isinstance(value, Mapping):
        return {key: _materialize_details(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_materialize_details(item) for item in value]
    if isinstance(value, frozenset):
        items = [_materialize_details(item) for item in value]
        return sorted(items, key=repr)
    return copy.deepcopy(value)


class OperatorError(ValueError):
    """Base error carrying stable machine-readable diagnostics."""

    code = "operator_error"

    def __init__(
        self,
        message: str,
        *,
        operator_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        if operator_id is not None and not isinstance(operator_id, str):
            raise TypeError("operator_id must be a string or None")
        if type(retryable) is not bool:
            raise TypeError("retryable must be a boolean")
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("details must be a mapping")
        self.operator_id = operator_id
        self.details = _freeze_details(details if details is not None else {})
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "operator_id": self.operator_id,
            "message": str(self),
            "details": _materialize_details(self.details),
            "retryable": self.retryable,
        }


class InvalidPanelError(OperatorError):
    code = "invalid_panel"


class MissingColumnError(InvalidPanelError):
    code = "missing_column"


class DuplicateKeyError(InvalidPanelError):
    code = "duplicate_key"


class InvalidParameterError(OperatorError):
    code = "invalid_parameter"


class InvalidManifestError(OperatorError):
    code = "invalid_manifest"


class InsufficientHistoryError(OperatorError):
    code = "insufficient_history"


class InsufficientCrossSectionError(OperatorError):
    code = "insufficient_cross_section"


class CausalityViolationError(OperatorError):
    code = "causality_violation"


class DependencyUnavailableError(OperatorError):
    code = "dependency_unavailable"


class DataFetchError(OperatorError):
    code = "data_fetch_error"


class NumericalComputationError(OperatorError):
    code = "numerical_computation"


class ContractViolationError(OperatorError):
    code = "contract_violation"
