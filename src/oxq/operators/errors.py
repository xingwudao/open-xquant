"""Structured quant operator errors."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np


def _freeze_details(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _freeze_details(value.tolist())
        contiguous = np.ascontiguousarray(value)
        return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
    if isinstance(value, np.generic):
        return _freeze_details(value.item())
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_details(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_details(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_details(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return copy.deepcopy(value)


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
