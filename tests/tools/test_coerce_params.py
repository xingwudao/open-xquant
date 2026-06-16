"""Tests for tools-layer parameter coercion.

Tool clients (LLMs) may send numeric params as strings (e.g. "0.5" instead of 0.5).
The tools layer must coerce them before passing to SDK compute() methods.
"""

from __future__ import annotations

import pandas as pd
import pytest

from oxq.indicators.power_ratio import PowerRatio
from oxq.indicators.sma import SMA
from oxq.tools._coerce import coerce_compute_params


class TestCoerceComputeParams:
    """coerce_compute_params should cast string values to annotated types."""

    def test_float_from_string(self):
        """String '0.5' should become float 0.5 for PowerRatio.exponent."""
        params = {"col_a": "a", "col_b": "b", "exponent": "0.5"}
        result = coerce_compute_params(PowerRatio(), params)
        assert result["exponent"] == 0.5
        assert isinstance(result["exponent"], float)

    def test_float_passthrough(self):
        """Already-correct float 0.5 should pass through unchanged."""
        params = {"col_a": "a", "col_b": "b", "exponent": 0.5}
        result = coerce_compute_params(PowerRatio(), params)
        assert result["exponent"] == 0.5
        assert isinstance(result["exponent"], float)

    def test_int_from_string(self):
        """String '20' should become int 20 for SMA.period."""
        params = {"period": "20"}
        result = coerce_compute_params(SMA(), params)
        assert result["period"] == 20
        assert isinstance(result["period"], int)

    def test_int_passthrough(self):
        """Already-correct int 20 should pass through unchanged."""
        params = {"period": 20}
        result = coerce_compute_params(SMA(), params)
        assert result["period"] == 20
        assert isinstance(result["period"], int)

    def test_str_passthrough(self):
        """String params (like col_a) should remain strings."""
        params = {"col_a": "close", "col_b": "vol", "exponent": 1.0}
        result = coerce_compute_params(PowerRatio(), params)
        assert result["col_a"] == "close"
        assert isinstance(result["col_a"], str)

    def test_original_dict_not_mutated(self):
        """Coercion should return a new dict, not mutate the original."""
        params = {"col_a": "a", "col_b": "b", "exponent": "0.5"}
        coerce_compute_params(PowerRatio(), params)
        assert params["exponent"] == "0.5"  # unchanged

    def test_unknown_param_passthrough(self):
        """Params not in the signature should pass through unchanged."""
        params = {"col_a": "a", "col_b": "b", "exponent": 0.5, "extra": "foo"}
        result = coerce_compute_params(PowerRatio(), params)
        assert result["extra"] == "foo"

    def test_coerced_params_work_in_compute(self):
        """End-to-end: string exponent should produce correct compute result."""
        df = pd.DataFrame({"a": [10.0, 20.0], "b": [4.0, 16.0]})
        params = {"col_a": "a", "col_b": "b", "exponent": "0.5"}
        coerced = coerce_compute_params(PowerRatio(), params)
        result = PowerRatio().compute(df, **coerced)
        expected = pd.Series([5.0, 5.0])
        pd.testing.assert_series_equal(result, expected)
