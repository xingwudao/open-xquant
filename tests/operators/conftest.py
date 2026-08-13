from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from oxq.operators.types import OperatorContext


@pytest.fixture
def daily_context() -> OperatorContext:
    return OperatorContext(
        timezone="Asia/Shanghai",
        calendar="XSHG",
        frequency="1d",
        timestamp_semantics="session_date",
        currency="CNY",
        price_adjustment="forward_adjusted",
        data_version="fixture-v1",
        source="fake",
        evaluation_time="close_t",
    )


@pytest.fixture
def valid_manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "operator_id": "fake.indicators.sma",
        "operator_version": "1.0.0",
        "semantic_name": "SMA",
        "distribution": "fake-quant-operators",
        "module": "fake_provider.indicators",
        "callable": "sma",
        "execution_scope": "time_series",
        "lifecycle": "stateless",
        "causality": "past_only",
        "availability": {"value": "close_t"},
        "inputs": {
            "required_columns": ["close"],
            "optional_columns": [],
            "dtypes": {"close": ["float64", "float32", "int64"]},
            "min_assets": 1,
            "min_history": 2,
            "requires_complete_cross_section": False,
            "requires_benchmark": False,
            "requires_industry": False,
            "requires_market_cap": False,
            "requires_fundamentals": False,
            "requires_sorted": False,
            "mutates_input": False,
            "missing_value_policy": {"kind": "skip_window"},
        },
        "parameters": {
            "period": {
                "type": "integer",
                "default": 2,
                "required": False,
                "minimum": 1,
                "unit": "bars",
                "affects_warmup": True,
                "affects_output_fields": True,
                "affects_causality": False,
                "affects_availability": False,
            }
        },
        "outputs": {
            "fields": [{"name_template": "sma_{period}", "dtype": "float64"}],
            "alignment": "canonical_order",
            "warmup": {"kind": "parameter", "parameter": "period", "offset": -1},
            "nan_policy": "warmup_only",
            "multiple": False,
        },
    }


@pytest.fixture
def daily_symbol_frames() -> dict[str, pd.DataFrame]:
    dates = pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"], tz="Asia/Shanghai")
    return {
        "000001.SZ": pd.DataFrame({"close": [10.0, 11.0, 12.0], "volume": [100, 110, 120]}, index=dates),
        "600000.SH": pd.DataFrame({"close": [20.0, 18.0, 21.0], "volume": [200, 220, 210]}, index=dates),
    }
