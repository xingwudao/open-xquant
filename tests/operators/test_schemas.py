from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from oxq.operators.errors import DuplicateKeyError, InvalidPanelError
from oxq.operators.panel import validate_serialized_quant_panel

SCHEMA_DIR = Path("contracts/quant-operators")


def test_published_json_schemas_are_valid_draft_2020_12() -> None:
    for name in ("operator-manifest-v1.schema.json", "quant-panel-v1.schema.json"):
        payload = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(payload)
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_operator_schema_accepts_contract_manifest(valid_manifest_payload) -> None:
    schema = json.loads((SCHEMA_DIR / "operator-manifest-v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(valid_manifest_payload)


@pytest.mark.parametrize(
    "warmup",
    [
        {"kind": "fixed", "rows": 1.0},
        {"kind": "parameter", "parameter": "period", "offset": 0.0},
    ],
    ids=["fixed-rows", "parameter-offset"],
)
def test_operator_schema_accepts_integral_float_warmup_values(valid_manifest_payload, warmup) -> None:
    schema = json.loads((SCHEMA_DIR / "operator-manifest-v1.schema.json").read_text(encoding="utf-8"))
    payload = {**valid_manifest_payload, "outputs": {**valid_manifest_payload["outputs"], "warmup": warmup}}

    Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize("operator_id", ["vendor..sma", "vendor.-.sma", "vendor._.sma"])
def test_operator_schema_rejects_invalid_operator_id_segments(valid_manifest_payload, operator_id) -> None:
    schema = json.loads((SCHEMA_DIR / "operator-manifest-v1.schema.json").read_text(encoding="utf-8"))
    payload = {**valid_manifest_payload, "operator_id": operator_id}

    assert not Draft202012Validator(schema).is_valid(payload)


@pytest.mark.parametrize("min_history", [0, 2])
def test_operator_schema_requires_exactly_one_history_row_for_cross_section_scope(
    valid_manifest_payload,
    min_history,
) -> None:
    schema = json.loads((SCHEMA_DIR / "operator-manifest-v1.schema.json").read_text(encoding="utf-8"))
    payload = {
        **valid_manifest_payload,
        "execution_scope": "cross_section",
        "inputs": {**valid_manifest_payload["inputs"], "min_history": min_history},
    }

    assert not Draft202012Validator(schema).is_valid(payload)


@pytest.mark.parametrize(
    "distribution",
    [
        "Fake-Quant-Operators",
        "fake_quant_operators",
        "fake.quant.operators",
        "fake--quant-operators",
        "fake-quant-operators-",
    ],
)
def test_operator_schema_rejects_noncanonical_python_distribution_names(
    valid_manifest_payload,
    distribution,
) -> None:
    schema = json.loads((SCHEMA_DIR / "operator-manifest-v1.schema.json").read_text(encoding="utf-8"))
    payload = {**valid_manifest_payload, "distribution": distribution}

    assert not Draft202012Validator(schema).is_valid(payload)


def test_quant_panel_schema_accepts_serialized_daily_panel() -> None:
    schema = json.loads((SCHEMA_DIR / "quant-panel-v1.schema.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "context": {
            "timezone": "Asia/Shanghai",
            "calendar": "XSHG",
            "frequency": "1d",
            "timestamp_semantics": "session_date",
            "currency": "CNY",
            "price_adjustment": "forward_adjusted",
            "data_version": "fixture-v1",
            "source": "fake",
        },
        "rows": [{"date": "2026-01-05", "code": "000001.SZ", "close": 10.0}],
    }
    Draft202012Validator(schema).validate(payload)
    validate_serialized_quant_panel(payload)


def test_serialized_quant_panel_validation_rejects_duplicate_keys() -> None:
    payload = {
        "schema_version": 1,
        "context": {
            "timezone": "Asia/Shanghai",
            "calendar": "XSHG",
            "frequency": "1d",
            "timestamp_semantics": "session_date",
            "currency": "CNY",
            "price_adjustment": "forward_adjusted",
            "data_version": "fixture-v1",
            "source": "fake",
        },
        "rows": [
            {"date": "2026-01-05", "code": "000001.SZ", "close": 10.0},
            {"date": "2026-01-05", "code": "000001.SZ", "close": 11.0},
        ],
    }

    with pytest.raises(DuplicateKeyError, match="duplicate"):
        validate_serialized_quant_panel(payload)


def test_serialized_quant_panel_validation_enforces_date_formats() -> None:
    payload = {
        "schema_version": 1,
        "context": {
            "timezone": "Asia/Shanghai",
            "calendar": "XSHG",
            "frequency": "1d",
            "timestamp_semantics": "session_date",
            "currency": "CNY",
            "price_adjustment": "forward_adjusted",
            "data_version": "fixture-v1",
            "source": "fake",
        },
        "rows": [{"date": "not-a-date", "code": "000001.SZ"}],
    }

    with pytest.raises(InvalidPanelError, match="date"):
        validate_serialized_quant_panel(payload)


def test_serialized_intraday_panel_requires_rfc3339_date_time() -> None:
    payload = {
        "schema_version": 1,
        "context": {
            "timezone": "UTC",
            "calendar": "XNYS",
            "frequency": "1min",
            "timestamp_semantics": "bar_close",
            "currency": "USD",
            "price_adjustment": "raw",
            "data_version": "fixture-v1",
            "source": "fake",
        },
        "rows": [{"date": "2026-01-01 00:00:00+00:00", "code": "AAPL"}],
    }

    with pytest.raises(InvalidPanelError, match="date"):
        validate_serialized_quant_panel(payload)
