from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path("contracts/quant-operators")


def test_published_json_schemas_are_valid_draft_2020_12() -> None:
    for name in ("operator-manifest-v1.schema.json", "quant-panel-v1.schema.json"):
        payload = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(payload)
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_operator_schema_accepts_contract_manifest(valid_manifest_payload) -> None:
    schema = json.loads((SCHEMA_DIR / "operator-manifest-v1.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(valid_manifest_payload)


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
