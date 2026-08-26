import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError


CONTRACT_DIR = Path(__file__).parents[2] / "contracts" / "quant-operators"


def _load(relative_path: str) -> dict:
    return json.loads((CONTRACT_DIR / relative_path).read_text(encoding="utf-8"))


def _validate_unique_panel_keys(panel: dict) -> None:
    seen: set[tuple[str, str]] = set()
    for record in panel["records"]:
        key = (record["date"], record["code"])
        if key in seen:
            raise ValueError(f"duplicate QuantPanel key: {key!r}")
        seen.add(key)


def test_frozen_schemas_are_valid_draft_2020_12() -> None:
    for filename in ("quant-panel-v1.schema.json", "operator-manifest-v1.schema.json"):
        Draft202012Validator.check_schema(_load(filename))


def test_daily_cn_panel_satisfies_quant_panel_schema() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    Draft202012Validator(_load("quant-panel-v1.schema.json")).validate(panel)
    _validate_unique_panel_keys(panel)


def test_quant_panel_requires_the_full_ordered_primary_key() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["primary_key"] = []
    validator = Draft202012Validator(_load("quant-panel-v1.schema.json"))
    with pytest.raises(ValidationError):
        validator.validate(panel)


def test_equant_sma_satisfies_operator_manifest_schema() -> None:
    manifest = _load("examples/valid/equant-ttr-sma.operator.json")
    Draft202012Validator(_load("operator-manifest-v1.schema.json")).validate(manifest)


def test_uppercase_distribution_is_rejected() -> None:
    invalid = _load("examples/invalid/uppercase-distribution.operator.json")
    validator = Draft202012Validator(_load("operator-manifest-v1.schema.json"))
    with pytest.raises(ValidationError, match="does not match"):
        validator.validate(invalid)


def test_duplicate_panel_keys_are_rejected_by_reference_contract_check() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["records"].append(dict(panel["records"][0]))
    with pytest.raises(ValueError, match="duplicate QuantPanel key"):
        _validate_unique_panel_keys(panel)
