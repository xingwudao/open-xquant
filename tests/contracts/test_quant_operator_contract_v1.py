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


def _validate_declared_panel_fields(panel: dict) -> None:
    declared_fields = {"date", "code"}
    declared_fields.update(column["name"] for column in panel["columns"])
    for record in panel["records"]:
        undeclared_fields = set(record).difference(declared_fields)
        if undeclared_fields:
            field = sorted(undeclared_fields)[0]
            raise ValueError(f"undeclared QuantPanel field: {field!r}")


def _manifest_validator() -> Draft202012Validator:
    return Draft202012Validator(_load("operator-manifest-v1.schema.json"))


def _valid_manifest() -> dict:
    return _load("examples/valid/equant-ttr-sma.operator.json")


def test_frozen_schemas_are_valid_draft_2020_12() -> None:
    for filename in ("quant-panel-v1.schema.json", "operator-manifest-v1.schema.json"):
        Draft202012Validator.check_schema(_load(filename))


def test_daily_cn_panel_satisfies_quant_panel_schema() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    Draft202012Validator(_load("quant-panel-v1.schema.json")).validate(panel)
    _validate_unique_panel_keys(panel)
    _validate_declared_panel_fields(panel)


def test_quant_panel_requires_the_full_ordered_primary_key() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["primary_key"] = []
    validator = Draft202012Validator(_load("quant-panel-v1.schema.json"))
    with pytest.raises(ValidationError):
        validator.validate(panel)


def test_equant_sma_satisfies_operator_manifest_schema() -> None:
    _manifest_validator().validate(_valid_manifest())


def test_uppercase_distribution_is_rejected() -> None:
    invalid = _load("examples/invalid/uppercase-distribution.operator.json")
    validator = Draft202012Validator(_load("operator-manifest-v1.schema.json"))
    with pytest.raises(ValidationError, match="does not match"):
        validator.validate(invalid)


@pytest.mark.parametrize(
    ("location", "invalid_version"),
    [
        (("operator_version",), "1.0.0-01"),
        (("implementation", "package_version"), "1.0.0-01"),
    ],
)
def test_manifest_rejects_numeric_prerelease_leading_zero(
    location: tuple[str, ...], invalid_version: str
) -> None:
    manifest = _valid_manifest()
    target = manifest
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = invalid_version
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize(
    ("location", "invalid_version"),
    [
        (("operator_version",), "1.0.0\n"),
        (("operator_version",), "1.0.0\r\n"),
        (("implementation", "package_version"), "1.0.0\n"),
        (("implementation", "package_version"), "1.0.0\r\n"),
    ],
)
def test_manifest_rejects_terminal_line_endings_in_versions(
    location: tuple[str, ...], invalid_version: str
) -> None:
    manifest = _valid_manifest()
    target = manifest
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = invalid_version
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("operator_id", "equant.ttr_sma"),
        ("distribution", "eQuant-TTR"),
        ("module", "equant-ttr"),
        ("callable", "sma.value"),
    ],
)
def test_manifest_rejects_invalid_identity_syntax(field: str, invalid_value: str) -> None:
    manifest = _valid_manifest()
    manifest[field] = invalid_value
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("module", "ettr_"),
        ("module", "ettr__core"),
        ("callable", "sma_"),
        ("callable", "sma__fast"),
    ],
)
def test_manifest_rejects_non_strict_snake_identity_segments(
    field: str, invalid_value: str
) -> None:
    manifest = _valid_manifest()
    manifest[field] = invalid_value
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize(
    "location",
    [
        ("mutates_input",),
        ("input", "requires_industry_data"),
        ("input", "requires_market_cap_data"),
        ("input", "requires_fundamental_data"),
        ("availability_depends_on_input",),
    ],
)
def test_manifest_requires_complete_input_governance(location: tuple[str, ...]) -> None:
    manifest = _valid_manifest()
    target = manifest
    for key in location[:-1]:
        target = target[key]
    target.pop(location[-1], None)
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_requires_non_mutating_input() -> None:
    manifest = _valid_manifest()
    manifest["mutates_input"] = True
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_requires_availability_derivation_when_input_dependent() -> None:
    manifest = _valid_manifest()
    manifest["availability_depends_on_input"] = True
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_rejects_explicit_fill_without_value_or_method() -> None:
    manifest = _valid_manifest()
    manifest["output"]["nan_policy"] = "explicit_fill"
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_requires_fill_value_for_constant_explicit_fill() -> None:
    manifest = _valid_manifest()
    manifest["output"].update(
        {"nan_policy": "explicit_fill", "fill_method": "constant"}
    )
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize("fill_method", ["forward_fill", "backward_fill", "interpolate"])
def test_manifest_allows_executable_nonconstant_fill_methods(fill_method: str) -> None:
    manifest = _valid_manifest()
    manifest["output"].update(
        {"nan_policy": "explicit_fill", "fill_method": fill_method}
    )
    _manifest_validator().validate(manifest)


def test_manifest_rejects_unknown_parameter_constraint() -> None:
    manifest = _valid_manifest()
    manifest["parameters"]["window"]["constraints"]["minimun"] = 1
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_rejects_parameter_default_with_wrong_declared_type() -> None:
    manifest = _valid_manifest()
    manifest["parameters"]["window"]["default"] = "20"
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_duplicate_panel_keys_are_rejected_by_reference_contract_check() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["records"].append(dict(panel["records"][0]))
    with pytest.raises(ValueError, match="duplicate QuantPanel key"):
        _validate_unique_panel_keys(panel)


def test_undeclared_panel_fields_are_rejected_by_reference_contract_check() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["records"][0]["unapproved_field"] = 1
    with pytest.raises(ValueError, match="undeclared QuantPanel field"):
        _validate_declared_panel_fields(panel)
