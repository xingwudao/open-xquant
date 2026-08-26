import base64
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import Draft202012Validator, ValidationError

CONTRACT_DIR = Path(__file__).parents[2] / "contracts" / "quant-operators"
REFERENCE_VALIDATOR_PATH = CONTRACT_DIR / "reference_validator_v1.py"


def _load(relative_path: str) -> dict:
    return json.loads((CONTRACT_DIR / relative_path).read_text(encoding="utf-8"))


def _reference_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "quant_operator_reference_validator_v1", REFERENCE_VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference validator: {REFERENCE_VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_validator() -> Draft202012Validator:
    return Draft202012Validator(_load("operator-manifest-v1.schema.json"))


def _binding_validator() -> Draft202012Validator:
    return Draft202012Validator(_load("operator-binding-v1.schema.json"))


def _valid_panel() -> dict:
    return _load("examples/valid/daily-cn-panel.json")


def _valid_manifest() -> dict:
    return _load("examples/valid/equant-ttr-sma.operator.json")


def _valid_binding() -> dict:
    return _load("examples/valid/equant-ttr-sma.binding.json")


def _contract_surface_paths() -> dict[str, Path]:
    return {
        "quant_panel_schema": CONTRACT_DIR / "quant-panel-v1.schema.json",
        "operator_manifest_schema": CONTRACT_DIR
        / "operator-manifest-v1.schema.json",
        "operator_binding_schema": CONTRACT_DIR / "operator-binding-v1.schema.json",
        "reference_validator": REFERENCE_VALIDATOR_PATH,
    }


def _implementation_artifact(tmp_path: Path) -> Path:
    wheel_base64_path = (
        CONTRACT_DIR / "examples/valid/equant_ttr-1.0.0-py3-none-any.whl.b64"
    )
    wheel_path = tmp_path / "equant_ttr-1.0.0-py3-none-any.whl"
    wheel_path.write_bytes(
        base64.b64decode(wheel_base64_path.read_bytes().strip(), validate=True)
    )
    return wheel_path


def _validate_binding_fixture(
    validator: ModuleType,
    tmp_path: Path,
    *,
    binding: dict | None = None,
    manifest: dict | None = None,
    manifest_path: Path | None = None,
    source_root: Path | None = None,
    implementation_artifact_path: Path | None = None,
    contract_surface_paths: dict[str, Path] | None = None,
) -> None:
    validator.validate_operator_binding(
        binding or _valid_binding(),
        manifest or _valid_manifest(),
        manifest_path
        or CONTRACT_DIR / "examples/valid/equant-ttr-sma.operator.json",
        source_root or CONTRACT_DIR,
        implementation_artifact_path or _implementation_artifact(tmp_path),
        contract_surface_paths or _contract_surface_paths(),
    )


def test_frozen_schemas_are_valid_draft_2020_12() -> None:
    for filename in (
        "quant-panel-v1.schema.json",
        "operator-manifest-v1.schema.json",
        "operator-binding-v1.schema.json",
    ):
        Draft202012Validator.check_schema(_load(filename))


def test_daily_cn_panel_satisfies_quant_panel_schema() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    Draft202012Validator(_load("quant-panel-v1.schema.json")).validate(panel)
    _reference_validator().validate_quant_panel(panel)


def test_quant_panel_requires_the_full_ordered_primary_key() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["primary_key"] = []
    validator = Draft202012Validator(_load("quant-panel-v1.schema.json"))
    with pytest.raises(ValidationError):
        validator.validate(panel)


def test_quant_panel_rejects_duplicate_column_names() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["columns"].append(dict(panel["columns"][0]))
    with pytest.raises(ValueError, match="duplicate QuantPanel column"):
        _reference_validator().validate_quant_panel(panel)


def test_quant_panel_requires_declared_required_columns_in_every_record() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["records"][0].pop("close")
    with pytest.raises(ValueError, match="missing required QuantPanel column"):
        _reference_validator().validate_quant_panel(panel)


def test_quant_panel_allows_json_null_for_a_present_required_missing_value() -> None:
    panel = _valid_panel()
    panel["records"][0]["close"] = None
    _reference_validator().validate_quant_panel(panel)


def test_quant_panel_allows_python_nan_as_adapter_missing_value() -> None:
    panel = _valid_panel()
    panel["records"][0]["close"] = float("nan")
    _reference_validator().validate_quant_panel(panel)


@pytest.mark.parametrize("infinite_value", [float("inf"), float("-inf")])
def test_quant_panel_does_not_treat_infinity_as_a_missing_value(
    infinite_value: float,
) -> None:
    panel = _valid_panel()
    panel["records"][0]["close"] = infinite_value
    with pytest.raises(ValueError, match="invalid QuantPanel value"):
        _reference_validator().validate_quant_panel(panel)


class _CoercibleNan:
    def __float__(self) -> float:
        return float("nan")


@pytest.mark.parametrize(
    "invalid_value",
    [_CoercibleNan(), 10**10000],
    ids=["coercible-object", "huge-integer"],
)
def test_quant_panel_does_not_coerce_arbitrary_values_for_missing_detection(
    invalid_value: object,
) -> None:
    panel = _valid_panel()
    panel["records"][0]["close"] = invalid_value
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match="invalid QuantPanel value",
    ):
        validator.validate_quant_panel(panel)


def test_quant_panel_absent_required_key_is_not_a_missing_value() -> None:
    panel = _valid_panel()
    del panel["records"][0]["close"]
    validator = _reference_validator()
    with pytest.raises(validator.ContractValidationError, match="missing required"):
        validator.validate_quant_panel(panel)


def test_quant_panel_accepts_declared_null_warmup_values() -> None:
    panel = _valid_panel()
    assert {column["name"] for column in panel["columns"]} == {"close", "sma_3"}
    assert panel["records"][0]["sma_3"] is None
    _reference_validator().validate_quant_panel(panel)


@pytest.mark.parametrize(
    ("dtype", "invalid_value"),
    [
        ("boolean", 1),
        ("int64", True),
        ("float64", True),
        ("string", 1),
        ("date", "2024-13-40"),
        ("datetime", "2024-01-02"),
    ],
)
def test_quant_panel_rejects_values_outside_declared_dtype(
    dtype: str, invalid_value: object
) -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["columns"] = [{"name": "value", "dtype": dtype, "required": True}]
    for record in panel["records"]:
        record.pop("close")
        record.pop("sma_3")
        record["value"] = invalid_value
    with pytest.raises(ValueError, match="invalid QuantPanel value"):
        _reference_validator().validate_quant_panel(panel)


@pytest.mark.parametrize(
    ("dtype", "valid_value"),
    [
        ("boolean", True),
        ("int64", 1),
        ("float64", 1.5),
        ("string", "value"),
        ("date", "2024-01-02"),
        ("datetime", "2024-01-02T15:00:00+08:00"),
    ],
)
def test_quant_panel_accepts_each_declared_dtype(
    dtype: str, valid_value: object
) -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["columns"] = [{"name": "value", "dtype": dtype, "required": True}]
    for record in panel["records"]:
        record.pop("close")
        record.pop("sma_3")
        record["value"] = valid_value
    _reference_validator().validate_quant_panel(panel)


def test_quant_panel_rejects_int64_overflow() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["columns"] = [{"name": "value", "dtype": "int64", "required": True}]
    for record in panel["records"]:
        record.pop("close")
        record.pop("sma_3")
        record["value"] = 2**63
    with pytest.raises(ValueError, match="invalid QuantPanel value"):
        _reference_validator().validate_quant_panel(panel)


def test_equant_sma_satisfies_operator_manifest_schema() -> None:
    manifest = _valid_manifest()
    _manifest_validator().validate(manifest)
    _reference_validator().validate_operator_manifest(manifest)


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


def test_manifest_requires_sorted_input_declaration() -> None:
    manifest = _valid_manifest()
    manifest["input"].pop("requires_sorted_input", None)
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_accepts_explicit_unique_sort_order_when_required() -> None:
    manifest = _valid_manifest()
    manifest["input"].update(
        {
            "requires_sorted_input": True,
            "required_sort_order": ["date", "code"],
        }
    )
    _manifest_validator().validate(manifest)


def test_manifest_requires_sort_order_when_sorted_input_is_required() -> None:
    manifest = _valid_manifest()
    manifest["input"]["requires_sorted_input"] = True
    manifest["input"].pop("required_sort_order", None)
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_forbids_sort_order_when_sorted_input_is_not_required() -> None:
    manifest = _valid_manifest()
    manifest["input"].update(
        {
            "requires_sorted_input": False,
            "required_sort_order": ["date", "code"],
        }
    )
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_rejects_duplicate_required_sort_columns() -> None:
    manifest = _valid_manifest()
    manifest["input"].update(
        {
            "requires_sorted_input": True,
            "required_sort_order": ["date", "date"],
        }
    )
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize("sort_key", ["mystery", "volume"])
def test_reference_validator_rejects_nonrequired_sort_order_key(
    sort_key: str,
) -> None:
    manifest = _valid_manifest()
    manifest["input"].update(
        {
            "requires_sorted_input": True,
            "required_sort_order": [sort_key],
            "optional_columns": ["volume"],
        }
    )
    validator = _reference_validator()
    with pytest.raises(validator.ContractValidationError, match="required sort key"):
        validator.validate_operator_manifest(manifest)


def test_reference_validator_accepts_executable_required_sort_order() -> None:
    manifest = _valid_manifest()
    manifest["input"].update(
        {
            "requires_sorted_input": True,
            "required_sort_order": ["date", "code", "close"],
        }
    )
    _reference_validator().validate_operator_manifest(manifest)


@pytest.mark.parametrize(
    "field",
    [
        "required_columns",
        "optional_columns",
        "supported_dtypes",
        "required_context",
    ],
)
def test_manifest_schema_rejects_duplicate_input_list_items(field: str) -> None:
    manifest = _valid_manifest()
    values = manifest["input"][field]
    duplicate = values[0] if values else "close"
    values.extend([duplicate, duplicate] if not values else [duplicate])
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize("field", ["required_columns", "optional_columns"])
def test_reference_validator_rejects_duplicate_input_columns(field: str) -> None:
    manifest = _valid_manifest()
    manifest["input"][field] = ["close", "close"]
    with pytest.raises(ValueError, match="duplicate input column"):
        _reference_validator().validate_operator_manifest(manifest)


def test_reference_validator_rejects_overlapping_input_columns() -> None:
    manifest = _valid_manifest()
    manifest["input"]["optional_columns"] = ["close"]
    with pytest.raises(ValueError, match="required and optional input columns overlap"):
        _reference_validator().validate_operator_manifest(manifest)


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


@pytest.mark.parametrize(
    ("parameter_type", "default", "constraints"),
    [
        ("integer", 20, {"enum": [10]}),
        ("integer", 0, {"minimum": 1}),
        ("string", "ABC", {"pattern": "^[a-z]+$"}),
        ("string", "a", {"min_length": 2}),
        ("array", [1], {"min_items": 2}),
    ],
)
def test_reference_validator_rejects_default_that_violates_constraints(
    parameter_type: str, default: object, constraints: dict
) -> None:
    manifest = _valid_manifest()
    definition = manifest["parameters"]["window"]
    definition.update(
        {"type": parameter_type, "default": default, "constraints": constraints}
    )
    with pytest.raises(ValueError, match="default for parameter 'window'"):
        _reference_validator().validate_operator_manifest(manifest)


@pytest.mark.parametrize(
    ("parameter_type", "constraint"),
    [
        ("integer", {"pattern": "^[0-9]+$"}),
        ("string", {"minimum": 1}),
        ("array", {"min_length": 1}),
        ("boolean", {"maximum": 1}),
        ("object", {"min_items": 1}),
    ],
)
def test_reference_validator_rejects_constraint_for_wrong_parameter_type(
    parameter_type: str, constraint: dict
) -> None:
    manifest = _valid_manifest()
    definition = manifest["parameters"]["window"]
    defaults = {
        "integer": 1,
        "string": "one",
        "array": [1],
        "boolean": True,
        "object": {"value": 1},
    }
    definition.update(
        {
            "type": parameter_type,
            "default": defaults[parameter_type],
            "constraints": constraint,
        }
    )
    with pytest.raises(ValueError, match="not valid for parameter type"):
        _reference_validator().validate_operator_manifest(manifest)


@pytest.mark.parametrize(
    ("parameter_type", "constraints"),
    [
        ("integer", {"minimum": 2, "maximum": 1}),
        ("number", {"exclusive_minimum": 1, "maximum": 1}),
        ("string", {"min_length": 2, "max_length": 1}),
        ("array", {"min_items": 2, "max_items": 1}),
    ],
)
def test_reference_validator_rejects_conflicting_parameter_constraints(
    parameter_type: str, constraints: dict
) -> None:
    manifest = _valid_manifest()
    definition = manifest["parameters"]["window"]
    defaults = {"integer": 1, "number": 1.0, "string": "x", "array": [1]}
    definition.update(
        {
            "type": parameter_type,
            "default": defaults[parameter_type],
            "constraints": constraints,
        }
    )
    with pytest.raises(ValueError, match="conflicting constraints"):
        _reference_validator().validate_operator_manifest(manifest)


def test_reference_validator_rejects_invalid_parameter_pattern() -> None:
    manifest = _valid_manifest()
    definition = manifest["parameters"]["window"]
    definition.update(
        {"type": "string", "default": "x", "constraints": {"pattern": "["}}
    )
    with pytest.raises(ValueError, match="invalid pattern"):
        _reference_validator().validate_operator_manifest(manifest)


def test_operator_request_accepts_known_parameter_satisfying_constraints() -> None:
    manifest = _valid_manifest()
    _reference_validator().validate_operator_request_parameters(
        manifest, {"window": 10}
    )


def test_operator_request_rejects_unknown_parameter() -> None:
    manifest = _valid_manifest()
    with pytest.raises(ValueError, match="unknown request parameter"):
        _reference_validator().validate_operator_request_parameters(
            manifest, {"window": 10, "mystery": True}
        )


def test_operator_request_rejects_missing_required_parameter() -> None:
    manifest = _valid_manifest()
    manifest["parameters"]["window"]["required"] = True
    with pytest.raises(ValueError, match="missing required request parameter"):
        _reference_validator().validate_operator_request_parameters(manifest, {})


@pytest.mark.parametrize(
    ("parameter_type", "default", "invalid_value"),
    [
        ("integer", 1, True),
        ("number", 1.0, True),
        ("boolean", True, 1),
        ("string", "one", 1),
        ("array", [1], {}),
        ("object", {"value": 1}, []),
    ],
)
def test_operator_request_rejects_value_with_wrong_declared_type(
    parameter_type: str, default: object, invalid_value: object
) -> None:
    manifest = _valid_manifest()
    definition = manifest["parameters"]["window"]
    definition.update(
        {"type": parameter_type, "default": default, "constraints": {}}
    )
    with pytest.raises(ValueError, match="invalid value for request parameter"):
        _reference_validator().validate_operator_request_parameters(
            manifest, {"window": invalid_value}
        )


@pytest.mark.parametrize(
    ("parameter_type", "default", "constraints", "invalid_value"),
    [
        ("integer", 2, {"enum": [2, 3]}, 4),
        ("integer", 2, {"minimum": 2}, 1),
        ("number", 2.0, {"exclusive_minimum": 1.0}, 1.0),
        ("string", "abc", {"pattern": "^[a-z]+$"}, "ABC"),
        ("string", "ab", {"min_length": 2, "max_length": 3}, "a"),
        ("array", [1, 2], {"min_items": 2, "max_items": 3}, [1]),
    ],
)
def test_operator_request_rejects_value_that_violates_constraints(
    parameter_type: str,
    default: object,
    constraints: dict,
    invalid_value: object,
) -> None:
    manifest = _valid_manifest()
    definition = manifest["parameters"]["window"]
    definition.update(
        {
            "type": parameter_type,
            "default": default,
            "constraints": constraints,
        }
    )
    with pytest.raises(ValueError, match="request parameter 'window'"):
        _reference_validator().validate_operator_request_parameters(
            manifest, {"window": invalid_value}
        )


def test_manifest_accepts_required_seed_bound_to_integer_parameter() -> None:
    manifest = _valid_manifest()
    manifest["parameters"]["seed"] = {
        "type": "integer",
        "default": 0,
        "required": True,
        "constraints": {"minimum": 0},
        "unit": "seed",
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }
    manifest["determinism"].update(
        {"random_seed_required": True, "seed_parameter": "seed"}
    )
    _manifest_validator().validate(manifest)
    _reference_validator().validate_operator_manifest(manifest)


def test_manifest_requires_seed_parameter_when_random_seed_is_required() -> None:
    manifest = _valid_manifest()
    manifest["determinism"]["random_seed_required"] = True
    manifest["determinism"].pop("seed_parameter", None)
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_forbids_seed_parameter_without_required_random_seed() -> None:
    manifest = _valid_manifest()
    manifest["determinism"].update(
        {"random_seed_required": False, "seed_parameter": "seed"}
    )
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_reference_validator_rejects_unknown_seed_parameter() -> None:
    manifest = _valid_manifest()
    manifest["determinism"].update(
        {"random_seed_required": True, "seed_parameter": "seed"}
    )
    with pytest.raises(ValueError, match="unknown seed parameter"):
        _reference_validator().validate_operator_manifest(manifest)


def test_reference_validator_rejects_noninteger_seed_parameter() -> None:
    manifest = _valid_manifest()
    manifest["parameters"]["seed"] = {
        "type": "number",
        "default": 0.0,
        "required": True,
        "constraints": {"minimum": 0},
        "unit": "seed",
        "affects_warmup": False,
        "affects_output_fields": False,
        "affects_causality": False,
        "affects_availability": False,
    }
    manifest["determinism"].update(
        {"random_seed_required": True, "seed_parameter": "seed"}
    )
    with pytest.raises(ValueError, match="seed parameter must have type 'integer'"):
        _reference_validator().validate_operator_manifest(manifest)


def test_manifest_keeps_manifest_digest_in_external_binding() -> None:
    manifest = _valid_manifest()
    manifest["implementation"].pop("manifest_digest", None)
    manifest["implementation"].setdefault(
        "source_files", ["examples/valid/provider-source/ettr.py"]
    )
    _manifest_validator().validate(manifest)

    manifest["implementation"]["manifest_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_requires_nonempty_source_files() -> None:
    manifest = _valid_manifest()
    manifest["implementation"].pop("source_files", None)
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)

    manifest["implementation"]["source_files"] = []
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_manifest_rejects_duplicate_source_files() -> None:
    manifest = _valid_manifest()
    manifest["implementation"]["source_files"] = ["src/ettr.py", "src/ettr.py"]
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize(
    "invalid_path",
    [
        "/src/ettr.py",
        "../src/ettr.py",
        "src/../ettr.py",
        "./src/ettr.py",
        "src\\ettr.py",
        "src//ettr.py",
    ],
)
def test_manifest_rejects_nonrelative_posix_source_file(invalid_path: str) -> None:
    manifest = _valid_manifest()
    manifest["implementation"]["source_files"] = [invalid_path]
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


@pytest.mark.parametrize(
    "source_commit",
    [
        "git-sha1:0123456789abcdef0123456789abcdef01234567",
        "git-sha256:" + "0123456789abcdef" * 4,
    ],
)
def test_manifest_accepts_full_algorithm_prefixed_source_commit(
    source_commit: str,
) -> None:
    manifest = _valid_manifest()
    manifest["implementation"]["source_commit"] = source_commit
    _manifest_validator().validate(manifest)


@pytest.mark.parametrize(
    "source_commit",
    [
        "0123456",
        "0123456789abcdef0123456789abcdef01234567",
        "git-sha1:0123456",
        "git-sha1:0123456789abcdef0123456789abcdef0123456",
        "git-sha256:" + "0" * 63,
        "git-sha1:0123456789ABCDEF0123456789abcdef01234567",
    ],
)
def test_manifest_rejects_short_unprefixed_or_nonlowercase_source_commit(
    source_commit: str,
) -> None:
    manifest = _valid_manifest()
    manifest["implementation"]["source_commit"] = source_commit
    with pytest.raises(ValidationError):
        _manifest_validator().validate(manifest)


def test_sha256_file_matches_known_raw_byte_vector(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"abc")
    assert _reference_validator().sha256_file(target) == (
        "sha256:ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_source_tree_digest_matches_known_vector_and_sorts_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "a.txt").write_bytes(b"alpha\n")
    (tmp_path / "src" / "b.py").write_bytes(b"beta\n")
    expected = (
        "sha256:76b8e9c973a22ecc7420b637c53c6b01"
        "342d2852ab290217ba708e48efcc9658"
    )
    validator = _reference_validator()
    assert validator.sha256_source_tree(
        tmp_path, ["src/b.py", "a.txt"]
    ) == expected
    assert validator.sha256_source_tree(
        tmp_path, ["a.txt", "src/b.py"]
    ) == expected


def test_source_tree_digest_rejects_unsafe_or_duplicate_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"alpha\n")
    validator = _reference_validator()
    with pytest.raises(ValueError, match="relative POSIX"):
        validator.sha256_source_tree(tmp_path, ["../a.txt"])
    with pytest.raises(ValueError, match="duplicate source file"):
        validator.sha256_source_tree(tmp_path, ["a.txt", "a.txt"])


def test_operator_binding_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(_load("operator-binding-v1.schema.json"))


def test_operator_binding_fixture_satisfies_schema() -> None:
    _binding_validator().validate(
        _load("examples/valid/equant-ttr-sma.binding.json")
    )


def test_operator_binding_requires_reference_validator_pin() -> None:
    binding = _load("examples/valid/equant-ttr-sma.binding.json")
    del binding["contract_surface"]["reference_validator"]
    with pytest.raises(ValidationError):
        _binding_validator().validate(binding)


def test_operator_binding_contract_surface_pins_exact_artifact_bytes() -> None:
    binding = _load("examples/valid/equant-ttr-sma.binding.json")
    validator = _reference_validator()
    artifact_paths = {
        "quant_panel_schema": CONTRACT_DIR / "quant-panel-v1.schema.json",
        "operator_manifest_schema": CONTRACT_DIR
        / "operator-manifest-v1.schema.json",
        "operator_binding_schema": CONTRACT_DIR / "operator-binding-v1.schema.json",
        "reference_validator": REFERENCE_VALIDATOR_PATH,
    }

    assert binding["surface_release"] == "1.0.0"
    assert set(binding["contract_surface"]) == set(artifact_paths)
    for artifact, path in artifact_paths.items():
        assert binding["contract_surface"][artifact] == {
            "release": "1.0.0",
            "digest": validator.sha256_file(path),
        }


def test_operator_binding_semantics_accept_valid_exact_artifacts(
    tmp_path: Path,
) -> None:
    _validate_binding_fixture(_reference_validator(), tmp_path)


def test_operator_binding_manifest_artifact_rejects_different_json_object(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest()
    different_manifest = _valid_manifest()
    different_manifest["semantic_name"] = "DIFFERENT_SMA"
    _manifest_validator().validate(different_manifest)
    manifest_path = tmp_path / "different-operator.json"
    manifest_path.write_text(json.dumps(different_manifest), encoding="utf-8")
    validator = _reference_validator()
    binding = _valid_binding()
    binding["manifest_digest"] = validator.sha256_file(manifest_path)

    with pytest.raises(
        validator.ContractValidationError,
        match=(
            "operator binding mismatch: manifest artifact: "
            "manifest object does not match manifest artifact"
        ),
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            binding=binding,
            manifest=manifest,
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("manifest_value", "artifact_value"),
    [
        pytest.param(False, 0, id="false-versus-zero"),
        pytest.param(True, 1, id="true-versus-one"),
    ],
)
def test_operator_binding_manifest_artifact_review_rejects_bool_number_collision(
    manifest_value: bool,
    artifact_value: int,
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest()
    manifest["mutates_input"] = manifest_value
    artifact_manifest = _valid_manifest()
    artifact_manifest["mutates_input"] = artifact_value
    manifest_path = tmp_path / "wrong-json-type-operator.json"
    manifest_path.write_text(json.dumps(artifact_manifest), encoding="utf-8")
    validator = _reference_validator()
    binding = _valid_binding()
    binding["manifest_digest"] = validator.sha256_file(manifest_path)

    with pytest.raises(
        validator.ContractValidationError,
        match=(
            "operator binding mismatch: manifest artifact: "
            "manifest object does not match manifest artifact"
        ),
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            binding=binding,
            manifest=manifest,
            manifest_path=manifest_path,
        )


def test_operator_binding_manifest_artifact_review_hashes_single_snapshot(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest()
    first_snapshot = json.dumps(manifest).encode("utf-8")
    replacement_manifest = _valid_manifest()
    replacement_manifest["semantic_name"] = "DIFFERENT_SMA"
    second_snapshot = json.dumps(replacement_manifest).encode("utf-8")
    manifest_path = tmp_path / "mutable-operator.json"
    manifest_path.write_bytes(first_snapshot)

    class MutatingPath:
        def __init__(self) -> None:
            self.calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            if self.calls == 2:
                manifest_path.write_bytes(second_snapshot)
            return str(manifest_path)

    binding = _valid_binding()
    binding["manifest_digest"] = (
        f"sha256:{hashlib.sha256(second_snapshot).hexdigest()}"
    )
    validator = _reference_validator()

    with pytest.raises(
        validator.ContractValidationError,
        match="operator binding mismatch: manifest_digest",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            binding=binding,
            manifest=manifest,
            manifest_path=MutatingPath(),
        )


def test_operator_binding_manifest_artifact_review_normalizes_read_oserror(
    tmp_path: Path,
) -> None:
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match="operator binding mismatch: manifest artifact",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            manifest_path=tmp_path / "missing-operator.json",
        )


@pytest.mark.parametrize(
    "manifest_bytes",
    [
        pytest.param(b"\xff", id="invalid-utf8"),
        pytest.param(b'{"schema_version":', id="malformed-json"),
        pytest.param(
            b'{"schema_version": 1, "schema_version": 1}',
            id="duplicate-object-keys",
        ),
        pytest.param(b'{"value": NaN}', id="nan"),
        pytest.param(b'{"value": Infinity}', id="infinity"),
        pytest.param(b'{"value": -Infinity}', id="negative-infinity"),
    ],
)
def test_operator_binding_manifest_artifact_rejects_non_strict_json(
    manifest_bytes: bytes,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "invalid-operator.json"
    manifest_path.write_bytes(manifest_bytes)
    validator = _reference_validator()
    binding = _valid_binding()
    binding["manifest_digest"] = validator.sha256_file(manifest_path)

    with pytest.raises(
        validator.ContractValidationError,
        match="operator binding mismatch: manifest artifact",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            binding=binding,
            manifest=_valid_manifest(),
            manifest_path=manifest_path,
        )


def test_operator_binding_semantics_validate_matching_manifest_before_provenance(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest()
    manifest["input"]["optional_columns"] = ["close"]
    manifest_path = tmp_path / "invalid-semantic-operator.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validator = _reference_validator()
    binding = _valid_binding()
    binding["manifest_digest"] = validator.sha256_file(manifest_path)
    with pytest.raises(
        validator.ContractValidationError,
        match="required and optional input columns overlap",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            binding=binding,
            manifest=manifest,
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("operator_id", "other.vendor.sma"),
        ("operator_version", "2.0.0"),
        ("distribution", "other-ttr"),
        ("distribution_version", "2.0.0"),
        (
            "source_commit",
            "git-sha1:1111111111111111111111111111111111111111",
        ),
        ("source_tree_digest", "sha256:" + "1" * 64),
        ("implementation_digest", "sha256:" + "1" * 64),
    ],
)
def test_operator_binding_semantics_reject_manifest_identity_mismatch(
    field: str,
    invalid_value: str,
    tmp_path: Path,
) -> None:
    binding = _valid_binding()
    binding[field] = invalid_value
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match=f"operator binding mismatch: {field}",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            binding=binding,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        (
            "schema_id",
            "https://open-xquant.dev/contracts/quant-operators/quant-panel-v1.schema.json",
        ),
        ("schema_release", "1.0.1"),
        ("schema_digest", "sha256:" + "1" * 64),
    ],
)
def test_operator_binding_semantics_reject_legacy_surface_schema_pin_mismatch(
    field: str,
    invalid_value: str,
    tmp_path: Path,
) -> None:
    binding = _valid_binding()
    binding[field] = invalid_value
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match="operator binding mismatch: legacy operator manifest schema pin",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            binding=binding,
        )


def test_operator_binding_semantics_reject_manifest_exact_bytes(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "operator.json"
    manifest_path.write_text(json.dumps(_valid_manifest()), encoding="utf-8")
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match="operator binding mismatch: manifest_digest",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            manifest_path=manifest_path,
        )


def test_operator_binding_semantics_reject_source_tree_exact_bytes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-root"
    source_path = source_root / "examples/valid/provider-source/ettr.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"tampered provider source\n")
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match="operator binding mismatch: source_tree_digest",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            source_root=source_root,
        )


def test_operator_binding_semantics_reject_implementation_exact_bytes(
    tmp_path: Path,
) -> None:
    implementation_path = tmp_path / "tampered.whl"
    implementation_path.write_bytes(b"tampered wheel bytes")
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match="operator binding mismatch: implementation_digest",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            implementation_artifact_path=implementation_path,
        )


@pytest.mark.parametrize(
    "artifact",
    [
        "quant_panel_schema",
        "operator_manifest_schema",
        "operator_binding_schema",
        "reference_validator",
    ],
)
def test_operator_binding_semantics_reject_contract_surface_artifact_exact_bytes(
    artifact: str,
    tmp_path: Path,
) -> None:
    artifact_paths = _contract_surface_paths()
    tampered_path = tmp_path / f"{artifact}.tampered"
    tampered_path.write_bytes(b"tampered contract artifact\n")
    artifact_paths[artifact] = tampered_path
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match=rf"operator binding mismatch: contract_surface\.{artifact}\.digest",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            contract_surface_paths=artifact_paths,
        )


@pytest.mark.parametrize(
    "artifact",
    [
        "quant_panel_schema",
        "operator_manifest_schema",
        "operator_binding_schema",
        "reference_validator",
    ],
)
def test_operator_binding_semantics_reject_contract_surface_release_mismatch(
    artifact: str,
    tmp_path: Path,
) -> None:
    binding = _valid_binding()
    binding["contract_surface"][artifact]["release"] = "1.0.1"
    if artifact == "operator_manifest_schema":
        binding["schema_release"] = "1.0.1"
    validator = _reference_validator()
    with pytest.raises(
        validator.ContractValidationError,
        match=rf"operator binding mismatch: contract_surface\.{artifact}\.release",
    ):
        _validate_binding_fixture(
            validator,
            tmp_path,
            binding=binding,
        )


def test_valid_digest_fixtures_bind_exact_artifact_bytes(tmp_path: Path) -> None:
    manifest_path = CONTRACT_DIR / "examples/valid/equant-ttr-sma.operator.json"
    schema_path = CONTRACT_DIR / "operator-manifest-v1.schema.json"
    wheel_base64_path = (
        CONTRACT_DIR / "examples/valid/equant_ttr-1.0.0-py3-none-any.whl.b64"
    )
    binding = _load("examples/valid/equant-ttr-sma.binding.json")
    manifest = _valid_manifest()
    validator = _reference_validator()

    assert binding["schema_release"] == "1.0.0"
    assert binding["schema_digest"] == validator.sha256_file(schema_path)
    assert binding["manifest_digest"] == validator.sha256_file(manifest_path)
    assert manifest["implementation"]["source_tree_digest"] == (
        validator.sha256_source_tree(
            CONTRACT_DIR, manifest["implementation"]["source_files"]
        )
    )

    wheel_bytes = base64.b64decode(
        wheel_base64_path.read_bytes().strip(), validate=True
    )
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel:
        assert "equant_ttr-1.0.0.dist-info/WHEEL" in wheel.namelist()
    wheel_path = tmp_path / "equant_ttr-1.0.0-py3-none-any.whl"
    wheel_path.write_bytes(wheel_bytes)
    assert manifest["implementation"]["implementation_digest"] == (
        validator.sha256_file(wheel_path)
    )
    assert binding["source_tree_digest"] == manifest["implementation"][
        "source_tree_digest"
    ]
    assert binding["implementation_digest"] == manifest["implementation"][
        "implementation_digest"
    ]


def test_duplicate_panel_keys_are_rejected_by_reference_contract_check() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["records"].append(dict(panel["records"][0]))
    with pytest.raises(ValueError, match="duplicate QuantPanel key"):
        _reference_validator().validate_quant_panel(panel)


def test_undeclared_panel_fields_are_rejected_by_reference_contract_check() -> None:
    panel = _load("examples/valid/daily-cn-panel.json")
    panel["records"][0]["unapproved_field"] = 1
    with pytest.raises(ValueError, match="undeclared QuantPanel field"):
        _reference_validator().validate_quant_panel(panel)
