"""Strict packaged contracts for operator distribution and installation."""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from oxq.operators.formats import (
    canonical_json_bytes,
    safe_relative_path,
    sha256_bytes,
    strict_json_object,
)
from oxq.operators.resources import (
    materialize_operator_distribution_profile,
    materialize_operator_install_profile,
)


def _schemas() -> dict[str, dict[str, object]]:
    with materialize_operator_distribution_profile() as distribution:
        with materialize_operator_install_profile() as install:
            return {
                name: json.loads(path.read_text(encoding="utf-8"))
                for name, path in {**distribution, **install}.items()
                if name != "official_providers"
            }


def test_operator_distribution_resources_are_packaged_and_strict() -> None:
    schemas = _schemas()

    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False

    record = schemas["certification_record_v2"]
    assert "target" in record["required"]
    assert "baseline_sets" in record["required"]
    case = record["$defs"]["operator"]["properties"]["baseline_cases"]["items"]
    assert {"baseline_path", "case_index", "case_digest"} <= set(case["required"])


@pytest.mark.parametrize(
    "name",
    ["operator_release", "installed_release"],
)
def test_install_schemas_bound_size_bytes_and_reject_extra_fields(name: str) -> None:
    schema = _schemas()[name]
    serialized = json.dumps(schema, sort_keys=True)
    assert '"size_bytes"' in serialized
    assert '"maximum"' in serialized

    invalid = {key: None for key in schema["required"]}
    invalid["unexpected"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(invalid)


def test_official_provider_map_uses_canonical_provider_name() -> None:
    with materialize_operator_install_profile() as paths:
        provider_map = json.loads(paths["official_providers"].read_text(encoding="utf-8"))
    assert provider_map == {
        "schema_version": 1,
        "providers": {
            "equant-py": {
                "repository": "xingwudao/equant-py",
                "release_asset": "operator-release-v1.json",
            }
        },
    }


def test_shared_strict_formats_reject_ambiguous_json_and_paths() -> None:
    assert strict_json_object(b'{"key": 1}') == {"key": 1}
    assert canonical_json_bytes({"z": 1, "a": "x"}) == b'{"a":"x","z":1}\n'
    assert sha256_bytes(b"open-xquant") == (
        "sha256:2068031c336a3d98dbdfce0325431289d6ef75167af65e043f5a20ae8f1093ec"
    )
    assert safe_relative_path("publication/record.json").as_posix() == "publication/record.json"

    for raw in (b'[]', b'{"key": NaN}', b'{"key": 1, "key": 2}'):
        with pytest.raises(ValueError):
            strict_json_object(raw)
    for path in ("", "/absolute.json", "../escape.json", "one\\two.json", "nul\x00.json"):
        with pytest.raises(ValueError):
            safe_relative_path(path)
