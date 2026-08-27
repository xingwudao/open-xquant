"""Environment provider index parsing tests."""

from __future__ import annotations

import pytest
import oxq.operators.environment_index as environment_index

from oxq.operators.environment_index import (
    load_environment_provider,
    parse_exact_provider_requirement,
)


def test_parse_exact_provider_requirement_accepts_only_exact_version() -> None:
    assert parse_exact_provider_requirement("equant-py==1.0.0") == ("equant-py", "1.0.0")

    with pytest.raises(ValueError, match="exact"):
        parse_exact_provider_requirement("equant-py>=1.0.0")


@pytest.mark.parametrize(
    "value",
    [
        " equant-py==1.0.0",
        "equant-py ==1.0.0",
        "equant-py==1.0.0 ",
        "EQuant-Py==1.0.0",
        "equant-py==1.0.0+local",
    ],
)
def test_parse_exact_provider_requirement_rejects_noncanonical_spelling(value: str) -> None:
    with pytest.raises(ValueError, match="exact"):
        parse_exact_provider_requirement(value)


def test_official_index_contains_equant_py_100() -> None:
    provider = load_environment_provider("equant-py", "1.0.0")

    assert provider.provider == "equant-py"
    assert provider.distribution == "equant-core"
    assert provider.version == "1.0.0"
    assert provider.certification_state == "research-certified"
    assert len(provider.operators) == 60
    sma = next(operator for operator in provider.operators if operator.operator_id == "equant.ttr.sma")
    assert sma.operator_version == "1.0.0"
    assert sma.manifest_path == "compat/open_xquant/manifests/equant.ttr.sma.operator.json"
    assert sma.baseline_paths == (
        "compat/open_xquant/numerical_baselines/technical-v1.json",
    )


def test_rejects_non_object_operator_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        environment_index,
        "_load_index_payload",
        lambda: {
            "schema_version": 1,
            "providers": {
                "equant-py": {
                    "1.0.0": {
                        "distribution": "equant-py",
                        "certification_state": "research-certified",
                        "operators": [None],
                        "manifest_digests": {},
                        "baseline_digests": {},
                    }
                }
            },
        },
    )

    with pytest.raises(ValueError, match="official environment operator entry is invalid"):
        load_environment_provider("equant-py", "1.0.0")
