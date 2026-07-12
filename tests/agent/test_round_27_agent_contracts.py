from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
CONTRACTS = (
    ROOT / "agent/skills/select-final-version/SKILL.md",
    ROOT / "agent/roles/oxq-final-selector-worker.md",
)


def _text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").lower().split())


@pytest.mark.parametrize("path", CONTRACTS)
def test_pointer_gate_uses_current_decision_and_transitive_schema_versions(
    path: Path,
) -> None:
    text = _text(path)
    assert "before direct schema-version-5 transitive revalidation" in text
    assert (
        "schema-version-5 decision validation again" in text
        or "run full schema-version-5 decision validation again" in text
    )
    assert "schema-version-3 candidate set" in text or "schema-version-3 candidate-set" in text
    assert "schema-version-3 comparisons" in text or "schema-version-3 comparison" in text
    assert "lineage-v3 validator" in text
    assert "before direct schema-version-4 transitive revalidation" not in text
    assert "schema-version-4 decision validation again" not in text


@pytest.mark.parametrize("path", CONTRACTS)
def test_obsolete_schema_two_result_is_historical_only(path: Path) -> None:
    text = _text(path)
    assert "historical compatibility example only" in text
    assert "every current selector result uses schema version 2" not in text
    assert "current selector result handoffs use schema" in text
    assert "selected result publishes the schema-version-5" in text
