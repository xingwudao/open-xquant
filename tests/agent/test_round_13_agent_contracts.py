from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from oxq.cli.agent_targets import ROLE_TARGETS, discover_agent_roles, render_agent_role_for_target

ROLE_CONTRACT_FIELDS = (
    "role_kind",
    "required_skills",
    "inputs",
    "outputs",
    "forbidden_outputs",
    "ownership_resolution",
)
CONTRACT_HEADING = "## Ownership And Handoff Contract"
RESULT_HANDOFF_TEXT = "On completion or blockage, return a result to the caller"


def _markdown_parts(rendered: str) -> tuple[dict[str, object], str]:
    _, raw_frontmatter, body = rendered.split("---", 2)
    metadata = yaml.safe_load(raw_frontmatter)
    assert isinstance(metadata, dict)
    return metadata, body.lstrip()


def _rendered_prompt(target_id: str, rendered: str) -> str:
    if target_id == "codex":
        payload = tomllib.loads(rendered)
        assert set(payload) == {"name", "description", "developer_instructions"}
        return payload["developer_instructions"]

    frontmatter, body = _markdown_parts(rendered)
    if target_id == "opencode":
        assert set(frontmatter) == {"description", "mode", "permission"}
    else:
        assert set(frontmatter) == {"name", "description"}
    return body


def _rendered_contract(prompt: str) -> dict[str, object]:
    assert prompt.count(CONTRACT_HEADING) == 1
    contract_section = prompt.split(f"{CONTRACT_HEADING}\n\n```yaml\n", 1)[1]
    raw_contract = contract_section.split("\n```", 1)[0]
    contract = yaml.safe_load(raw_contract)
    assert isinstance(contract, dict)
    return contract


def test_router_routes_every_governed_workspace_comparison_to_version_comparator() -> None:
    text = Path("agent/skills/open-xquant/SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "inspect `.open-xquant/workspace.yaml` before choosing a comparison skill" in normalized
    assert "route every comparison request through `compare-strategy-versions`" in normalized
    assert 'including a generic "compare two experiments" request' in normalized


def test_router_uses_legacy_comparator_only_for_non_governed_workspaces() -> None:
    text = Path("agent/skills/open-xquant/SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "report in a legacy, non-governed workspace: use `compare-experiments`" in normalized
    assert (
        '"Compare two experiments": in a version-governed workspace use '
        "`compare-strategy-versions`; only in a legacy, non-governed workspace use "
        "`compare-experiments`"
    ) in normalized


def test_legacy_comparator_refuses_version_governed_workspaces() -> None:
    text = Path("agent/skills/compare-experiments/SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "Do not use this skill in a version-governed workspace" in normalized
    assert "route the request to `compare-strategy-versions`" in normalized


def test_universe_symbol_contract_allows_dotted_exchange_suffixes() -> None:
    text = Path("agent/skills/build-universe/SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "permit dots within normal symbols" in normalized
    assert "`600519.SH` and `000001.SZ`" in normalized
    assert "reject a symbol merely because it contains a dot" in normalized


def test_universe_symbol_contract_rejects_only_path_unsafe_shapes() -> None:
    text = Path("agent/skills/build-universe/SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "path separators (`/` or `\\`)" in normalized
    assert "literal `.` or `..` path components" in normalized
    assert "absolute paths" in normalized
    assert "reject unsafe symbols containing `/`, `\\`, `.`, `..`, or absolute paths" not in normalized


@pytest.mark.parametrize("target_id", ROLE_TARGETS)
def test_every_rendered_role_preserves_ownership_and_handoff_contract(target_id: str) -> None:
    roles = discover_agent_roles(Path.cwd())
    assert roles

    for role in roles:
        expected_contract = {field: role.metadata[field] for field in ROLE_CONTRACT_FIELDS}
        assert all(expected_contract.values()), role.name
        for field in ("required_skills", "inputs", "outputs", "forbidden_outputs"):
            values = expected_contract[field]
            assert isinstance(values, list), (role.name, field)
            assert all(isinstance(value, str) and value.strip() for value in values), (
                role.name,
                field,
            )
        assert expected_contract["ownership_resolution"] == {
            "placeholder_order": "resolve_before_match",
            "overlap_policy": "output_wins_within_declared_output_only",
            "outside_declared_output": "forbidden_still_applies",
        }

        filename, rendered = render_agent_role_for_target(role, target_id)
        expected_suffix = ".toml" if target_id == "codex" else ".md"
        assert filename == f"{role.name}{expected_suffix}"

        prompt = _rendered_prompt(target_id, rendered)
        assert _rendered_contract(prompt) == expected_contract, (target_id, role.name)
        assert RESULT_HANDOFF_TEXT in prompt, (target_id, role.name)
        assert "outputs produced" in prompt, (target_id, role.name)
        assert "next required handoff" in prompt, (target_id, role.name)
        assert "never produce an entry listed under `forbidden_outputs`" in prompt, (
            target_id,
            role.name,
        )
        assert "Resolve all placeholders before matching paths" in prompt
        assert "the declared output wins only for that exact file" in prompt
