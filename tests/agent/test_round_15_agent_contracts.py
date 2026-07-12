from __future__ import annotations

import json
import re
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
OWNERSHIP_RESOLUTION = {
    "placeholder_order": "resolve_before_match",
    "overlap_policy": "output_wins_within_declared_output_only",
    "outside_declared_output": "forbidden_still_applies",
}
CONFIGURABLE_OUTPUT_ROOTS = {
    "comparison_registry",
    "comparisons_dir",
    "components_dir",
    "conversations_dir",
    "experiment_registry",
    "final_dir",
    "governance_dir",
    "phase_paths.01_brainstorm",
    "phase_paths.02_idea_audit",
    "phase_paths.03_component_authoring",
    "phase_paths.04_spec_build",
    "phase_paths.05_data_inspection",
    "phase_paths.06_spec_audit",
    "phase_paths.07_compile_preview",
    "phase_paths.08_runtime_audit",
    "phase_paths.09_backtests",
    "phase_paths.10_reports",
    "version_root",
}
FINAL_SELECTOR_CONTRACTS = (
    Path("agent/skills/select-final-version/SKILL.md"),
    Path("agent/roles/oxq-final-selector-worker.md"),
)
LINEAGE_CONTRACTS = (
    Path("agent/skills/audit-artifact-lineage/SKILL.md"),
    Path("agent/roles/oxq-lineage-auditor-worker.md"),
)
GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
FULL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _frontmatter(path: Path) -> dict[str, object]:
    metadata = yaml.safe_load(_text(path).split("---", 2)[1])
    assert isinstance(metadata, dict)
    return metadata


def _json_examples(path: Path) -> list[dict[str, object]]:
    return [
        payload
        for block in re.findall(r"```json\n(.*?)\n```", _text(path), flags=re.DOTALL)
        if isinstance(payload := json.loads(block), dict)
    ]


def _example_with(path: Path, field: str) -> dict[str, object]:
    example = next((item for item in _json_examples(path) if field in item), None)
    assert example is not None, (path, field)
    return example


def _rendered_contract(target_id: str, rendered: str) -> dict[str, object]:
    if target_id == "codex":
        prompt = tomllib.loads(rendered)["developer_instructions"]
    else:
        prompt = rendered.split("---", 2)[2].lstrip()
    raw_contract = prompt.split("## Ownership And Handoff Contract\n\n```yaml\n", 1)[1]
    contract = yaml.safe_load(raw_contract.split("\n```", 1)[0])
    assert isinstance(contract, dict)
    return contract


def _resolved_output(pattern: str, root_name: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        placeholder = match.group(1)
        if placeholder == root_name:
            suffix = placeholder.replace("phase_paths.", "phase-").replace("_", "-")
            return f"runs/{suffix}"
        return "sample"

    return re.sub(r"<([^>]+)>", replacement, pattern)


def test_all_roles_declare_and_render_exact_subtree_overlap_policy() -> None:
    roles = discover_agent_roles(Path.cwd())
    assert roles

    for role in roles:
        assert role.metadata["ownership_resolution"] == OWNERSHIP_RESOLUTION, role.name
        for target_id in ROLE_TARGETS:
            _, rendered = render_agent_role_for_target(role, target_id)
            contract = _rendered_contract(target_id, rendered)
            assert tuple(contract) == ROLE_CONTRACT_FIELDS, (target_id, role.name)
            assert contract["ownership_resolution"] == OWNERSHIP_RESOLUTION


def test_every_configurable_output_resolves_nested_forbidden_root_unambiguously() -> None:
    roles = discover_agent_roles(Path.cwd())
    covered_roots: set[str] = set()

    for role in roles:
        outputs = role.metadata["outputs"]
        assert isinstance(outputs, list)
        for output in outputs:
            assert isinstance(output, str)
            match = re.match(r"^<([^>]+)>", output)
            if match is None or match.group(1) not in CONFIGURABLE_OUTPUT_ROOTS:
                continue
            root_name = match.group(1)
            covered_roots.add(root_name)
            resolved = _resolved_output(output, root_name)
            assert resolved == "runs" or resolved.startswith("runs/"), (role.name, output)
            assert role.metadata["ownership_resolution"] == OWNERSHIP_RESOLUTION

    assert covered_roots == CONFIGURABLE_OUTPUT_ROOTS


def test_final_schema_v3_has_nonempty_hash_bound_comparison_manifest_refs() -> None:
    schemas = []

    for path in FINAL_SELECTOR_CONTRACTS:
        schema = _example_with(path, "selected_run")
        schemas.append(schema)
        comparison_refs = schema["comparison_refs"]
        assert isinstance(comparison_refs, list) and comparison_refs, path
        for reference in comparison_refs:
            assert set(reference) == {"path", "sha256"}, (path, reference)
            assert reference["path"].startswith("<comparisons_dir>/"), (path, reference)
            assert reference["path"].endswith("/comparison_manifest.json"), (path, reference)
            assert ".." not in Path(reference["path"]).parts, (path, reference)
            assert FULL_SHA256.fullmatch(reference["sha256"]), (path, reference)

        normalized = " ".join(_text(path).lower().split())
        assert "validate the referenced `comparison_manifest.json` producer schema" in normalized
        assert "recompute sha-256 over the exact current file bytes" in normalized

    assert schemas[0] == schemas[1]


def test_normative_governance_doc_matches_final_schema_v3() -> None:
    selector = FINAL_SELECTOR_CONTRACTS[0]

    assert _example_with(GOVERNANCE_DOC, "selected_run") == _example_with(selector, "selected_run")
    assert _example_with(GOVERNANCE_DOC, "final_decision") == _example_with(selector, "final_decision")

    normalized = " ".join(_text(GOVERNANCE_DOC).lower().split())
    assert "schema version 3" in normalized
    assert "recompute sha-256 over the exact current file bytes" in normalized
    assert "write `current_final.json` last" in normalized


@pytest.mark.parametrize(
    ("status", "verdict", "blocking_findings", "required_report_edits", "errors", "eligible"),
    [
        ("pass", "consistent", [], [], [], True),
        ("pass", "needs_revision", [], [], [], False),
        ("pass", "inconsistent", [], [], [], False),
        ("pass", "consistent", ["blocker"], [], [], False),
        ("pass", "consistent", [], ["edit"], [], False),
        ("pass", "consistent", [], [], ["error"], False),
        ("blocked", "needs_revision", [], [], [], False),
        ("fail", "inconsistent", [], [], [], False),
    ],
)
def test_lineage_report_review_eligibility_matches_producer_matrix(
    status: str,
    verdict: str,
    blocking_findings: list[str],
    required_report_edits: list[str],
    errors: list[str],
    eligible: bool,
) -> None:
    expected = status == "pass" and verdict == "consistent" and not blocking_findings and not required_report_edits and not errors
    assert expected is eligible

    for path in LINEAGE_CONTRACTS:
        normalized = " ".join(_text(path).lower().split())
        assert "report review is eligible if and only if" in normalized, path
        assert "status exactly `pass`" in normalized, path
        assert "verdict exactly `consistent`" in normalized, path
        for field in ("blocking_findings", "required_report_edits", "errors"):
            assert f"`{field}` exactly empty" in normalized, (path, field)


@pytest.mark.parametrize(
    ("workspace", "classification"),
    [
        ({}, "legacy"),
        ({"workflow": {}}, "legacy"),
        ({"paths": {}}, "legacy"),
        ({"workflow": {"layout": "legacy"}}, "legacy"),
        ({"workflow": {"layout": "version_governed"}}, "governed"),
        ({"paths": {"versions_dir": "versions"}}, "governed"),
        ({"workflow": {"layout": "legacy"}, "paths": {"versions_dir": "versions"}}, "governed"),
    ],
)
def test_router_workspace_classifier_matches_cli_matrix(workspace: dict[str, object], classification: str) -> None:
    workflow = workspace.get("workflow")
    paths = workspace.get("paths")
    governed = (isinstance(workflow, dict) and workflow.get("layout") == "version_governed") or (
        isinstance(paths, dict) and "versions_dir" in paths
    )
    assert ("governed" if governed else "legacy") == classification

    router = _text(Path("agent/skills/open-xquant/SKILL.md"))
    blocks = re.findall(r"```yaml\n(.*?)\n```", router, flags=re.DOTALL)
    classifier = next(
        (
            payload["workspace_classifier"]
            for block in blocks
            if isinstance(payload := yaml.safe_load(block), dict) and "workspace_classifier" in payload
        ),
        None,
    )
    assert classifier == {
        "governed_if_any": [
            "workflow.layout == version_governed",
            "paths.versions_dir is present",
        ],
        "invalid_governed_if": ("paths.versions_dir is present but not a non-empty safe workspace-relative path"),
        "otherwise": "legacy",
        "workspace_yaml_presence_only": "legacy",
    }
