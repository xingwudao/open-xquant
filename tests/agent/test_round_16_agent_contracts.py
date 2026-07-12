from __future__ import annotations

import fnmatch
import json
import re
import tomllib
from pathlib import Path, PurePosixPath

import yaml

from oxq.cli.agent_targets import ROLE_TARGETS, discover_agent_roles, render_agent_role_for_target

GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
FINAL_SELECTOR_CONTRACTS = (
    Path("agent/skills/select-final-version/SKILL.md"),
    Path("agent/roles/oxq-final-selector-worker.md"),
)
LINEAGE_CONTRACTS = (
    Path("agent/skills/audit-artifact-lineage/SKILL.md"),
    Path("agent/roles/oxq-lineage-auditor-worker.md"),
)
COMPARISON_CONTRACTS = (
    Path("agent/skills/compare-strategy-versions/SKILL.md"),
    Path("agent/roles/oxq-experiment-comparator-worker.md"),
)
FULL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).lower().split())


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


def _rendered_prompt(target_id: str, rendered: str) -> str:
    if target_id == "codex":
        return tomllib.loads(rendered)["developer_instructions"]
    return rendered.split("---", 2)[2].lstrip()


def _rendered_contract(target_id: str, rendered: str) -> dict[str, object]:
    prompt = _rendered_prompt(target_id, rendered)
    raw = prompt.split("## Ownership And Handoff Contract\n\n```yaml\n", 1)[1]
    contract = yaml.safe_load(raw.split("\n```", 1)[0])
    assert isinstance(contract, dict)
    return contract


def _body_output_paths(body: str) -> list[str]:
    match = re.search(r"^## Outputs\n(.*?)(?=^## |\Z)", body, flags=re.DOTALL | re.MULTILINE)
    if match is None:
        return []
    return re.findall(r"^- `([^`]+)`", match.group(1), flags=re.MULTILINE)


def _declared_output_owns(path: str, declared_outputs: list[str]) -> bool:
    for declared in declared_outputs:
        pattern = declared.split(" only when", 1)[0].rstrip("/")
        candidate = path.rstrip("/")
        if pattern.endswith("/**") and (candidate == pattern[:-3] or candidate.startswith(f"{pattern[:-3]}/")):
            return True
        if fnmatch.fnmatchcase(candidate, pattern):
            return True
    return False


def _classify_workspace(workspace: dict[str, object]) -> str:
    workflow = workspace.get("workflow")
    paths = workspace.get("paths")
    if isinstance(paths, dict) and "versions_dir" in paths:
        versions_dir = paths["versions_dir"]
        if not isinstance(versions_dir, str) or not versions_dir.strip():
            return "invalid_governed"
        candidate = PurePosixPath(versions_dir)
        if candidate.is_absolute() or ".." in candidate.parts:
            return "invalid_governed"
        return "governed"
    if isinstance(workflow, dict) and workflow.get("layout") == "version_governed":
        return "governed"
    return "legacy"


def test_normative_governance_review_eligibility_has_no_advisory_escape_hatch() -> None:
    contracts = (GOVERNANCE_DOC, *LINEAGE_CONTRACTS, *FINAL_SELECTOR_CONTRACTS)

    for path in contracts:
        normalized = _normalized(path)
        assert "status exactly `pass`" in normalized, path
        assert "verdict exactly `consistent`" in normalized, path
        for field in ("blocking_findings", "required_report_edits", "errors"):
            assert f"`{field}` exactly empty" in normalized, (path, field)
        assert "pass or explicitly non-blocking" not in normalized, path
        assert "pass or explicitly recorded as non-blocking" not in normalized, path
        assert "pass \u6216\u660e\u786e\u8bb0\u5f55\u4e3a non-blocking" not in normalized, path


def test_lineage_audit_schema_is_structured_and_shared() -> None:
    expected_fields = {
        "schema_version",
        "status",
        "scope",
        "hash_algorithm",
        "input_hashes",
        "checked_artifacts",
        "blocking_findings",
        "warnings",
        "next_required_phase",
    }
    schemas = []

    for path in (*LINEAGE_CONTRACTS, GOVERNANCE_DOC):
        schema = _example_with(path, "input_hashes")
        schemas.append(schema)
        assert set(schema) == expected_fields, path
        assert schema["schema_version"] == 2, path
        assert schema["status"] == "pass", path
        assert schema["hash_algorithm"] == "sha256-file-bytes-v1", path
        assert set(schema["scope"]) == {"version_id", "run_id"}, path
        input_hashes = schema["input_hashes"]
        assert isinstance(input_hashes, list) and input_hashes, path
        for reference in input_hashes:
            assert set(reference) == {"path", "sha256"}, (path, reference)
            assert FULL_SHA256.fullmatch(reference["sha256"]), (path, reference)

    assert schemas[0] == schemas[1] == schemas[2]


def test_final_decision_binds_one_current_candidate_scoped_lineage_audit() -> None:
    expected_fields = {
        "schema_version",
        "selection_id",
        "status",
        "selected_version_id",
        "selected_run_id",
        "selected_as",
        "hash_algorithm",
        "selected_run",
        "report_artifacts",
        "report_review",
        "lineage_audit",
        "selection_policy",
        "comparison_refs",
        "blocked_candidates",
        "blocking_findings",
        "created_by_role",
    }
    schemas = []
    producer_audit = _example_with(LINEAGE_CONTRACTS[0], "input_hashes")

    for path in (*FINAL_SELECTOR_CONTRACTS, GOVERNANCE_DOC):
        schema = _example_with(path, "selected_run")
        schemas.append(schema)
        assert set(schema) == expected_fields, path
        assert schema["schema_version"] == 3, path
        lineage = schema["lineage_audit"]
        assert set(lineage) == {"path", "sha256", "scope", "input_hashes"}, path
        assert lineage["path"].startswith("<governance_dir>/lineage_audit_"), path
        assert lineage["path"].endswith(".json"), path
        assert FULL_SHA256.fullmatch(lineage["sha256"]), path
        assert lineage["scope"] == {
            "version_id": schema["selected_version_id"],
            "run_id": schema["selected_run_id"],
        }
        assert isinstance(lineage["input_hashes"], list) and lineage["input_hashes"], path
        assert lineage["scope"] == producer_audit["scope"], path
        assert lineage["input_hashes"] == producer_audit["input_hashes"], path
        input_paths = [reference["path"] for reference in lineage["input_hashes"]]
        assert len(input_paths) == len(set(input_paths)), path

    assert schemas[0] == schemas[1] == schemas[2]


def test_final_selector_rejects_stale_ambiguous_or_wrong_candidate_lineage_audits() -> None:
    required = (
        "require exactly one lineage audit reference",
        "direct regular file under the canonical `<governance_dir>`",
        "recompute sha-256 over every current `input_hashes` file",
        "reject zero or multiple matching lineage audits",
        "reject a lineage audit whose scope does not exactly equal the selected version_id/run_id",
        "copy `scope` and `input_hashes` exactly into `final_decision.lineage_audit`",
    )

    for path in FINAL_SELECTOR_CONTRACTS:
        normalized = _normalized(path)
        for phrase in required:
            assert phrase in normalized, (path, phrase)


def test_every_comparison_ref_targets_a_validated_manifest_with_selected_identity() -> None:
    for path in (*FINAL_SELECTOR_CONTRACTS, GOVERNANCE_DOC):
        schema = _example_with(path, "selected_run")
        for reference in schema["comparison_refs"]:
            assert set(reference) == {"path", "sha256"}, (path, reference)
            assert reference["path"].endswith("/comparison_manifest.json"), (path, reference)
            assert FULL_SHA256.fullmatch(reference["sha256"]), (path, reference)

        normalized = _normalized(path)
        assert "validate the referenced `comparison_manifest.json` producer schema" in normalized, path
        assert "selected version_id/run_id identity appears exactly once" in normalized, path


def test_comparison_manifest_candidate_identity_schema_is_shared() -> None:
    schemas = []

    for path in (*COMPARISON_CONTRACTS, GOVERNANCE_DOC):
        schema = _example_with(path, "candidate_identities")
        schemas.append(schema)
        assert schema["schema_version"] == 1, path
        assert isinstance(schema["comparison_id"], str) and schema["comparison_id"], path
        identities = schema["candidate_identities"]
        assert isinstance(identities, list) and len(identities) >= 2, path
        assert all(set(identity) == {"version_id", "run_id"} for identity in identities), path
        assert len({(item["version_id"], item["run_id"]) for item in identities}) == len(identities)

    assert schemas[0] == schemas[1] == schemas[2]


def test_comparison_manifest_example_contains_selected_decision_identity_once() -> None:
    for path in (*FINAL_SELECTOR_CONTRACTS, GOVERNANCE_DOC):
        decision = _example_with(path, "selected_run")
        comparison = _example_with(path if path == GOVERNANCE_DOC else COMPARISON_CONTRACTS[0], "candidate_identities")
        selected = (decision["selected_version_id"], decision["selected_run_id"])
        identities = [(item["version_id"], item["run_id"]) for item in comparison["candidate_identities"]]
        assert identities.count(selected) == 1, path


def test_every_role_body_artifact_output_is_owned_in_source_and_rendered_contracts() -> None:
    roles = discover_agent_roles(Path.cwd())

    for role in roles:
        body_outputs = _body_output_paths(role.body)
        source_outputs = role.metadata["outputs"]
        assert isinstance(source_outputs, list), role.name
        for body_output in body_outputs:
            assert _declared_output_owns(body_output, source_outputs), (role.name, body_output)

        for target_id in ROLE_TARGETS:
            _, rendered = render_agent_role_for_target(role, target_id)
            rendered_contract = _rendered_contract(target_id, rendered)
            assert rendered_contract["outputs"] == source_outputs, (target_id, role.name)
            rendered_body_outputs = _body_output_paths(_rendered_prompt(target_id, rendered))
            for body_output in rendered_body_outputs:
                assert _declared_output_owns(body_output, rendered_contract["outputs"]), (
                    target_id,
                    role.name,
                    body_output,
                )


def test_report_reviewer_notes_are_response_only_not_an_owned_artifact() -> None:
    role = Path("agent/roles/oxq-report-reviewer-worker.md")
    text = _text(role)
    outputs_section = re.search(r"^## Outputs\n(.*?)(?=^## |\Z)", text, flags=re.DOTALL | re.MULTILINE)
    assert outputs_section is not None
    assert "reviewer notes" not in outputs_section.group(1).lower()
    assert "reviewer notes are response-only" in text.lower()
    assert "must not be written as an artifact" in text.lower()

    source = next(role for role in discover_agent_roles(Path.cwd()) if role.name == "oxq-report-reviewer-worker")
    for target_id in ROLE_TARGETS:
        _, rendered = render_agent_role_for_target(source, target_id)
        prompt = _rendered_prompt(target_id, rendered)
        rendered_outputs = re.search(r"^## Outputs\n(.*?)(?=^## |\Z)", prompt, flags=re.DOTALL | re.MULTILINE)
        assert rendered_outputs is not None
        assert "reviewer notes" not in rendered_outputs.group(1).lower()
        assert "reviewer notes are response-only" in prompt.lower()


def test_router_classifier_mirrors_invalid_governed_cli_contract() -> None:
    matrix = (
        ({}, "legacy"),
        ({"workflow": {"layout": "version_governed"}}, "governed"),
        ({"paths": {"versions_dir": "versions"}}, "governed"),
        ({"paths": {"versions_dir": ""}}, "invalid_governed"),
        ({"paths": {"versions_dir": None}}, "invalid_governed"),
        ({"paths": {"versions_dir": []}}, "invalid_governed"),
        ({"paths": {"versions_dir": "/tmp/versions"}}, "invalid_governed"),
        ({"paths": {"versions_dir": "../versions"}}, "invalid_governed"),
        (
            {"workflow": {"layout": "legacy"}, "paths": {"versions_dir": ""}},
            "invalid_governed",
        ),
    )
    for workspace, expected in matrix:
        assert _classify_workspace(workspace) == expected

    router = Path("agent/skills/open-xquant/SKILL.md")
    blocks = re.findall(r"```yaml\n(.*?)\n```", _text(router), flags=re.DOTALL)
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
    normalized = _normalized(router)
    assert "malformed `paths.versions_dir` is invalid governed, never legacy" in normalized


def test_final_selection_v3_migration_regenerates_current_evidence() -> None:
    for path in FINAL_SELECTOR_CONTRACTS:
        normalized = _normalized(path)
        assert "schema-version-1 or schema-version-2 `final_decision.json`" in normalized, path
        assert "historical" in normalized, path
        assert "do not backfill `lineage_audit` in place" in normalized, path
        assert "rerun final selection" in normalized, path
        assert "existing current pointer unchanged" in normalized, path


def test_lineage_audit_v2_migration_regenerates_current_inputs() -> None:
    for path in (*LINEAGE_CONTRACTS, GOVERNANCE_DOC):
        normalized = _normalized(path)
        assert "schema-version-1 lineage audits are historical only" in normalized, path
        assert "do not backfill `scope` or `input_hashes`" in normalized, path
        assert "rerun artifact lineage audit" in normalized, path
