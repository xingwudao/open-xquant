from __future__ import annotations

from pathlib import Path

import yaml


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _frontmatter(path: str) -> dict[str, object]:
    return yaml.safe_load(_text(path).split("---", 2)[1])


def test_workspace_governor_resolves_configured_artifact_roots() -> None:
    skill = _text("agent/skills/govern-research-workspace/SKILL.md")
    role_path = "agent/roles/oxq-artifact-governor-worker.md"
    role = _text(role_path)
    metadata = _frontmatter(role_path)

    for contract in (skill, role):
        normalized = " ".join(contract.split())
        for key, placeholder, default in (
            ("paths.comparisons_dir", "<comparisons_dir>", "comparisons"),
            ("paths.final_dir", "<final_dir>", "final"),
            ("paths.governance_dir", "<governance_dir>", "governance"),
        ):
            assert key in contract
            assert placeholder in contract
            assert default in contract
        assert "Resolve each key independently" in contract
        assert "only when that key is absent" in normalized
        assert "safe relative path" in contract
        assert "stays inside the workspace" in normalized

    assert metadata["inputs"][-2:] == ["<comparisons_dir>/**", "<final_dir>/**"]
    assert metadata["outputs"] == [
        "<governance_dir>/workspace_audit.json",
        "<governance_dir>/workspace_audit.md",
    ]


def test_coordinator_resolves_configured_conversation_root() -> None:
    path = "agent/roles/oxq-coordinator.md"
    role = _text(path)
    metadata = _frontmatter(path)
    normalized = " ".join(role.split())
    expected_outputs = [
        "<conversations_dir>/<conversation_id>/transcript.md",
        "<conversations_dir>/<conversation_id>/confirmations.jsonl",
        "<conversations_dir>/<conversation_id>/conversation_hash.txt",
    ]

    assert "paths.conversations_dir" in role
    assert "<conversations_dir>" in role
    assert "conversations" in role
    assert "only when that key is absent" in normalized
    assert "safe relative path" in role
    assert "stays inside the workspace" in normalized
    for output in expected_outputs:
        assert output in metadata["outputs"]
        assert output in role


def test_component_author_resolves_configured_component_root() -> None:
    skill = _text("agent/skills/author-component/SKILL.md")
    role_path = "agent/roles/oxq-component-author-worker.md"
    role = _text(role_path)
    metadata = _frontmatter(role_path)

    for contract in (skill, role):
        normalized = " ".join(contract.split())
        assert "paths.components_dir" in contract
        assert "<components_dir>/bundles/<bundle_id>" in contract
        assert "components" in contract
        assert "only when that key is absent" in normalized
        assert "safe relative path" in contract
        assert "stays inside the workspace" in normalized

    assert "<components_dir>/bundles/<bundle_id>/**" in metadata["outputs"]


def test_component_author_forbidden_outputs_do_not_conflict_with_owned_outputs() -> None:
    path = "agent/roles/oxq-component-author-worker.md"
    role = _text(path)
    metadata = _frontmatter(path)
    local_artifacts = {
        "component_request.json",
        "component_manifest.json",
        "component_catalog.json",
        "result.json",
    }

    assert local_artifacts.isdisjoint(metadata["forbidden_outputs"])
    for artifact in local_artifacts:
        assert f"<phase_paths.03_component_authoring>/{artifact}" in role
    assert "<components_dir>/bundles/<bundle_id>/component_manifest.json" in role
    assert "<components_dir>/bundles/<bundle_id>/component_catalog.json" in role
    assert "<installed_sdk_bundle>/**" in metadata["forbidden_outputs"]
    assert "installed open-xquant SDK bundle" in role


def test_runtime_and_backtest_contracts_resolve_configured_component_root() -> None:
    contracts = [
        _text("agent/skills/audit-runtime-semantics/SKILL.md"),
        _text("agent/skills/run-authorized-backtest/SKILL.md"),
        _text("docs/agent-guide.md"),
    ]

    for contract in contracts:
        normalized = " ".join(contract.split())
        assert "paths.components_dir" in contract
        assert "<components_dir>/bundles/<bundle_id>/component_manifest.json" in contract
        assert "components" in contract
        assert "only when that key is absent" in normalized
        assert "safe relative path" in contract
        assert "stays inside the workspace" in normalized
        assert "components/bundles/<bundle_id>/component_manifest.json" not in contract


def test_runtime_auditor_resolves_configured_conversation_root() -> None:
    skill = _text("agent/skills/audit-runtime-semantics/SKILL.md")
    normalized = " ".join(skill.split())

    assert "paths.conversations_dir" in skill
    assert "<conversations_dir>/<conversation_id>/confirmations.jsonl" in skill
    assert "conversations" in skill
    assert "only when that key is absent" in normalized
    assert "safe relative path" in skill
    assert "stays inside the workspace" in normalized
    assert "conversations/<conversation_id>/confirmations.jsonl" not in skill


def test_performance_reviewer_resolves_configured_experiment_registry() -> None:
    skill = _text("agent/skills/review-performance/SKILL.md")
    normalized = " ".join(skill.split())

    assert "paths.experiment_registry" in skill
    assert "<experiment_registry>" in skill
    assert "experiments.jsonl" in skill
    assert "only when that key is absent" in normalized
    assert "safe relative path" in skill
    assert "stays inside the workspace" in normalized
    assert "cat <experiment_registry>" in skill
    assert "cat experiments.jsonl" not in skill


def test_router_resolves_configured_registry_comparison_and_final_roots() -> None:
    router = _text("agent/skills/open-xquant/SKILL.md")
    normalized = " ".join(router.split())

    for key, placeholder, default in (
        ("paths.experiment_registry", "<experiment_registry>", "experiments.jsonl"),
        ("paths.comparisons_dir", "<comparisons_dir>", "comparisons"),
        ("paths.final_dir", "<final_dir>", "final"),
    ):
        assert key in router
        assert placeholder in router
        assert default in router
    assert "Resolve each key independently" in router
    assert "only when that key is absent" in normalized
    assert "safe relative path" in router
    assert "stays inside the workspace" in normalized
    assert "comparisons/<comparison_id>/" not in router
    assert "final/current_final.json" not in router


def test_agent_guide_documents_opencode_dynamic_path_approval_boundary() -> None:
    guide = _text("docs/agent-guide.md")
    normalized = " ".join(guide.split())

    assert "OpenCode" in guide
    assert "edit: ask" in guide
    assert "bash: ask" in guide
    assert "basename glob" in guide
    assert "does not expand" in guide
    assert "custom `paths.conversations_dir`" in guide
    assert "permission.task" in guide
    assert '"*": deny' in guide
    assert "exact managed worker names" in guide
    assert "`--auto` automatically approves permission requests that are not explicitly denied" in normalized
    assert "explicit `deny` remains enforced" in normalized


def test_monitor_role_resolves_configured_experiment_registry() -> None:
    path = "agent/roles/oxq-monitor-worker.md"
    role = _text(path)
    metadata = _frontmatter(path)
    normalized = " ".join(role.split())

    assert "paths.experiment_registry" in role
    assert "<experiment_registry>" in role
    assert "experiments.jsonl" in role
    assert "only when that key is absent" in normalized
    assert "safe relative path" in role
    assert "stays inside the workspace" in normalized
    assert metadata["outputs"][-1] == "<experiment_registry>"
