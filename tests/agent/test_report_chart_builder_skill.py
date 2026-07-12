from __future__ import annotations

import json
from pathlib import Path

import yaml


def _skill_bundle_text(skill_name: str) -> str:
    skill_dir = Path(f"agent/skills/{skill_name}")
    parts = [skill_dir.joinpath("SKILL.md").read_text(encoding="utf-8")]
    references = skill_dir / "references"
    if references.exists():
        parts.extend(path.read_text(encoding="utf-8") for path in sorted(references.glob("*.md")))
    return "\n".join(parts)


def test_strategy_skill_entrypoints_use_progressive_disclosure() -> None:
    for skill_name in ["build-strategy-spec", "audit-strategy-spec"]:
        skill_dir = Path(f"agent/skills/{skill_name}")
        skill_text = skill_dir.joinpath("SKILL.md").read_text(encoding="utf-8")
        line_count = len(skill_text.splitlines())
        assert line_count <= 500

        references = sorted(skill_dir.glob("references/*.md"))
        assert references
        for path in references:
            assert path.relative_to(skill_dir).as_posix() in skill_text


def test_report_chart_builder_skill_documents_chart_asset_workflow() -> None:
    skill = Path("agent/skills/build-report-charts/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "build-report-charts" in text
    assert "discuss chart requirements" in text
    assert "plotting Python" in text
    assert "report_assets/figures" in text
    assert "report_assets/scripts" in text
    assert "publish_report_artifacts" in text
    assert "atomic all-or-rollback batch" in text
    assert "oxq report asset add" not in text
    assert "input_artifacts" in text
    assert "trade_curve" in text
    assert "research_report.md" in text
    assert "research_report.html" in text
    assert "write-research-report" in text
    assert "oxq report write" not in text
    assert "report_evidence.md" not in text
    assert "Do not modify metrics" in text
    assert "Do not modify audit" in text
    assert "non-empty" in text
    assert "dimensions" in text
    assert "manifest" in text
    assert "Chart Applicability Matrix" in text
    assert "Violin Plot" in text
    assert "Pair Plot" in text
    assert "scan the run directory" in text
    assert "recommended chart set" in text
    assert "Canonical Report Chart Order" in text
    assert "Require `seaborn`" in text
    assert "do not silently downgrade" in text
    assert "uv run --extra chart python" in text
    assert "import seaborn as sns" in text
    assert 'matplotlib.use("Agg")' in text
    assert "Numeric claim review is semantic/advisory" in text
    assert "treating the CLI command as proof" in text


def test_report_chart_builder_skill_defines_unified_visual_style_defaults() -> None:
    skill = Path("agent/skills/build-report-charts/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "OpenXQuant Report Chart Style" in text
    assert "OXQ_REPORT_STYLE" in text
    assert "figure.figsize" in text
    assert "(12, 6.75)" in text
    assert "figure.dpi" in text
    assert "savefig.dpi" in text
    assert "axes.facecolor" in text
    assert "grid.color" in text
    assert "font.sans-serif" in text
    assert "PingFang SC" in text
    assert "Noto Sans CJK SC" in text
    assert "axes.unicode_minus" in text
    assert "sns.set_theme" in text
    assert 'market_region == "cn"' in text
    assert "market.region == cn" not in text
    assert "red-up / green-down" in text
    assert "custom chart" in text


def test_report_chart_builder_skill_requires_professional_chart_pack() -> None:
    skill = Path("agent/skills/build-report-charts/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "Default Professional Chart Pack" in text
    assert "trade curve" in text
    assert "equity curve vs benchmark" in text
    assert "drawdown" in text
    assert "monthly return heatmap" in text
    assert "IS/OOS" in text
    assert "cost sensitivity" in text
    assert "parameter perturbation" in text
    assert "regime analysis" in text
    assert "position exposure" in text
    assert "trade PnL distribution" in text
    assert "message title" in text
    assert "source artifact" in text
    assert "caption" in text
    assert "buy/sell markers" in text
    assert "orders.csv" in text
    assert "target_weights.csv" in text

    default_pack = text[
        text.index("## Default Professional Chart Pack"): text.index("## Chart Applicability Matrix")
    ]
    assert default_pack.index("- equity curve vs benchmark") < default_pack.index("- drawdown")
    assert default_pack.index("- drawdown") < default_pack.index("- trade curve")
    assert "Use this order unless the user explicitly requests a different order" in default_pack
    assert "Default Report Mode" in text
    assert "build the Default Professional Chart Pack automatically" in text
    assert "Do not ask the user to confirm the default report chart batch" in text
    assert "Do not omit charts merely because the user did not request them" in text


def test_report_chart_builder_skill_uses_canonical_report_chart_order() -> None:
    skill = Path("agent/skills/build-report-charts/SKILL.md")

    text = skill.read_text(encoding="utf-8")
    order = text[text.index("## Canonical Report Chart Order"): text.index("## Default Professional Chart Pack")]

    expected_order = [
        "equity_curve",
        "drawdown",
        "trade_curve",
        "position_exposure",
        "monthly_returns",
        "cost_sensitivity",
        "is_oos_comparison",
        "parameter_perturbation",
        "regime_analysis",
        "trade_pnl_distribution",
    ]
    positions = [order.index(chart_id) for chart_id in expected_order]
    assert positions == sorted(positions)


def test_report_chart_builder_skill_batch_example_sorts_in_canonical_order() -> None:
    skill = Path("agent/skills/build-report-charts/SKILL.md")

    text = skill.read_text(encoding="utf-8")
    batch_json = text.split("```json", 1)[1].split("```", 1)[0]
    assets = json.loads(batch_json)["assets"]
    canonical_core = ["equity_curve", "drawdown", "trade_curve"]
    core_assets = [asset for asset in assets if asset["id"] in canonical_core]

    assert [asset["id"] for asset in core_assets] == canonical_core
    sorted_ids = [
        asset["id"]
        for asset in sorted(
            core_assets,
            key=lambda asset: (asset["section"], asset["order"], asset["id"]),
        )
    ]
    assert sorted_ids == canonical_core


def test_report_chart_builder_skill_localizes_manifest_titles_and_captions() -> None:
    skill = Path("agent/skills/build-report-charts/SKILL.md")

    text = skill.read_text(encoding="utf-8")
    batch_json = text.split("```json", 1)[1].split("```", 1)[0]
    assets = json.loads(batch_json)["assets"]

    assert "manifest `title` and `caption`" in text
    assert "report_language" in text
    assert "Equity curve vs benchmark" not in batch_json
    assert "Drawdown curve" not in batch_json
    assert "Trade curve" not in batch_json
    assert "Generated from" not in batch_json
    for asset in assets:
        assert asset["title"]
        assert asset["caption"]
        assert any("\u4e00" <= char <= "\u9fff" for char in asset["title"])
        assert any("\u4e00" <= char <= "\u9fff" for char in asset["caption"])


def test_report_chart_builder_skill_defines_trade_curve_requirements() -> None:
    skill = Path("agent/skills/build-report-charts/SKILL.md")

    text = skill.read_text(encoding="utf-8")
    matrix = text[text.index("## Chart Applicability Matrix"): text.index("## Red Lines")]

    assert "Trade Curve" in matrix
    assert "`equity_curve.csv`, non-empty `trades.csv`" in matrix
    assert "`orders.csv`" in matrix
    assert "`target_weights.csv`" in matrix
    assert "`benchmark_curve.csv`" in matrix
    assert "Rotation-strategy value: core/default" in matrix


def test_opencode_target_specific_source_bundle_is_removed() -> None:
    assert not Path("agent/opencode").exists()


def test_skills_are_directory_canonical_sources() -> None:
    canonical_skills = sorted(Path("agent/skills").glob("*/SKILL.md"))
    assert canonical_skills
    for canonical in canonical_skills:
        skill_name = canonical.parent.name
        assert not canonical.is_symlink()
        canonical_meta = yaml.safe_load(canonical.read_text(encoding="utf-8").split("---", 2)[1])
        assert canonical_meta["name"] == skill_name
        assert not Path(f"agent/skills/{skill_name}.md").exists()


def test_research_report_writer_skill_requires_agent_authored_final_report() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "write-research-report" in text
    assert "research_report.md" in text
    assert "research_report.html" in text
    assert "render_markdown_html_report" in text
    assert "human researcher" in text
    assert "potential investor" in text
    assert "report_evidence.md" not in text
    assert "oxq report write" not in text
    assert "Do not invent evidence" in text
    assert "monthly returns" in text
    assert "positive/negative month counts" in text
    assert "artifact or facts API" in text
    assert "Mandatory Routing" in text
    assert "not a reason to bypass this skill" in text
    assert "write the report directly" in text
    assert "Chart Decision Gate" in text
    assert "Do not ask the user questions directly from this skill" in text
    assert "writer_result.json" in text
    assert "report_writer_result.json" not in text
    assert "missing_required_report_charts" in text
    assert "missing_chart_decision" not in text
    assert "no charts were requested" not in text
    assert "build-report-charts" in text
    assert "Evidence is generated by the framework; the narrative is authored by the Agent" in text


def test_research_report_writer_skill_defines_default_language_parameter() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "Language Parameter Gate" in text
    assert "report_language" in text
    assert "中文" in text
    assert "If the user does not explicitly request another language" in text
    assert '"language": "中文"' in text
    assert "language_to_html_lang" in text
    assert "html_lang = language_to_html_lang(report_language)" in text
    assert "render_markdown_html_report(markdown, lang=html_lang)" in text
    assert "render_markdown_html_report(markdown, lang=\"zh\")" not in text
    assert "Do not switch the whole report to English" in text


def test_research_report_writer_skill_resolves_language_before_chart_gate() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert text.index("## Language Parameter Gate") < text.index("## Chart Decision Gate")
    chart_gate = text[text.index("## Chart Decision Gate"): text.index("## Inputs")]
    assert "`language`: `report_language`" in chart_gate
    assert "blocked" in chart_gate
    assert "charts are required by default" in chart_gate
    assert "`chart_decision: default_professional_chart_pack`" in chart_gate
    assert "`next_skill: build-report-charts`" in chart_gate
    assert "Do not ask the user whether to build charts" in chart_gate
    assert "missing_required_report_charts" in chart_gate


def test_research_report_writer_skill_preserves_canonical_decision_tokens() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    language_gate = text[text.index("## Language Parameter Gate"): text.index("## Inputs")]
    assert "Do not localize the canonical executive decision token" in language_gate
    assert "REJECT" in language_gate
    assert "NO EVIDENCE" in language_gate
    assert "WATCHLIST" in language_gate
    assert "PAPER TRADING CANDIDATE" in language_gate


def test_research_report_writer_skill_preserves_unknown_html_language_codes() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    html_output = text[text.index("## HTML Output"): text.index("## Red Lines")]
    assert "Do not fall back to `zh` for an explicitly requested non-Chinese language" in html_output
    assert "return normalized" in html_output
    assert 'return "und"' in html_output


def test_research_report_writer_skill_requires_institutional_report_structure() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "Institutional Report Standard" in text
    assert "Executive Snapshot" in text
    assert "30-second investor view" in text
    assert "3-minute research view" in text
    assert "professional appendix" in text
    assert "trust and audit status" in text
    assert "risks near the decision" in text
    assert "metric scorecard" in text
    assert "message-first" in text


def test_research_report_writer_skill_requires_deterministic_date_labels() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md")
    role = Path("agent/roles/oxq-report-writer-worker.md")

    skill_text = skill.read_text(encoding="utf-8")
    role_text = role.read_text(encoding="utf-8")

    assert "配置结束日：YYYY-MM-DD" in skill_text
    assert "有效数据最后交易日：YYYY-MM-DD" in skill_text
    assert "Configured end date: YYYY-MM-DD" in skill_text
    assert "Effective last trading day: YYYY-MM-DD" in skill_text
    assert "Do not use label variants" in skill_text
    assert "English fallback labels inside a Chinese report" in skill_text
    assert "配置结束日：YYYY-MM-DD" in role_text
    assert "有效数据最后交易日：YYYY-MM-DD" in role_text
    assert "English\n  fallback labels inside a Chinese report" in role_text


def test_research_report_writer_skill_requires_artifact_backed_drawdown_periods() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "maximum drawdown peak/trough dates" in text
    assert "equity_curve.csv" in text
    assert "omit the peak/trough dates and equity values" in text


def test_open_xquant_router_skill_routes_quant_tasks_to_leaf_skills() -> None:
    skill = Path("agent/skills/open-xquant/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "name: open-xquant" in text
    assert "Use when" in text
    assert "quantitative research" in text
    assert "Router Contract" in text
    assert "Do not run other `oxq` commands" in text
    assert "Do not write report files directly" in text
    assert "minimal runner/workspace commands" in text
    assert "`research init --sdk`" in text
    assert "Do not resolve a runner for `brainstorm-strategy-idea` or `audit-strategy-idea`" in text
    assert "Install And Upgrade Questions" in text
    assert "installed Agents must not depend on that file" in text
    assert "<runner> agent status" in text
    assert "<runner> agent upgrade --all-targets --from-local . --yes" in text
    for leaf_skill in [
        "manage-strategy-version",
        "govern-research-workspace",
        "audit-artifact-lineage",
        "brainstorm-strategy-idea",
        "audit-strategy-idea",
        "build-strategy-spec",
        "audit-strategy-spec",
        "audit-runtime-semantics",
        "run-authorized-backtest",
        "monitor-strategy-run",
        "build-report-charts",
        "compare-experiments",
        "compare-strategy-versions",
        "select-final-version",
        "write-research-report",
        "review-research-report",
        "review-performance",
        "evaluate-factor",
        "screen-factors",
        "tune-parameters",
        "author-component",
        "create-component",
        "manage-live-trading",
    ]:
        assert leaf_skill in text
    assert "Multi-Agent workflows use narrow leaf skills only" in text
    assert "Workspace-local custom Rule requests must block" in text
    assert "audited spec, compile, runtime, and backtest support" in text
    assert "Final version selection" in text
    assert "version governance" in text
    assert "Cross-version strategy comparison" in text
    assert "strategy-builder-standalone" not in text
    assert "quant-research" not in text


def test_coordinator_role_documents_subagent_workflow() -> None:
    role = Path("agent/roles/oxq-coordinator.md")

    text = role.read_text(encoding="utf-8")

    assert "open-xquant SubAgent workflow" in text
    assert "Prefer SubAgents by default" in text
    assert "Brainstormer writes `strategy_idea_brief.json`" in text
    assert "Idea auditor writes `strategy_idea_audit.json`" in text
    assert "Builder reads the audited idea artifacts" in text
    assert "If builder returns `next_required_phase: data_inspection`" in text
    assert "Data inspector checks required symbols" in text
    assert "oxq-data-inspection-worker" in text
    assert "Spec auditor reads those artifacts" in text
    phase_order = text[text.index("## Strategy Phase Order"): text.index("If `oxq-strategy-idea-auditor-worker` blocks")]
    assert phase_order.index("`oxq-strategy-builder-worker`") < phase_order.index("`oxq-data-inspection-worker`")
    assert phase_order.index("`oxq-data-inspection-worker`") < phase_order.index("`oxq-spec-auditor-worker`")
    assert phase_order.index("`oxq-spec-auditor-worker`") < phase_order.index("`user_spec_confirmation`")
    assert phase_order.index("`user_spec_confirmation`") < phase_order.index("`oxq-runtime-auditor-worker`")
    assert "Runtime auditor reads the authorized spec/audit artifacts" in text
    assert "Runner reads `backtest_authorization.json`" in text
    assert "Do not\n  delegate this file to a generic worker" in text
    assert "`run-authorized-backtest` contract exactly" in text
    assert "`status: authorized`" in text
    assert "`spec_hash`, `spec_audit_hash`, and `runtime_audit_hash`" in text
    assert "nested\n  `canonical_hashes` object" in text
    assert "Version manager decides whether a user change creates a new version" in text
    assert "Artifact governor audits workspace layout" in text
    assert "Lineage auditor verifies version/run/final references" in text
    assert "Final selector writes final selection artifacts" in text
    assert "Main agent only coordinates" in text
    assert "phase completion" in text
    assert "active_phase" in text
    assert "10_reports" in text
    assert "When `oxq-runner-worker` returns `status: pass`" in text
    assert "immediately route `oxq-monitor-worker`" in text
    assert "When `oxq-monitor-worker` returns `status: pass`" in text
    assert "immediately route `oxq-report-writer-worker`" in text
    assert "Do not stop after backtest completion" in text
    assert "chart_decision: default_professional_chart_pack" in text
    assert "Do not set `chart_decision: no_charts_requested`" in text
    assert "Do not ask the user whether to generate report charts" in text


def test_workspace_governance_skills_are_canonical_sources() -> None:
    expected = {
        "manage-strategy-version": [
            "strategy family",
            "new version",
            "semantic change",
            "phase completion",
            "current_phase",
            "lineage.json",
            "current.json",
            "version_manifest.json",
        ],
        "govern-research-workspace": [
            "workspace artifact governance",
            "root-level `strategy_spec.yaml`",
            "workflow_manifest.json",
            "workspace_audit.json",
            "phase artifact",
        ],
        "audit-artifact-lineage": [
            "artifact lineage",
            "version/run/final",
            "hash_type",
            "lineage_audit",
            "eligible candidate",
        ],
        "compare-strategy-versions": [
            "cross-version",
            "within-version",
            "comparison_manifest.json",
            "comparability_audit.json",
            "spec_diff.yaml",
        ],
        "select-final-version": [
            "final version",
            "selection_policy.json",
            "final_decision.json",
            "current_final.json",
            "confirmed_by_user",
        ],
    }

    for skill_name, required_fragments in expected.items():
        skill = Path(f"agent/skills/{skill_name}/SKILL.md")
        assert skill.exists(), skill
        text = skill.read_text(encoding="utf-8")
        assert f"name: {skill_name}" in text
        for fragment in required_fragments:
            assert fragment in text


def test_workspace_governance_worker_roles_are_installed_boundaries() -> None:
    expected = {
        "oxq-version-manager-worker": "manage-strategy-version",
        "oxq-artifact-governor-worker": "govern-research-workspace",
        "oxq-lineage-auditor-worker": "audit-artifact-lineage",
        "oxq-experiment-comparator-worker": "compare-strategy-versions",
        "oxq-final-selector-worker": "select-final-version",
        "oxq-monitor-worker": "monitor-strategy-run",
    }

    for role_name, skill_name in expected.items():
        role = Path(f"agent/roles/{role_name}.md")
        assert role.exists(), role
        text = role.read_text(encoding="utf-8")
        assert f"name: {role_name}" in text
        assert skill_name in text
        assert "forbidden_outputs" in text
        assert "strategy_spec.yaml" in text or "runs/**" in text or "research_report.md" in text


def test_version_governed_phase_paths_are_mandatory_for_leaf_skills_and_roles() -> None:
    required_skill_fragments = {
        "brainstorm-strategy-idea": [
            "<phase_paths.01_brainstorm>/strategy_idea_brief.json",
            "Do not write root-level `strategy_idea_brief.json`",
        ],
        "audit-strategy-idea": [
            "<phase_paths.02_idea_audit>/strategy_idea_audit.json",
            "Do not write root-level `strategy_idea_audit.json`",
        ],
        "build-strategy-spec": [
            "<phase_paths.04_spec_build>/strategy_spec.yaml",
            "Do not write root-level `strategy_spec.yaml`",
        ],
        "audit-strategy-spec": [
            "<phase_paths.06_spec_audit>/spec_confirmation_table.md",
            "Do not write root-level `spec_audit.json`",
        ],
        "audit-runtime-semantics": [
            "<phase_paths.07_compile_preview>/compiled_plan.json",
            "<phase_paths.08_runtime_audit>/runtime_audit.json",
        ],
        "run-authorized-backtest": [
            "<phase_paths.09_backtests>/<run_id>/strategy_spec.yaml",
            "Do not write formal run outputs to root `runs/`",
        ],
        "monitor-strategy-run": [
            "<phase_paths.09_backtests>/<run_id>/reproducibility_audit.json",
            "version_id",
        ],
        "write-research-report": [
            "<phase_paths.10_reports>/<run_id>/research_report.md",
            "Do not write root-level `research_report.md`",
        ],
        "review-research-report": [
            "<phase_paths.10_reports>/<run_id>/report_review.json",
            "Do not write root-level `report_review.json`",
        ],
    }

    for skill_name, fragments in required_skill_fragments.items():
        text = Path(f"agent/skills/{skill_name}/SKILL.md").read_text(encoding="utf-8")
        for fragment in fragments:
            assert fragment in text, f"{skill_name} missing {fragment}"

    coordinator = Path("agent/roles/oxq-coordinator.md").read_text(encoding="utf-8")
    assert "Version-Governed Artifact Contract" in coordinator
    assert "active_version" in coordinator
    assert "Root-level phase artifacts are layout pollution" in coordinator
    assert "<phase_paths.04_spec_build>/strategy_spec.yaml" in coordinator


def test_agent_contracts_resolve_custom_version_root_through_phase_manifest() -> None:
    contract_paths = [
        *Path("agent/roles").glob("*.md"),
        *Path("agent/skills").glob("*/SKILL.md"),
        *Path("agent/skills").glob("*/references/*.md"),
        Path("docs/agent-guide.md"),
        Path("docs/architecture.md"),
        Path("docs/strategy-workflow-artifact-governance.md"),
        Path("README.md"),
    ]
    forbidden_patterns = (
        "versions/v001",
        "versions/vNNN",
        "versions/**",
        "versions/<active_version>",
        "versions/<version_id>",
        "versions/{version_id}",
    )
    offenders = [
        path.as_posix()
        for path in contract_paths
        if any(pattern in path.read_text(encoding="utf-8") for pattern in forbidden_patterns)
    ]

    assert offenders == []

    for path in (
        Path("agent/roles/oxq-coordinator.md"),
        Path("agent/skills/manage-strategy-version/SKILL.md"),
        Path("agent/skills/build-strategy-spec/SKILL.md"),
        Path("agent/skills/run-authorized-backtest/SKILL.md"),
        Path("agent/skills/write-research-report/SKILL.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "paths.versions_dir" in text, path
        assert "version_manifest.json" in text, path
        assert "phase_paths" in text, path
        assert "research_versions" in text, path

    coordinator = Path("agent/roles/oxq-coordinator.md").read_text(encoding="utf-8")
    assert "<phase_paths.04_spec_build>/strategy_spec.yaml" in coordinator
    assert "<phase_paths.09_backtests>/<run_id>/" in coordinator
    assert "research_versions/v003/04_spec_build" in coordinator
    assert "`versions/v003/04_spec_build`" not in coordinator


def test_manage_strategy_version_bootstraps_new_version_under_custom_root() -> None:
    text = Path("agent/skills/manage-strategy-version/SKILL.md").read_text(encoding="utf-8")

    assert "new-version bootstrap" in text
    assert "<version_root>/<version_id>/version_manifest.json" in text
    assert "<version_root>/<version_id>/phase_state.json" in text
    assert "<version_root>/<version_id>/01_brainstorm" in text
    assert "canonical `phase_paths`" in text
    assert "versions/vNNN" not in text


def test_manage_strategy_version_bootstrap_contract_matches_initialized_workspace_schema() -> None:
    text = Path("agent/skills/manage-strategy-version/SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "`schema_version`" in text
    for phase in (
        "01_brainstorm",
        "02_idea_audit",
        "03_component_authoring",
        "04_spec_build",
        "05_data_inspection",
        "06_spec_audit",
        "07_compile_preview",
        "08_runtime_audit",
        "09_backtests",
        "10_reports",
    ):
        assert f"research_versions/v003/{phase}" in text
    assert "Create every directory named by `phase_paths` before publishing" in normalized
    assert "`lineage.json.versions`" in text
    assert "`phase_state.json.current_phase`" in text
    assert "`version_manifest.json.active_phase`" in text
    assert "`current.json.active_version`" in text
    assert "`current.json.active_phase`" in text


def test_manage_strategy_version_bootstrap_atomically_supersedes_prior_active_version() -> None:
    text = Path("agent/skills/manage-strategy-version/SKILL.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "v001 -> v002 bootstrap transaction" in normalized
    assert "prior `v001` lineage entry status to `superseded`" in normalized
    assert "prior `v001` version_manifest.json status to `superseded`" in normalized
    assert "new `v002` lineage entry and version_manifest.json status to `active`" in normalized
    assert "exactly one `active` lineage entry" in normalized
    assert "must match `current.json.active_version`" in normalized
    assert "publish `current.json` last" in normalized


def test_version_manager_worker_bootstraps_before_new_version_manifest_exists() -> None:
    text = Path("agent/roles/oxq-version-manager-worker.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "new-version bootstrap" in text
    assert "the new version manifest does not exist yet" in normalized
    assert "must not be treated as an input prerequisite" in normalized
    assert "For an existing version, read" in text


def test_governance_and_lineage_contracts_resolve_custom_version_roots() -> None:
    paths = [
        Path("agent/roles/oxq-artifact-governor-worker.md"),
        Path("agent/roles/oxq-lineage-auditor-worker.md"),
        Path("agent/skills/audit-artifact-lineage/SKILL.md"),
        Path("agent/skills/govern-research-workspace/SKILL.md"),
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert ".open-xquant/workspace.yaml" in text, path
        assert "paths.versions_dir" in text, path
        assert "<version_root>/**" in text, path
        assert "<phase_paths.09_backtests>" in text, path
        assert "version_manifest.json" in text, path
        assert "phase_paths" in text, path
        assert "versions/**" not in text, path
        assert "versions/<active_version>" not in text, path


def test_lineage_contracts_resolve_each_cross_version_candidate_manifest() -> None:
    for path in (
        Path("agent/roles/oxq-lineage-auditor-worker.md"),
        Path("agent/skills/audit-artifact-lineage/SKILL.md"),
    ):
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "active version is `v003`" in normalized, path
        assert "candidates reference `v001` and `v002`" in normalized, path
        assert "<version_root>/v001/version_manifest.json" in text, path
        assert "<version_root>/v002/version_manifest.json" in text, path
        assert "Reserve `current.json` for active-state checks" in text, path


def test_final_governance_contracts_resolve_each_configured_artifact_root() -> None:
    defaults = {
        "paths.experiment_registry": "experiments.jsonl",
        "paths.governance_dir": "governance",
        "paths.comparisons_dir": "comparisons",
        "paths.comparison_registry": "comparisons/comparisons.jsonl",
        "paths.final_dir": "final",
    }
    contracts = {
        Path("agent/skills/select-final-version/SKILL.md"): {
            "paths.experiment_registry": "<experiment_registry>",
            "paths.comparisons_dir": "<comparisons_dir>",
            "paths.governance_dir": "<governance_dir>",
            "paths.final_dir": "<final_dir>",
        },
        Path("agent/roles/oxq-final-selector-worker.md"): {
            "paths.experiment_registry": "<experiment_registry>",
            "paths.comparisons_dir": "<comparisons_dir>",
            "paths.governance_dir": "<governance_dir>",
            "paths.final_dir": "<final_dir>",
        },
        Path("agent/skills/compare-strategy-versions/SKILL.md"): {
            "paths.experiment_registry": "<experiment_registry>",
            "paths.governance_dir": "<governance_dir>",
            "paths.comparisons_dir": "<comparisons_dir>",
            "paths.comparison_registry": "<comparison_registry>",
        },
        Path("agent/roles/oxq-experiment-comparator-worker.md"): {
            "paths.experiment_registry": "<experiment_registry>",
            "paths.governance_dir": "<governance_dir>",
            "paths.comparisons_dir": "<comparisons_dir>",
            "paths.comparison_registry": "<comparison_registry>",
        },
        Path("agent/skills/audit-artifact-lineage/SKILL.md"): {
            "paths.governance_dir": "<governance_dir>",
            "paths.comparisons_dir": "<comparisons_dir>",
            "paths.final_dir": "<final_dir>",
        },
        Path("agent/roles/oxq-lineage-auditor-worker.md"): {
            "paths.governance_dir": "<governance_dir>",
            "paths.comparisons_dir": "<comparisons_dir>",
            "paths.final_dir": "<final_dir>",
        },
    }

    for path, required_paths in contracts.items():
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert ".open-xquant/workspace.yaml" in text, path
        assert "Resolve each key independently" in text, path
        assert "only when that key is absent" in normalized, path
        assert "absolute" in text, path
        assert "traversal" in text, path
        assert "symlink" in text, path
        for config_key, resolved_path in required_paths.items():
            assert config_key in text, f"{path} missing {config_key}"
            assert resolved_path in text, f"{path} missing {resolved_path}"
            assert defaults[config_key] in text, f"{path} missing default for {config_key}"


def test_router_and_builder_define_read_only_existing_spec_validation_mode() -> None:
    router = Path("agent/skills/open-xquant/SKILL.md").read_text(encoding="utf-8")
    builder = Path("agent/skills/build-strategy-spec/SKILL.md").read_text(encoding="utf-8")
    normalized_builder = " ".join(builder.split())

    assert "read-only validation-only mode" in router
    assert "## Read-Only Validation-Only Mode" in builder
    assert "does not require `strategy_idea_brief.json`" in normalized_builder
    assert "does not require `strategy_idea_audit.json`" in normalized_builder
    assert "Do not run `oxq registry export`" in builder
    assert "Do not create or modify `builder_phase_result.json`" in builder
    assert "Do not create or modify `spec_mapping_contract.json`" in builder


def test_formal_contracts_do_not_override_resolved_version_root() -> None:
    paths = [
        Path("agent/skills/brainstorm-strategy-idea/SKILL.md"),
        Path("agent/skills/audit-strategy-idea/SKILL.md"),
        Path("README.md"),
    ]
    forbidden = (
        "versions/v001",
        "versions/vNNN",
        "versions/**",
        "versions/<active_version>",
        "versions/<version_id>",
        "versions/{version_id}",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text, f"{path} contains concrete default-root override {pattern}"

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "<phase_paths.08_runtime_audit>/runtime_audit.json" in readme
    assert "<phase_paths.09_backtests>/<run_id>/" in readme
    assert "<phase_paths.10_reports>/<run_id>/" in readme
    assert "paths.versions_dir" in readme
    assert "version_root" in readme


def test_architecture_doc_links_strategy_workflow_governance_design() -> None:
    text = Path("docs/architecture.md").read_text(encoding="utf-8")

    assert "Strategy Workflow Artifact Governance" in text
    assert "docs/strategy-workflow-artifact-governance.md" in text
    assert "docs/images/strategy-workflow-artifact-governance.png" in text
    assert "strategy family -> strategy version -> run attempt" in text
    assert "select-final-version" in text


def test_architecture_and_agent_guide_list_workspace_governance_roles() -> None:
    expected_roles = [
        "oxq-version-manager-worker",
        "oxq-artifact-governor-worker",
        "oxq-lineage-auditor-worker",
        "oxq-experiment-comparator-worker",
        "oxq-final-selector-worker",
    ]

    for doc_path in ["docs/architecture.md", "docs/agent-guide.md"]:
        text = Path(doc_path).read_text(encoding="utf-8")
        for role in expected_roles:
            assert role in text, f"{doc_path} missing {role}"


def test_version_governed_docs_and_skill_examples_do_not_use_root_run_paths() -> None:
    checked_paths = [
        "docs/architecture.md",
        "docs/agent-guide.md",
        "agent/skills/build-strategy-spec/SKILL.md",
        "agent/skills/audit-strategy-idea/SKILL.md",
        "agent/skills/audit-runtime-semantics/SKILL.md",
        "agent/skills/run-authorized-backtest/SKILL.md",
        "agent/skills/review-performance/SKILL.md",
    ]

    for checked_path in checked_paths:
        text = Path(checked_path).read_text(encoding="utf-8")
        assert "runs/<run_id>" not in text, checked_path
        assert "oxq spec validate strategy_spec.yaml" not in text, checked_path
        assert "oxq strategy compile strategy_spec.yaml" not in text, checked_path
        assert "oxq backtest run strategy_spec.yaml" not in text, checked_path
        assert '"strategy_spec": "strategy_spec.yaml"' not in text, checked_path
        assert '"component_catalog": "component_catalog.json"' not in text, checked_path
        assert '"strategy_idea_brief": "strategy_idea_brief.json"' not in text, checked_path


def test_data_inspection_worker_is_narrow_role() -> None:
    role = Path("agent/roles/oxq-data-inspection-worker.md")
    text = role.read_text(encoding="utf-8")

    assert "role_kind: data_inspection" in text
    assert "explore-data" in text
    assert "data_inspection_result.json" in text
    assert "data_availability_report.md" in text
    assert "Do not edit `strategy_spec.yaml`" in text
    assert "Do not run formal backtests" in text


def test_component_author_skill_documents_workspace_extension_contract() -> None:
    skill = Path("agent/skills/author-component/SKILL.md")
    role = Path("agent/roles/oxq-component-author-worker.md")

    text = skill.read_text(encoding="utf-8")
    role_text = role.read_text(encoding="utf-8")

    assert "name: author-component" in text
    assert "component_request.json" in text
    assert "<phase_paths.03_component_authoring>/component_request.json" in text
    assert "<phase_paths.03_component_authoring>/result.json" in text
    assert "<components_dir>/bundles/<bundle_id>/" in text
    assert "Do not write root-level `component_request.json`" in text
    assert "Do not write root-level `result.json`" in text
    assert "custom_components/" in text
    assert "component_manifest.json" in text
    assert "result.json" in text
    assert "oxq component-manifest hash" in text
    assert "oxq component-manifest validate" in text
    assert "--component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json" in text
    assert "Workspace-local `Rule` authoring is currently blocked" in text
    assert "Do not emit `component_ready` for a workspace-local custom `Rule`" in text
    assert "Do not build or edit `strategy_spec.yaml`" in text
    assert "Do not modify the installed SDK bundle" in text
    assert "role_kind: component_author" in role_text
    assert "author-component" in role_text
    assert "<phase_paths.03_component_authoring>/" in role_text
    assert "<components_dir>/bundles/<bundle_id>/" in role_text
    assert "create-rule" not in role_text
    assert "Block workspace-local custom `Rule` requests" in role_text
    assert "forbidden_outputs" in role_text
    for ignored_artifact in ["__pycache__", ".pytest_cache", "*.egg-info", ".mypy_cache", ".ruff_cache"]:
        assert ignored_artifact in text
        assert ignored_artifact in role_text
    assert "must not remain" in text
    assert "must not remain" in role_text


def test_cross_sectional_component_logic_prefers_optimizer_without_forcing_it() -> None:
    builder = _skill_bundle_text("build-strategy-spec")
    author = Path("agent/skills/author-component/SKILL.md").read_text(encoding="utf-8")
    auditor = _skill_bundle_text("audit-strategy-spec")

    for text in (builder, author, auditor):
        assert "Cross-Sectional Component Feasibility" in text
        assert "PortfolioOptimizer first" in text
        assert "do not force `PortfolioOptimizer`" in text
        assert "framework_unsupported" in text
        assert "all-symbol same-date" in text

    assert "needs_custom_component" in builder
    assert "`suggested_kind`: `PortfolioOptimizer`" in builder
    assert "`feasibility_status`: `candidate`" in builder
    assert "`feasibility_status`: `unsupported`" in builder
    assert "`signal.indicators.<name>.type: RPS`" in builder

    assert "If the request says `Indicator` but the behavior requires all-symbol same-date input" in author
    assert "reclassify the candidate kind to `PortfolioOptimizer` only when" in author
    assert "write `status: blocked` with `blocked_reason`" in author

    assert "A SPEC must not claim a cross-sectional transform is implemented" in auditor
    assert "registered `RPS`" in auditor


def test_builder_documents_tradability_lag_latest_and_timing_boundaries() -> None:
    builder = _skill_bundle_text("build-strategy-spec")
    auditor = _skill_bundle_text("audit-strategy-spec")
    builder_role = Path("agent/roles/oxq-strategy-builder-worker.md").read_text(encoding="utf-8")
    data_skill = Path("agent/skills/explore-data/SKILL.md").read_text(encoding="utf-8")
    data_role = Path("agent/roles/oxq-data-inspection-worker.md").read_text(encoding="utf-8")

    assert "Data Inspection Boundary" in builder
    assert "Current SPEC Template Shape" in builder
    assert "`schema_version` must be the string `\"0.1\"`" in builder
    assert "top-level `strategy_id` and `name`" in builder
    assert "`validation.train_period` and `validation.test_period` are two-item lists" in builder
    assert "`cost` is a top-level section" in builder
    assert "list `data_dir`, read parquet files" in builder
    assert "Do not inspect market data to resolve it inside this builder phase" in builder
    assert "Do not choose an arbitrary calendar date to make `oxq spec validate` pass" in builder
    assert "A validation error caused by an" in builder
    assert "unresolved data-inspection dependency is a blocked handoff" in builder
    assert "`next_required_phase: data_inspection`" in builder
    assert "`data.filters.exclude_suspended: true`" in builder
    assert "`data.filters.suspension_policy: hold_existing`" in builder
    assert "`data.required_columns`" in builder
    assert "do not drop the column because it is" in builder
    assert "unverified or missing" in builder
    assert "before-close" in builder
    assert "substituting `next_open`" in builder
    assert "Do not replace `latest_available`" in builder
    assert "every indicator that directly reads `close`" in builder
    assert "Data Boundary Audit" in auditor
    assert "Do not accept" in auditor
    assert "builder-authored notes saying it inspected parquet files" in auditor
    assert "A silent default `lag_bars: 0`" in auditor
    assert "Data coverage, required columns, and" in builder_role
    assert "`latest_available` resolution belong to `oxq-data-inspection-worker`" in builder_role
    assert "`schema_version: \"0.1\"`" in builder_role
    assert "Do not choose an arbitrary fixed date to make validation pass" in builder_role

    assert "Use the resolved runner's virtualenv Python" in data_skill
    assert "`oxq run python` does not exist" in data_skill
    assert "Do not run `uv run python` in an installed research workspace" in data_skill
    assert "`oxq run python` does not exist" in data_role
    assert "optimizer-internal transform" in auditor


def test_spec_auditor_skill_documents_source_trace_gate() -> None:
    text = _skill_bundle_text("audit-strategy-spec")

    assert "name: audit-strategy-spec" in text
    assert "confirmed" in text
    assert "default" in text
    assert "unconfirmed" in text
    assert "start of the current experiment" in text
    assert "just-finished" in text
    assert "only a checkpoint" in text
    assert "blocks backtest" in text
    assert "group related fields" in text
    assert "CONVERSATION_HISTORY_RAW" in text
    assert "Do not hardcode `conversation.json` as a required path" in text
    assert "data.min_start_date" in text
    assert "data warmup" in text
    assert "Block the audit when lookback behavior exists" in text
    assert "field_path" in text
    assert "agent_added" in text
    assert "spec_audit.json" in text
    assert "oxq spec-audit validate <phase_paths.06_spec_audit>/spec_audit.json" in text
    assert "component_catalog.json" in text
    assert "catalog_hash" in text
    assert '"catalog_hash": "sha256:<component_catalog.catalog_hash>"' not in text
    assert "RiskAdjustedMomentum" in text
    assert "NdayReturn + RollingVolatility + Ratio" in text
    assert "TopNRanking" in text


def test_spec_auditor_requires_all_pass_then_user_confirmed_table_gate() -> None:
    spec_skill = _skill_bundle_text("audit-strategy-spec")
    runtime_skill = Path("agent/skills/audit-runtime-semantics/SKILL.md").read_text(encoding="utf-8")
    spec_role = Path("agent/roles/oxq-spec-auditor-worker.md").read_text(encoding="utf-8")
    runtime_role = Path("agent/roles/oxq-runtime-auditor-worker.md").read_text(encoding="utf-8")
    coordinator_role = Path("agent/roles/oxq-coordinator.md").read_text(encoding="utf-8")
    normalized_spec_skill = " ".join(spec_skill.split())
    normalized_spec_role = " ".join(spec_role.split())
    normalized_coordinator = " ".join(coordinator_role.split())

    assert "Two-Step Spec Audit Gate" in spec_skill
    assert "`audit_conclusion: all_pass`" in spec_skill
    assert "`user_confirmation_status: pending`" in spec_skill
    assert "`status: block`" in spec_skill
    assert "`next_required_phase: user_spec_confirmation`" in spec_skill
    assert "Full Spec Confirmation Table" in spec_skill
    assert "| Section | Field path | Spec value | Source | Audit status | Impact |" in spec_skill
    assert "Do not summarize only blockers" in spec_skill
    assert "the audit remains blocked" in spec_skill
    assert "`user_confirmation_status: confirmed`" in spec_skill
    assert "Only after the user explicitly confirms the full Markdown table" in spec_skill
    assert "`confirmation_event` referencing the durable confirmation log event" in spec_skill
    assert '"confirmation_event": {' in spec_skill
    assert "pre_confirmation_spec_audit_hash" not in spec_skill
    assert '"spec_audit_path": "<phase_paths.06_spec_audit>/spec_audit.json"' in spec_skill
    assert '"spec_audit_hash": "sha256:<pre-confirmation spec_audit hash>"' in spec_skill
    assert "Pending all-pass audits must not\ninvent a confirmation event" in spec_skill
    assert "complete `strategy.py` source" in spec_skill
    assert "`blocking_findings` must\nbe an empty list" in spec_skill
    assert "Do not keep resolved historical blockers" in spec_skill
    assert "empty `missing_user_requirements`" in spec_skill
    assert "do not keep the resolved item in\n`agent_added_fields`" in spec_skill
    assert "spec_confirmation_table.hash" in spec_skill
    assert "from oxq.spec.compiler import _hash_file" in spec_skill
    assert "not with `shasum`" in spec_skill
    assert "hash the body after that marker" in spec_skill
    assert "`strategy_idea_brief.json.conversation_hash`" in spec_skill
    assert "do not hash the entire transcript" in spec_skill
    assert "without a nested `event_hash` field" in spec_skill
    assert "raw JSONL line content" in spec_skill
    assert "not the hash of the parsed JSON object" in normalized_spec_skill
    assert "pre-confirmation audit payload" in spec_skill
    assert "not the final\npost-confirmation `spec_audit.json` hash" in spec_skill
    assert "Do not use the pending Full SPEC Confirmation Table itself as evidence" in spec_skill
    assert "Field value included in full SPEC confirmation table for user approval" in spec_skill
    assert "Run `--strict-confirmed` before returning any\n`audit_conclusion: all_pass`" in spec_skill
    assert "including the user-confirmation-pending state" in spec_skill
    assert "For a pending all-pass audit, `field_audits` must already satisfy strict-confirmed coverage" in spec_skill
    assert "Do not leave any effective field row as `status: default`" in spec_skill
    assert "Avoid evidence wording that says the user did not specify or confirm the field" in spec_skill
    assert "Use the top-level effective field prefix as the `Section` value" in spec_skill
    assert "Represent empty strings as an empty cell" in spec_skill
    assert "it is not downstream runtime or backtest authorization" in spec_skill
    assert "Do not create a placeholder `spec_confirmation_table.md`" in spec_skill
    assert "For `audit_conclusion: blocked`, omit\n`spec_confirmation_table` or set it to `null`" in spec_skill
    assert "required only when the SPEC has\nno audit blockers" in spec_skill
    assert '"Effective StrategySpec default value"' in spec_skill
    assert '"Documented for full SPEC coverage"' in spec_skill
    assert "v003 inherits all v002\nconfirmed values except TopNRanking n=2" in spec_skill

    assert "spec_confirmation_table.md" in spec_role
    assert (
        "spec_confirmation_table.md only when audit_conclusion is all_pass "
        "and user_confirmation_status is pending or confirmed"
    ) in spec_role
    assert "`audit_conclusion: all_pass` and `user_confirmation_status` is pending or confirmed" in normalized_spec_role
    assert "all_pass or user confirmation is pending/confirmed" not in spec_role
    assert "Do not write a placeholder\n  `spec_confirmation_table.md` for `audit_conclusion: blocked`" in spec_role
    assert (
        "Return `spec_confirmation_table.md` only for `audit_conclusion: all_pass` "
        "with pending or confirmed user confirmation"
    ) in normalized_spec_role
    assert "`blocking_findings: []`" in spec_role
    assert "empty `missing_user_requirements`" in spec_role
    assert "not in\n  `agent_added_fields`" in spec_role
    assert "oxq.spec.compiler._hash_file(Path(...))" in spec_role
    assert "all_pass but user-confirmation-pending audit" in spec_role
    assert '"Effective StrategySpec default value"' in spec_role
    assert "Use actual user confirmation evidence" in spec_role
    assert "Do not hand off to `oxq-runtime-auditor-worker`" in spec_role
    assert "`confirmation_event` reference with `path`, `event_id`, `decision: confirmed`" in spec_role
    assert "relay the full Markdown Spec table to the user" in coordinator_role
    assert "`user_spec_confirmation`" in coordinator_role
    assert "`user_confirmation_status: confirmed`" in coordinator_role
    assert "`confirmation_event` reference" in coordinator_role
    for field in ("`path`", "`event_id`", "`line_number`", "`event_hash`", "`artifact_path`", "`artifact_hash`"):
        assert field in coordinator_role
    assert "`phase: spec_confirmation`" in spec_skill
    assert "`field_scope: full_spec_table`" in spec_skill
    assert '"decision": "confirmed"' in spec_skill
    assert "`decision: confirmed`" in coordinator_role
    assert "`decision: confirmed`" in runtime_skill
    assert "`decision: confirmed`" in runtime_role
    assert "json.dumps(candidate, sort_keys=True, default=str)" in spec_skill
    assert "Do not start `oxq-runtime-auditor-worker`" in coordinator_role
    assert "confirmed `spec_audit.json`" in runtime_skill
    assert "`confirmation_event` exists" in runtime_skill
    assert "`schema_version: 4`" in runtime_skill
    assert "spec-audit validate <phase_paths.06_spec_audit>/spec_audit.json" in runtime_skill
    assert "--strict-confirmed" in runtime_skill
    assert "--json" in runtime_skill
    assert "`spec_audit_path` plus `spec_audit_hash`" in runtime_skill
    assert "Do not compare it with a nested\n`event_hash` field inside the JSONL payload" in runtime_skill
    assert "canonical pre-confirmation audit hash" in runtime_skill
    assert "not the current final\npost-confirmation `spec_audit.json` hash" in runtime_skill
    for field in ("`path`", "`event_id`", "`line_number`", "`event_hash`", "`artifact_path`", "`artifact_hash`"):
        assert field in runtime_skill
    assert "print the complete `strategy.py` source" in runtime_skill
    assert "`strategy_source_code`" in runtime_skill
    assert '"schema_version": 2' in runtime_skill
    assert '"strategy_source_path"' in runtime_skill
    assert '"strategy_source_hash"' in runtime_skill
    assert "must not write source-presentation evidence" in runtime_skill
    assert "`strategy_source_printed`" not in runtime_skill
    assert "Return both the `strategy.py` path and the complete source text" in runtime_skill
    assert "confirmed `spec_audit.json`" in runtime_role
    assert "`schema_version: 4`" in runtime_role
    assert "valid `confirmation_event`" in runtime_role
    assert "<resolved_runner> spec-audit validate <phase_paths.06_spec_audit>/spec_audit.json" in runtime_role
    assert "--strict-confirmed --json" in runtime_role
    for field in ("`path`", "`event_id`", "`line_number`", "`event_hash`", "`artifact_path`", "`artifact_hash`"):
        assert field in runtime_role
    assert "`spec_audit_path`" in runtime_role
    assert "`spec_audit_hash`" in runtime_role
    assert "`strategy_source_code`" in runtime_role
    assert "Do not return only the file path" in runtime_role
    assert "must not certify or record user-facing presentation" in runtime_role
    assert "When receiving the runtime auditor result" in coordinator_role
    assert "relay the complete `strategy.py` source to the user" in coordinator_role
    assert "Do not replace it with only a file path" in coordinator_role
    assert "runtime-source-presentations.jsonl" in coordinator_role
    assert "append the presentation event only after" in normalized_coordinator
    assert "`strategy_source_presentation`" in coordinator_role
    assert "conditionally writes `spec_confirmation_table.md` only" in normalized_coordinator
    assert "blocked audits omit it or set `spec_confirmation_table: null`" in normalized_coordinator


def test_spec_auditor_is_read_only_and_returns_mapping_errors_to_builder() -> None:
    auditor = _skill_bundle_text("audit-strategy-spec")
    auditor_role = Path("agent/roles/oxq-spec-auditor-worker.md").read_text(encoding="utf-8")
    builder = _skill_bundle_text("build-strategy-spec")
    builder_role = Path("agent/roles/oxq-strategy-builder-worker.md").read_text(encoding="utf-8")

    assert "Read-Only SPEC Boundary" in auditor
    assert "Do not write, patch, rewrite, normalize, or repair" in auditor
    assert "<phase_paths.04_spec_build>/strategy_spec.yaml" in auditor
    assert "misplaced, ignored, dropped, or mistranslated" in auditor
    assert "block with `next_required_phase: build`" in auditor
    assert "`execution.initial_cash`" in auditor
    assert "Do not convert that finding into a confirmation-table row" in auditor
    assert "The builder must make the YAML change" in auditor
    assert "return to `build-strategy-spec`" in auditor
    assert "User-confirmed source value vs effective value check" in auditor
    assert "before any Default Confirmation Checklist" in auditor
    assert "`portfolio.initial_cash: 1000000`" in auditor
    assert "`execution.initial_cash: 100000.0`" in auditor
    assert "cannot be repaired by user confirmation" in auditor
    assert "do not write `audit_conclusion: all_pass`" in auditor
    assert "Use `StrategySpec.from_yaml(...).to_effective_dict()`" in auditor
    assert "`portfolio.initial_cash` is not an effective StrategySpec field" in auditor
    assert "audit `execution.initial_cash`, not `portfolio.initial_cash`" in auditor
    assert "material_category" in auditor
    assert "`strategy_logic`" in auditor
    assert "`backtest_assumption`" in auditor
    assert "`execution_assumption`" in auditor
    assert "`field_audits` must contain only effective StrategySpec field paths" in auditor
    assert "`source_yaml_path`" in auditor
    assert "`builder_required_fix`" in auditor
    assert "Do not write YAML-only paths such as `portfolio.initial_cash` as `field_audits` rows" in auditor

    assert "<phase_paths.04_spec_build>/strategy_spec.yaml" in auditor_role
    assert "Do not edit, patch, repair, or normalize" in auditor_role
    assert "return `next_required_phase: build`" in auditor_role
    assert "User-confirmed source values must match effective StrategySpec values" in auditor_role
    assert "`portfolio.initial_cash: 1000000`" in auditor_role
    assert "`execution.initial_cash: 100000.0`" in auditor_role
    assert "must not become `user_spec_confirmation`" in auditor_role
    assert "`portfolio.initial_cash` is not an effective field" in auditor_role
    assert "`field_audits` contain only effective StrategySpec field paths" in auditor_role
    assert "`source_yaml_path`" in auditor_role
    assert "`builder_required_fix`" in auditor_role

    assert "SPEC Audit Repair Handoff" in builder
    assert "move the value to the effective field path" in builder
    assert "remove the non-operative YAML path" in builder
    assert "`source_yaml_path`" in builder
    assert "`effective_field_path`" in builder
    assert "rerun `oxq spec validate`" in builder
    assert "SPEC Audit Repair Handoff" in builder_role
    assert "move values to effective field paths" in builder_role


def test_experiment_comparator_skill_documents_cross_run_outputs() -> None:
    skill = Path("agent/skills/compare-experiments/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "name: compare-experiments" in text
    assert "comparisons/" in text
    assert ".open-xquant/workspace.yaml" in text
    assert "paths.comparisons_dir" in text
    assert "paths.comparison_registry" in text
    assert "spec_diff.yaml" in text
    assert "metrics_comparison.json" in text
    assert "comparison_report.md" in text
    assert "equity_overlay" in text
    assert "drawdown_overlay" in text
    assert "metrics_bar" in text
    assert "metrics.json" in text
    assert "execution_assumptions.json" in text
    assert "research_bias_audit.json" in text
    assert "reproducibility_audit.json" in text
    assert "audited comparison requires" in text


def test_open_xquant_router_resumes_writer_after_chart_builder_before_rendering() -> None:
    text = Path("agent/skills/open-xquant/SKILL.md").read_text(encoding="utf-8")

    start = text.index('- "Write the final report":')
    end = text.index('- "Review whether this can be traded":')
    sequence = text[start:end]

    chart_step = sequence.index("`build-report-charts` builds the default professional chart pack")
    resume_step = sequence.index("resume `write-research-report`")
    render_step = sequence.index("render HTML")

    assert chart_step < resume_step < render_step
    assert "do not ask the user whether charts are needed" in sequence
    assert "no_charts_requested" not in sequence


def test_report_writer_role_defaults_to_professional_chart_pack() -> None:
    role = Path("agent/roles/oxq-report-writer-worker.md").read_text(encoding="utf-8")

    assert "chart_decision: default_professional_chart_pack" in role
    assert "Build the Default Professional Chart Pack by default" in role
    assert "Do not ask the user whether charts are needed" in role
    assert "Do not return a successful report without registered chart assets" in role
    assert "no_charts_requested" not in role


def test_research_report_reviewer_skill_covers_semantic_report_qa() -> None:
    skill = Path("agent/skills/review-research-report/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "review-research-report" in text
    assert "decision_policy" in text
    assert "REJECT" in text
    assert "WATCHLIST" in text
    assert "PAPER TRADING CANDIDATE" in text
    assert "audit" in text
    assert "robustness" in text
    assert "numeric_claim_unverified" in text
    assert "optional/advisory numeric QA output" in text
    assert "facts registry" in text
    assert "writer_result.json" in text
    assert "Canonical Report Chart Order" in text
    assert "style is consistent" in text
    assert "unexpectedly switch to an all-English report" in text
    assert "local-language chart labels" in text
    assert "chart" in text
    assert "A final research report without registered chart assets is a blocker" in text
    assert "return the report to `build-report-charts`" in text
    assert "do not rewrite" in text.lower()


def test_report_reviewer_worker_receives_writer_result() -> None:
    role = Path("agent/roles/oxq-report-reviewer-worker.md")

    text = role.read_text(encoding="utf-8")

    assert "writer_result.json" in text
    assert text.index("writer_result.json") < text.index("## Outputs")


def test_report_writer_result_declares_lineage_identity_fields() -> None:
    skill = Path("agent/skills/write-research-report/SKILL.md").read_text(encoding="utf-8")
    role = Path("agent/roles/oxq-report-writer-worker.md").read_text(encoding="utf-8")

    for field in ("version_id", "run_id", "strategy_id", "source_run_dir"):
        assert f'"{field}"' in skill
        assert f"`{field}`" in role


def test_opencode_legacy_agent_command_bundle_is_removed() -> None:
    assert not Path("agent/opencode").exists()
    assert not Path("agent/opencode/agents").exists()
    assert not Path("agent/opencode/commands").exists()


def test_end_to_end_strategy_skills_are_removed() -> None:
    assert not Path("agent/skills/quant-research/SKILL.md").exists()
    assert not Path("agent/skills/strategy-builder-standalone/SKILL.md").exists()


def test_runtime_auditor_skill_documents_compile_consistency_gate() -> None:
    text = Path("agent/skills/audit-runtime-semantics/SKILL.md").read_text(encoding="utf-8")

    assert "audit-runtime-semantics" in text
    assert "<resolved_runner> strategy compile <phase_paths.04_spec_build>/strategy_spec.yaml \\" in text
    assert "--data-dir data" in text
    assert "compiled_plan.json" in text
    assert "runtime_audit.json" in text
    assert "<resolved_runner> runtime-audit validate <phase_paths.08_runtime_audit>/runtime_audit.json" in text
    assert "Runner Resolution" in text
    assert "same `data_dir` and every `component_manifest` path" in text
    assert "--component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json" in text
    assert "--component-manifest <phase_paths.03_component_authoring>/component_manifest.json" not in text
    assert "Omit `--component-manifest` when no workspace-local custom" in text
    assert "components are authorized; repeat it for each authorized bundle manifest" in text
    assert "component_bundle_hashes" in text
    assert "rebalance interval" in text
    assert "runtime_semantics_pass" in text
    assert "from oxq.spec.schema import StrategySpec" in text
    assert "from pathlib import Path" in text
    assert "from oxq.spec.compiler import _hash_json_file" in text
    assert 'spec_path = Path("<phase_paths.04_spec_build>/strategy_spec.yaml")' in text
    assert "Do not import non-existent helpers from `oxq.core.hashing`" in text


def test_strategy_builder_is_build_only_for_multi_agent_systems() -> None:
    text = _skill_bundle_text("build-strategy-spec")

    assert "multi-Agent systems" in text
    assert "Do not:" in text
    assert "produce `spec_audit.json`" in text
    assert "call audit skills" in text
    assert "download market data" in text
    assert "run `oxq strategy compile`" in text
    assert "run `oxq backtest run`" in text
    assert "attach provenance" in text
    assert "experiment" in text
    assert "component_catalog.json" in text
    assert "Search `recipes` before composing custom indicator chains" in text
    assert "validation.required_oos: false" in text
    assert "data.min_start_date" in text
    assert "data_warmup_policy" in text
    assert "builder_phase_result.json" in text
    assert "needs_custom_component" in text
    assert "Do not call component creation skills" in text
    assert "strategy-builder-standalone" not in text
    assert "audit-strategy-spec" not in text


def test_strategy_brainstorm_skill_owns_pre_spec_workflow() -> None:
    text = Path("agent/skills/brainstorm-strategy-idea/SKILL.md").read_text(encoding="utf-8")

    assert "name: brainstorm-strategy-idea" in text
    assert "strategy_idea_brief.json" in text
    assert "Do not run `oxq`" in text
    assert "Do not write or edit `strategy_spec.yaml`" in text
    assert "Explain the phase before asking for values" in text
    assert "Pull the user back to the earliest incomplete phase" in text
    assert "Any default or candidate value must be explicitly confirmed by the user" in text
    assert "Compute `conversation_hash` from the exact raw brainstorm conversation" in text
    assert "Never write `sha256:placeholder`" in text
    assert "strip only leading and trailing" in text
    assert "whitespace from that body" in text
    for phase in [
        "research intent and hypothesis",
        "market, universe, and benchmark",
        "data and evaluation window",
        "Indicator definitions",
        "Signal rule definitions",
        "Portfolio construction",
        "execution, costs, rebalance, and risk constraints",
        "metrics, robustness, and decision policy",
    ]:
        assert phase in text
    for field_path in [
        "`research.hypothesis`",
        "`market.*`",
        "`universe.*`",
        "`data.min_start_date`",
        "`signal.indicators.*`",
        "`signal.rules.*`",
        "`portfolio.type`",
        "`execution.*`",
        "`metrics.*`",
        "`decision_policy.*`",
    ]:
        assert field_path in text


def test_strategy_idea_auditor_skill_audits_brainstorm_process() -> None:
    text = Path("agent/skills/audit-strategy-idea/SKILL.md").read_text(encoding="utf-8")

    assert "name: audit-strategy-idea" in text
    assert "strategy_idea_brief.json" in text
    assert "strategy_idea_audit.json" in text
    assert "Strategy Idea Workflow Audit" in text
    assert "every required brainstorm phase is present" in text
    assert "the brainstormer explained the phase before asking for values" in text
    assert "default or candidate values were explicitly confirmed by the user" in text
    assert "next_required_phase: brainstorm" in text
    assert "Do not use the SHA-256 of an empty string" in text
    assert "strategy_idea_brief.json.conversation_hash" in text
    assert "placeholder" in text
    assert "mismatched" in text
    assert "Canonical hash rule" in text
    assert "CONVERSATION_HISTORY_RAW" in text
    assert "Do not read, write, or edit `strategy_spec.yaml`" in text
    assert "Indicator definitions" in text
    assert "signal.indicators.*" in text


def test_strategy_builder_requires_audited_idea_before_spec_work() -> None:
    text = _skill_bundle_text("build-strategy-spec")

    assert "Audited Idea Input Gate" in text
    assert "strategy_idea_brief.json" in text
    assert "strategy_idea_audit.json" in text
    assert "Do not run `oxq spec init`" in text
    assert "Do not run `oxq registry export`" in text
    assert "Do not write or edit `strategy_spec.yaml`" in text
    assert "before `strategy_idea_audit.json` passes" in text
    assert "next_required_phase: brainstorm" in text
    assert "strategy_idea_brief_hash" in text
    assert "strategy_idea_audit_hash" in text
    assert "Explain the phase before asking for values" not in text
    assert "Pull the user back to the earliest incomplete phase" not in text


def test_strategy_builder_forbids_root_level_spec_initializer() -> None:
    builder = _skill_bundle_text("build-strategy-spec")
    builder_role = Path("agent/roles/oxq-strategy-builder-worker.md").read_text(encoding="utf-8")

    assert "Never run `oxq spec init` without an explicit `--out` path" in builder
    assert "root-level `strategy_spec.yaml`" in builder
    assert "workspace layout violation" in builder
    assert "deleted later" in builder
    assert "record `layout_violation` in `builder_phase_result.json`" in builder
    assert "The only acceptable" in builder
    assert "initializer target is" in builder

    assert "Never run `oxq spec init` without `--out`" in builder_role
    assert "root-level `strategy_spec.yaml`" in builder_role
    assert "workspace layout violation" in builder_role
    assert "later deleted" in builder_role
    assert "record `layout_violation` in" in builder_role


def test_strategy_builder_records_required_open_xquant_version() -> None:
    builder = _skill_bundle_text("build-strategy-spec")
    spec_auditor = _skill_bundle_text("audit-strategy-spec")
    runtime_auditor = Path("agent/skills/audit-runtime-semantics/SKILL.md").read_text(encoding="utf-8")

    assert "OpenXQuant Version Provenance" in builder
    assert "`schema_version` is the SPEC schema version" in builder
    assert "`required_oxq_version` is the OpenXQuant package version" in builder
    assert "Do not change `schema_version` to the package version" in builder
    assert "write `required_oxq_version`" in builder
    assert "builder_phase_result.json" in builder

    assert "`required_oxq_version`" in spec_auditor
    assert "OpenXQuant version provenance" in spec_auditor
    assert "blocks formal backtest" in spec_auditor

    assert "`required_oxq_version`" in runtime_auditor
    assert "open_xquant_version" in runtime_auditor
    assert "environment.json" in runtime_auditor


def test_builder_and_spec_auditor_reject_unconfirmed_effective_defaults() -> None:
    builder = _skill_bundle_text("build-strategy-spec")
    auditor = _skill_bundle_text("audit-strategy-spec")

    assert "`market.region: cn`" in builder
    assert "`market.currency: CNY`" in builder
    assert "`execution.lot_size`" in builder
    assert "`execution.rebalance.frequency`" in builder
    assert "`decision_policy.promote_if`" in builder
    assert "`decision_policy.pass.conditions`" in builder
    assert "`cost_multiplier: [2.0]`" in builder
    assert "Do not write boolean\n  `true` for `cost_multiplier` or `parameter_perturbation`" in builder

    assert "Framework default" in auditor
    assert "`market.region: us`" in auditor
    assert "`market.currency: USD`" in auditor
    assert "`execution.rebalance.interval_days: 1`" in auditor
    assert "next_required_phase: build" in auditor


def test_builder_and_spec_auditor_require_unsupported_mapping_disclosure() -> None:
    builder = _skill_bundle_text("build-strategy-spec")
    auditor = _skill_bundle_text("audit-strategy-spec")
    builder_role = Path("agent/roles/oxq-strategy-builder-worker.md").read_text(encoding="utf-8")
    auditor_role = Path("agent/roles/oxq-spec-auditor-worker.md").read_text(encoding="utf-8")

    assert "spec_mapping_notes.md" in builder
    assert "spec_mapping_contract.json" in builder
    assert "unsupported_mappings" in builder
    assert "unmapped_source_fields" in builder
    assert "threshold_then_rank_top_n" in builder
    assert "Unsupported strategy semantics are blocking by default" in builder
    assert "mapping-contract row must use `blocking: true`" in builder
    assert "strategy source field is marked `needs_user_confirmation`" in builder
    assert "`confirmation_required: true` and `blocking: true`" in builder
    assert "validate_mapping_contract_for_builder_pass_file" in builder
    assert "do not run `oxq spec validate_mapping_contract`" in builder.lower()
    assert "every `semantic: strategy` row must be mapped and non-blocking" in builder
    assert "derive the\nallowed field paths from the effective StrategySpec" in builder
    assert "StrategySpec.from_yaml(\"<phase_paths.04_spec_build>/strategy_spec.yaml\").to_effective_dict()" in builder
    assert "Do not use parent container paths\nlike `portfolio`" in builder
    assert "conceptual absent paths like `execution.leverage.allowed`" in builder
    assert "Every `field_mappings` row must have a non-empty `reason`" in builder
    assert "Do not leave `reason` as an empty string" in builder
    assert "Do not label run, report, studio, or metadata source fields as `semantic: strategy`" in builder
    assert "Use `excluded_non_material` only with `semantic: run`, `report`, `studio`, or `metadata`" in builder
    assert "`signal.indicators.<name>.lag_bars`, not `signal.indicators.<name>.params.lag_bars`" in builder
    assert "Run both validators again after every mapping-contract edit" in builder
    assert "`status: blocked`, `status: unsupported`" in builder
    assert "Do not read the full catalog into the model context" in builder
    assert "jq -r '.catalog_hash'" in builder
    assert '"catalog_hash": "sha256:<component_catalog.catalog_hash>"' not in builder
    assert '"catalog_hash": "<component_catalog.catalog_hash>"' in builder

    assert "unsupported_mappings" in auditor
    assert "spec_mapping_contract.json" in auditor
    assert "--mapping-contract <phase_paths.04_spec_build>/spec_mapping_contract.json" in auditor
    assert "Do not use `--strict-confirmed` for `audit_conclusion: blocked`" in auditor
    assert "Strategy rows with `status: needs_user_confirmation` and `blocking: false`" in auditor
    assert "builder-pass" in auditor
    assert "mapping gate" in auditor
    assert "`blocked`, `unsupported`, `needs_user_confirmation`, or `blocking: true`" in auditor
    assert "do not run `oxq spec validate_mapping_contract`" in auditor.lower()
    assert "blocked" in auditor
    assert "source_field" in auditor
    assert "requested_semantic" in auditor
    assert "disposition" in auditor
    assert "no unsupported source fields were found" in auditor
    assert '"catalog_hash": "sha256:<component_catalog.catalog_hash>"' not in auditor
    assert '"catalog_hash": "<component_catalog.catalog_hash>"' in auditor

    assert "spec_mapping_contract.json" in builder_role
    assert "spec_mapping_contract.json" in auditor_role
    assert "unsupported_mappings" in auditor_role
    assert "Validate `spec_mapping_contract.json`" in builder_role
    assert "validate_mapping_contract_for_builder_pass" in builder_role
    assert "do not run\n  `oxq spec validate_mapping_contract`" in builder_role
    assert "every strategy row to be mapped and" in builder_role
    assert "Do not read the full catalog into context" in builder_role
    assert "Treat strategy mapping-contract rows with `status: needs_user_confirmation`" in builder_role
    assert "Unsupported `strategy` semantics with `blocking: false`" in auditor_role
    assert "Strategy rows with `status: needs_user_confirmation` and `blocking: false`" in auditor_role
    assert "Passing audits must satisfy the builder-pass mapping gate" in auditor_role


def test_spec_auditor_requires_audited_idea_and_calibrates_spec() -> None:
    text = _skill_bundle_text("audit-strategy-spec")

    assert "strategy_idea_brief.json" in text
    assert "strategy_idea_audit.json" in text
    assert "Spec Calibration Audit" in text
    assert "fast-fail" in text
    assert "next_required_phase: brainstorm" in text
    assert "next_required_phase: build" in text
    assert "verify the spec faithfully maps the audited idea" in text
    assert "Indicator definitions" in text
    assert "signal.indicators.*" in text


def test_strategy_idea_artifacts_are_in_worker_handoffs() -> None:
    brainstorm_role = Path("agent/roles/oxq-strategy-brainstorm-worker.md").read_text(encoding="utf-8")
    idea_auditor_role = Path("agent/roles/oxq-strategy-idea-auditor-worker.md").read_text(
        encoding="utf-8"
    )
    builder_role = Path("agent/roles/oxq-strategy-builder-worker.md").read_text(encoding="utf-8")
    spec_auditor_role = Path("agent/roles/oxq-spec-auditor-worker.md").read_text(encoding="utf-8")
    coordinator_role = Path("agent/roles/oxq-coordinator.md").read_text(encoding="utf-8")

    for artifact in ["strategy_idea_brief.json", "strategy_idea_audit.json"]:
        assert artifact in builder_role
        assert artifact in spec_auditor_role
        assert artifact in coordinator_role
    assert "strategy_idea_brief.json" in brainstorm_role
    assert "strategy_idea_audit.json" in idea_auditor_role
    assert "CONVERSATION_HISTORY_RAW" in idea_auditor_role
    assert "sha256:placeholder" in brainstorm_role
    assert "compare it to" in idea_auditor_role
    assert "stripping only leading and" in brainstorm_role
    assert "trailing whitespace" in brainstorm_role
    assert "stripping only leading and trailing" in idea_auditor_role
    assert "whitespace" in idea_auditor_role
    assert "CONVERSATION_HISTORY_RAW" in coordinator_role
    assert "Require passing `strategy_idea_audit.json` before writing `strategy_spec.yaml`" in builder_role
    assert "audit the strategy brainstorm workflow" in idea_auditor_role
    assert "audit the spec calibration against the audited strategy idea" in spec_auditor_role


def test_strategy_monitor_is_post_run_and_uses_runtime_audit() -> None:
    strategy_monitor = Path("agent/skills/monitor-strategy-run/SKILL.md").read_text(encoding="utf-8")
    monitor_role = Path("agent/roles/oxq-monitor-worker.md").read_text(encoding="utf-8")

    assert "_cost_x2" in strategy_monitor
    assert "created sub-run directory" in strategy_monitor
    assert "spec_audit.json" in strategy_monitor
    assert "runtime_audit.json" in strategy_monitor
    assert "component_catalog_hash.txt" in strategy_monitor
    assert "recipe_catalog_hash.txt" in strategy_monitor
    assert "conversation_hash.txt" in strategy_monitor
    assert (
        "oxq spec-audit validate <phase_paths.06_spec_audit>/spec_audit.json"
        in strategy_monitor
    )
    assert 'uv run oxq audit reproducibility "$RUN_DIR" --json --publish' in strategy_monitor
    assert 'uv run oxq audit research "$RUN_DIR" --json --publish' in strategy_monitor
    assert 'uv run oxq robustness run "$RUN_DIR" --json' in strategy_monitor
    assert "Shell redirection into governed artifacts is invalid" in strategy_monitor
    assert "Monitoring is not a separate version phase name" in strategy_monitor
    assert "Do not set `current.json.active_phase`" in strategy_monitor
    assert "Keep the active phase at\n`09_backtests`" in strategy_monitor
    assert "Monitoring is not a standalone active phase" in monitor_role
    assert "keep\n  `09_backtests` until report artifacts" in monitor_role
    assert '"next_phase": "oxq-report-writer-worker"' in strategy_monitor
    assert "The next phase is `oxq-report-writer-worker`" in monitor_role
    assert "Do not stop after monitoring pass" in monitor_role


def test_agent_guide_is_install_only_and_points_to_router_skill() -> None:
    text = Path("docs/agent-guide.md").read_text(encoding="utf-8")

    assert "skill 单一来源是 `agent/skills/*/SKILL.md`" in text
    assert "不要维护 `agent/opencode/`" in text
    assert "OpenCode 专用 skills、agents 或 commands 副本" in text
    assert "open-xquant router skill" in text
    assert "current worktree runner" in text
    assert "preferred_runner" in text
    assert "`~/.config/open-xquant/agent-install.json`" in text
    assert "SDK bundle" in text
    assert "`oxq-component-author-worker`" in text
    assert "`agent_roles`" in text
    assert "用户任务路由" not in text
    assert "Spec 最小模板" not in text
    assert "CLI 速查" not in text
    assert "策略想法或新策略" not in text


def test_human_guide_first_use_prompt_uses_runnable_status_command() -> None:
    text = Path("docs/human-guide.md").read_text(encoding="utf-8")

    assert "安装后运行 uv run oxq agent status" in text
    assert "安装后运行 oxq agent status" not in text


def test_examples_do_not_reference_removed_report_write_command() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in Path("examples").rglob("*.py"))

    assert "oxq report write" not in combined


def test_readme_workflows_do_not_reference_removed_report_write_command() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "oxq report write" not in text


def test_backtest_runner_is_authorized_execution_only() -> None:
    skill = Path("agent/skills/run-authorized-backtest/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "backtest_authorization.json" in text
    assert "`strategy_source_presentation`" in text
    assert "runtime-source-presentations.jsonl" in text
    assert "full source-file SHA-256" in text
    assert "--spec-audit spec_audit.json" in text
    assert "--runtime-audit runtime_audit.json" in text
    assert "--component-catalog component_catalog.json" in text
    assert "--component-manifest <components_dir>/bundles/<bundle_id>/component_manifest.json" in text
    assert "<resolved_runner> backtest run <phase_paths.04_spec_build>/strategy_spec.yaml \\" in text
    assert "Runner Resolution" in text
    assert "Omit `--component-manifest` only when" in text
    assert "same `component_bundle_hashes`" in text
    assert "`user_confirmation_status: confirmed`" in text
    assert "valid `confirmation_event`" in text
    for field in ("`path`", "`event_id`", "`line_number`", "`event_hash`", "`artifact_path`", "`artifact_hash`"):
        assert field in text
    assert "top-level fields above are required" in text.lower()
    assert "only records nested\ndiagnostic hashes" in text
    assert "must not run `oxq registry export`" in text.lower()
    assert "Use the authorized\n`<phase_paths.04_spec_build>/component_catalog.json`" in text
    assert "Do not write `component_catalog.json` outside" in text
    assert "pass\n`Path` objects to `_hash_json_file`" in text
    assert "formal run command attaches `spec_audit.json`" in text
    assert "workspace-local custom components" in text
    assert "runner_result.json" in text
    assert "next_phase" in text
    assert "oxq-monitor-worker" in text
    assert "Do not edit `strategy_spec.yaml`" in text
    assert "Do not edit `spec_audit.json`" in text
    assert "Do not edit `runtime_audit.json`" in text
    assert "Do not run reproducibility" in text

    role = Path("agent/roles/oxq-runner-worker.md").read_text(encoding="utf-8")
    assert "`user_confirmation_status: confirmed`" in role
    assert "valid `confirmation_event` with `path`, `event_id`, `decision: confirmed`" in role
    assert "Do not run\n  `oxq registry export`" in role
    assert "`08_runtime_audit`, `09_backtests`, or any root-level path" in role
    assert "_hash_json_file(Path(...))" in role
    assert role.count("reproducibility_audit.json") >= 2
    assert role.count("research_bias_audit.json") >= 2
    assert role.count("robustness.json") >= 2
    assert "uv run oxq audit reproducibility" not in text
    assert "uv run oxq audit research" not in text
    assert "uv run oxq robustness run" not in text
    assert "uv run oxq experiment add" not in text


def test_pyproject_packages_agent_roles() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"agent/skills" = "agent/skills"' in text
    assert '"agent/roles" = "agent/roles"' in text


def test_report_writer_and_reviewer_require_spec_audit_disclosure() -> None:
    writer = Path("agent/skills/write-research-report/SKILL.md").read_text(encoding="utf-8")
    reviewer = Path("agent/skills/review-research-report/SKILL.md").read_text(encoding="utf-8")

    assert "spec_audit.json" in writer
    assert "selected canonical recipes" in writer
    assert "component provenance" in writer
    assert "data_manifest.json" in writer
    assert "oxq backtest compare-runs" in writer
    assert "data warmup" in writer
    assert "Do not omit blocking or unresolved `spec_audit.json` findings" in writer
    assert "spec_audit.json" in reviewer
    assert "unconfirmed defaults" in reviewer
    assert "selected recipes" in reviewer
    assert "unresolved `spec_audit.json` blockers" in reviewer


def test_report_writer_documents_installed_bundle_facts_api_and_qa_safe_markdown() -> None:
    writer = Path("agent/skills/write-research-report/SKILL.md").read_text(encoding="utf-8")
    html_output = writer[writer.index("## HTML Output"): writer.index("## Red Lines")]
    normalized = " ".join(writer.split())
    normalized_html = " ".join(html_output.split())

    assert "from oxq.report.artifacts import RunArtifacts" in writer
    assert "from oxq.report.facts import build_report_facts" in writer
    assert "build_report_facts(RunArtifacts.load(run_dir))" in writer
    assert "Do not import `RunArtifacts` from `oxq.api`" in writer
    assert "Do not import `RunArtifacts` from `oxq.run.artifacts`" in normalized
    assert "Use the resolved runner's virtualenv Python" in html_output
    assert "`oxq python` does not exist" in normalized_html
    assert "Do not run `uv run python` in an installed research workspace" in normalized_html
    assert "Do not write raw `<` or `>` comparison text" in writer
    assert "report QA strips HTML tags" in writer


def test_version_governed_manifests_are_root_only_but_experiment_registry_is_configured() -> None:
    coordinator = Path("agent/roles/oxq-coordinator.md").read_text(encoding="utf-8")
    router = Path("agent/skills/open-xquant/SKILL.md").read_text(encoding="utf-8")
    governor = Path("agent/skills/govern-research-workspace/SKILL.md").read_text(encoding="utf-8")

    for text in (coordinator, router, governor):
        normalized = " ".join(text.split())
        assert "`.open-xquant/workspace.yaml` is configuration only" in text
        assert "`current.json` and `lineage.json` live at the workspace root" in normalized
        assert "`paths.experiment_registry`" in text
        assert "`experiments.jsonl`" in text
        assert any(phrase in normalized for phrase in ("only when absent", "only when that key is absent"))
        assert "Do not probe `.open-xquant/current.json`" in normalized
        assert ".open-xquant/experiments.jsonl" not in text


def test_comparison_skills_do_not_leave_empty_figures_directories() -> None:
    comparator = Path("agent/skills/compare-strategy-versions/SKILL.md").read_text(encoding="utf-8")
    legacy = Path("agent/skills/compare-experiments/SKILL.md").read_text(encoding="utf-8")
    role = Path("agent/roles/oxq-experiment-comparator-worker.md").read_text(encoding="utf-8")

    for text in (comparator, legacy, role):
        normalized = " ".join(text.split())
        assert "Do not leave `figures/` empty" in text
        assert "If no figure will be generated, do not create the directory" in normalized


def test_workspace_governor_accepts_version_local_run_registry_and_robustness_subruns() -> None:
    text = Path("agent/skills/govern-research-workspace/SKILL.md").read_text(encoding="utf-8")

    assert "root-level `runs/` is not required" in text
    assert "<phase_paths.09_backtests>/run_digests.jsonl" in text
    assert "_cost_x2" in text
    assert "not root-level pollution" in text


def test_workspace_governance_contract_lists_root_phase_pollution_artifacts() -> None:
    skill = Path("agent/skills/govern-research-workspace/SKILL.md").read_text(encoding="utf-8")
    role = Path("agent/roles/oxq-artifact-governor-worker.md").read_text(encoding="utf-8")
    doc = Path("docs/strategy-workflow-artifact-governance.md").read_text(encoding="utf-8")
    required_artifacts = [
        "spec_mapping_notes.md",
        "spec_mapping_contract.json",
        "audit_notes.md",
        "compile_preview/",
        "component_request.json",
        "component_manifest.json",
        "result.json",
    ]

    for artifact in required_artifacts:
        assert artifact in skill
        assert artifact in role
        assert artifact in doc
    assert "routes back to `oxq-spec-auditor-worker`" in doc
    assert "`audit-strategy-spec` updates `spec_audit.json`" in doc
    for input_root in ("<comparisons_dir>/**", "<final_dir>/**"):
        assert input_root in skill
        assert input_root in role
