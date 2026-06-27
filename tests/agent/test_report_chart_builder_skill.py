from __future__ import annotations

from pathlib import Path

import yaml


def test_report_chart_builder_skill_documents_chart_asset_workflow() -> None:
    skill = Path("agent/skills/build-report-charts/SKILL.md")

    text = skill.read_text(encoding="utf-8")

    assert "build-report-charts" in text
    assert "discuss chart requirements" in text
    assert "plotting Python" in text
    assert "report_assets/figures" in text
    assert "report_assets/scripts" in text
    assert "oxq report asset add" in text
    assert "oxq report asset add-batch" in text
    assert "source_artifacts" in text
    assert "trade_curve" in text
    assert "research_report.md" in text
    assert "research_report.html" in text
    assert "write-research-report" in text
    assert "oxq report write" not in text
    assert "report_evidence.md" not in text
    assert "Do not modify metrics" in text
    assert "Do not modify audit" in text
    assert "Prefer English chart labels" in text
    assert "default to English labels" in text
    assert "non-empty" in text
    assert "dimensions" in text
    assert "manifest" in text
    assert "Chart Applicability Matrix" in text
    assert "Violin Plot" in text
    assert "Pair Plot" in text
    assert "scan the run directory" in text
    assert "recommended chart set" in text
    assert "trade curve as the first/default recommendation" in text
    assert "Prefer `seaborn`" in text
    assert "fall back to direct `matplotlib`" in text
    assert "import seaborn as sns" in text
    assert "except ImportError" in text
    assert 'matplotlib.use("Agg")' in text
    assert "Numeric claim review is semantic/advisory" in text
    assert "treating the CLI command as proof" in text


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
    assert default_pack.index("- trade curve") < default_pack.index("- equity curve vs benchmark")
    assert "The trade curve is the default choice" in default_pack


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
    assert "report_writer_result.json" in text
    assert "missing_chart_decision" in text
    assert "build-report-charts" in text
    assert "Evidence is generated by the framework; the narrative is authored by the Agent" in text


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
    assert "Install And Upgrade Questions" in text
    assert "installed Agents must not depend on that file" in text
    assert "<runner> agent status" in text
    for leaf_skill in [
        "build-strategy-spec",
        "audit-strategy-spec",
        "audit-runtime-semantics",
        "run-authorized-backtest",
        "monitor-strategy-run",
        "build-report-charts",
        "compare-experiments",
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
    assert "strategy-builder-standalone" not in text
    assert "quant-research" not in text


def test_coordinator_role_documents_subagent_workflow() -> None:
    role = Path("agent/roles/oxq-coordinator.md")

    text = role.read_text(encoding="utf-8")

    assert "open-xquant SubAgent workflow" in text
    assert "Prefer SubAgents by default" in text
    assert "Builder writes `strategy_spec.yaml`" in text
    assert "Data inspector checks required symbols" in text
    assert "oxq-data-inspection-worker" in text
    assert "Spec auditor reads those artifacts" in text
    assert "Runtime auditor reads the authorized spec/audit artifacts" in text
    assert "Runner reads `backtest_authorization.json`" in text
    assert "Main agent only coordinates" in text


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
    assert "custom_components/" in text
    assert "component_manifest.json" in text
    assert "result.json" in text
    assert "oxq component-manifest hash" in text
    assert "oxq component-manifest validate" in text
    assert "--component-manifest component_manifest.json" in text
    assert "Workspace-local `Rule` authoring is currently blocked" in text
    assert "Do not emit `component_ready` for a workspace-local custom `Rule`" in text
    assert "Do not build or edit `strategy_spec.yaml`" in text
    assert "Do not modify the installed SDK bundle" in text
    assert "role_kind: component_author" in role_text
    assert "author-component" in role_text
    assert "create-rule" not in role_text
    assert "Block workspace-local custom `Rule` requests" in role_text
    assert "forbidden_outputs" in role_text


def test_spec_auditor_skill_documents_source_trace_gate() -> None:
    skill = Path("agent/skills/audit-strategy-spec/SKILL.md")

    text = skill.read_text(encoding="utf-8")

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
    assert "oxq spec-audit validate spec_audit.json" in text
    assert "component_catalog.json" in text
    assert "catalog_hash" in text
    assert "RiskAdjustedMomentum" in text
    assert "NdayReturn + RollingVolatility + Ratio" in text
    assert "TopNRanking" in text


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

    chart_step = sequence.index("`build-report-charts` when chart assets are required")
    resume_step = sequence.index("resume `write-research-report`")
    render_step = sequence.index("render HTML")

    assert chart_step < resume_step < render_step


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
    assert "local-language labels" in text
    assert "chart" in text
    assert "do not rewrite" in text.lower()


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
    assert "oxq strategy compile strategy_spec.yaml \\" in text
    assert "--data-dir data" in text
    assert "compiled_plan.json" in text
    assert "runtime_audit.json" in text
    assert "oxq runtime-audit validate runtime_audit.json" in text
    assert "same `data_dir` and every `component_manifest` path" in text
    assert "component_bundle_hashes" in text
    assert "rebalance interval" in text
    assert "runtime_semantics_pass" in text


def test_strategy_builder_is_build_only_for_multi_agent_systems() -> None:
    text = Path("agent/skills/build-strategy-spec/SKILL.md").read_text(encoding="utf-8")

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


def test_strategy_monitor_is_post_run_and_uses_runtime_audit() -> None:
    strategy_monitor = Path("agent/skills/monitor-strategy-run/SKILL.md").read_text(encoding="utf-8")

    assert "_cost_x2" in strategy_monitor
    assert "created sub-run directory" in strategy_monitor
    assert "spec_audit.json" in strategy_monitor
    assert "runtime_audit.json" in strategy_monitor
    assert "component_catalog_hash.txt" in strategy_monitor
    assert "recipe_catalog_hash.txt" in strategy_monitor
    assert "conversation_hash.txt" in strategy_monitor
    assert "oxq spec-audit validate runs/<run_id>/spec_audit.json" in strategy_monitor


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
    assert "--spec-audit spec_audit.json" in text
    assert "--runtime-audit runtime_audit.json" in text
    assert "--component-catalog component_catalog.json" in text
    assert "--component-manifest component_manifest.json" in text
    assert "Omit `--component-manifest` only when" in text
    assert "component_manifests.json" in text
    assert "same `component_bundle_hashes`" in text
    assert "formal run command attaches `spec_audit.json`" in text
    assert "do not rerun the" in text
    assert "workspace-local custom components" in text
    assert "runner_result.json" in text
    assert "Do not edit `strategy_spec.yaml`" in text
    assert "Do not edit `spec_audit.json`" in text
    assert "Do not edit `runtime_audit.json`" in text


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
