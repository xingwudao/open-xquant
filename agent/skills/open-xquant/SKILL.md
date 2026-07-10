---
name: open-xquant
description: >-
  Use when the user asks for any open-xquant or quantitative research task,
  including strategy design, backtesting, factor evaluation, parameter tuning,
  audit, robustness, reports, chart assets, SDK/component development,
  broker/live trading, workspace setup, Agent installation, or when deciding
  which open-xquant skill applies before CLI, SDK, tools, or file writes.
---

# open-xquant Router

This is the mandatory entry skill for open-xquant work. It routes the task to
the most specific open-xquant skill and sets only the minimum runner and
workspace context needed for that handoff.

## Router Contract

- Use this skill first for any open-xquant or quantitative research request.
- This skill routes; it does not replace the leaf skill.
- If a more specific skill applies, load and follow that skill before using
  CLI, SDK, scripts, or file writes.
- Existing artifacts, metrics, loaded context, or a simple-looking task are not
  reasons to skip the matching leaf skill.
- The only CLI commands this router may run before leaf-skill handoff are the
  minimal runner/workspace commands explicitly listed in this skill, such as
  `doctor`, `research init`, `research init --sdk`, and `agent status`.
- Do not run other `oxq` commands, import `oxq`, edit specs, create charts, or
  write report files directly from this router.
- Do not write report files directly. Route final report writing to
  `write-research-report`.

If you catch yourself thinking "I know which command to run directly", stop
and load the leaf skill first.

## Runner And Workspace

Resolve the runner before the leaf skill runs commands:

Do not resolve a runner for `brainstorm-strategy-idea` or `audit-strategy-idea`
unless a later routed skill will actually run `oxq`. These two skills forbid
`oxq` commands, SDK imports, and SPEC writes; reading runner metadata before
them only adds permission surface and can interrupt a pure conversation gate.

1. If the current directory is an open-xquant source worktree, or the user is
   developing the framework itself, use the current worktree runner such as
   `uv run oxq` or `uv run --project . oxq`.
2. Otherwise read `~/.config/open-xquant/agent.yaml`.
3. Prefer `preferred_runner_argv` when the shell tool accepts argv; otherwise
   use `preferred_runner`.
4. If that metadata is missing or fails, read
   `~/.config/open-xquant/agent-install.json` and use
   `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`.
5. Keep the shell in the user's research directory. Do not search unrelated
   home directories for another open-xquant checkout.

If a research directory lacks `.open-xquant/workspace.yaml`, this router may
initialize it with the resolved runner before loading the leaf skill that will
create strategy artifacts. Use `research init --sdk` when the user will write
SDK-based custom research code.

`.open-xquant/workspace.yaml` is configuration only. `current.json`,
`lineage.json`, and `experiments.jsonl` live at the workspace root. Do not probe
`.open-xquant/current.json` or other hidden-directory manifest paths when
checking active version, lineage, or experiment registry state.

## Task Routing

- Agent install, upgrade, uninstall, cached runner, or target directory
  questions: use the embedded "Install And Upgrade Questions" section below.
  If the source checkout is available, `docs/agent-guide.md` has the longer
  installation guide, but installed Agents must not depend on that file.
- Workspace artifact governance, misplaced root-level phase artifacts, messy
  research directories, or pre-handoff layout checks: use
  `govern-research-workspace`.
- version governance for a new strategy family, a user strategy edit, a
  semantic change, or deciding whether to create a new version or append a run:
  use `manage-strategy-version`.
- Artifact lineage checks across version/run/final references before
  comparison, migration, or final selection: use `audit-artifact-lineage`.
- New strategy idea or incomplete strategy description: use
  `brainstorm-strategy-idea`.
- Audit a completed strategy idea brief or brainstorm process: use
  `audit-strategy-idea`.
- Strategy spec creation from a passing `strategy_idea_audit.json`, or spec
  validation only: use `build-strategy-spec`.
- Multi-Agent workflows use narrow leaf skills only. Do not route worker tasks
  to end-to-end skills; the downstream system owns phase ordering and
  orchestration.
- Spec field provenance, SPEC calibration against an audited idea, or
  pre-backtest assumption confirmation: use `audit-strategy-spec`.
- SPEC-to-runtime compile consistency or `compiled_plan.json` material field
  checks: use `audit-runtime-semantics`.
- Authorized backtest execution from gated artifacts: use
  `run-authorized-backtest`.
- Universe, symbols, index membership, survivorship, or tradable pool design:
  use `build-universe`.
- Data availability, parquet inspection, downloads, or data quality: use
  `explore-data`.
- Costs, slippage, fill price, order timing, lot size, cash return, or broker
  simulation assumptions: use `configure-trade-execution`.
- Stop loss, take profit, drawdown guard, holding limit, rebalance constraint,
  or risk rule: use `build-rule`.
- Post-backtest audit, reproducibility, research bias, robustness, or
  experiment logging: use `monitor-strategy-run`.
- Performance interpretation or "did this strategy work?": use
  `review-performance`.
- Factor IC, Rank IC, ICIR, decay, hit rate, or factor predictiveness: use
  `evaluate-factor`.
- Value, quality, momentum, multi-factor screening, or candidate-list
  generation: use `screen-factors`.
- Parameter optimization, grid search, walk-forward, or overfitting checks:
  use `tune-parameters`.
- Indicator overlays or quick chart inspection: use `plot-indicators`.
- Report chart assets, figure requirements, plotting scripts, image QA, or
  registering generated figures: use `build-report-charts`.
- Cross-run experiment comparison, spec diff, metric comparison, or comparison
  report: use `compare-experiments`.
- Cross-version strategy comparison, within-version reproducibility
  comparison, or comparison in a version-governed workspace: use
  `compare-strategy-versions`.
- Final version selection, final research candidate selection, or requests to
  choose/promote/mark a version as final: use `select-final-version`.
- Final human-readable report writing or editing `research_report.md` /
  `research_report.html`: use `write-research-report`.
- Semantic review of a completed report, decision consistency, audit fidelity,
  robustness interpretation, or chart narrative quality: use
  `review-research-report`.
- New workspace-local Indicator, Signal, or PortfolioOptimizer component: use
  `author-component`. Workspace-local custom Rule requests must block unless
  the user explicitly asks for OpenXQuant framework development that adds
  audited spec, compile, runtime, and backtest support; in that framework case
  use `create-component`.
- Broker connectivity, paper trading, live trading, account checks, or order
  submission: use `manage-live-trading`.

## Install And Upgrade Questions

This section must work from an installed Agent home or a later research
directory where the original source checkout may have been deleted.

Use the cached metadata first:

1. Read `~/.config/open-xquant/agent.yaml` for `preferred_runner_argv` or
   `preferred_runner`.
2. If missing, read `~/.config/open-xquant/agent-install.json` for
   `sdk_bundle.runner.argv` or `sdk_bundle.runner.oxq`.
3. Do not search unrelated home directories for another checkout.
4. If the current directory is an open-xquant source worktree and the user is
   installing from that checkout, use `uv run oxq ...`.

Common commands:

```bash
<runner> agent status
<runner> agent install --all-targets
<runner> agent install --all-targets --profile multi-agent --yes
<runner> agent install --all-targets --profile standalone-agent --yes
<runner> agent install --repair --yes
<runner> agent upgrade --all-targets --yes
<runner> agent upgrade --all-targets --from-local . --yes
<runner> agent uninstall --all-targets --yes
<runner> agent uninstall --all-targets --purge-config --yes
```

When installing interactively, ask the user to choose one profile for this
machine:

- `multi-agent`: recommended when the Agent supports multi-Agent or subagent
  workflows; installs narrow leaf skills and prebuilt OpenXQuant worker roles
  where the target has an official agent-role directory.
- `standalone-agent`: for a single Agent that should orchestrate the same
  narrow phase skills itself; does not install prebuilt worker roles.

With `--yes`, the installer uses the recommended profile unless `--profile` is
provided explicitly.

Supported target skill roots:

- Codex: `${CODEX_HOME:-~/.codex}/skills/`
- OpenCode: `~/.config/opencode/skills/`
- Claude Code: `~/.claude/skills/`
- Cursor: `~/.cursor/skills/`
- OpenClaw: `~/.openclaw/skills/`
- TRAE: `~/.trae/skills/`

Supported target role roots for `multi-agent` profile:

- Codex: `${CODEX_HOME:-~/.codex}/agents/*.toml`
- OpenCode: `~/.config/opencode/agents/*.md`
- Claude Code: `~/.claude/agents/*.md`
- Cursor: `~/.cursor/agents/*.md`

Prebuilt roles are `oxq-coordinator`, `oxq-version-manager-worker`,
`oxq-artifact-governor-worker`, `oxq-strategy-brainstorm-worker`,
`oxq-strategy-idea-auditor-worker`, `oxq-strategy-builder-worker`,
`oxq-data-inspection-worker`, `oxq-component-author-worker`,
`oxq-spec-auditor-worker`, `oxq-runtime-auditor-worker`, `oxq-runner-worker`,
`oxq-monitor-worker`, `oxq-report-writer-worker`,
`oxq-report-reviewer-worker`, `oxq-lineage-auditor-worker`,
`oxq-experiment-comparator-worker`, and `oxq-final-selector-worker`.

The installed skills are flat peer directories such as `open-xquant/`,
`brainstorm-strategy-idea/`, `build-strategy-spec/`, and
`write-research-report/`. Do not nest leaf skills under `open-xquant/`.

## Common Sequences

- "Build and test this idea":
  `manage-strategy-version` -> `brainstorm-strategy-idea` ->
  `audit-strategy-idea` ->
  `build-strategy-spec` -> `explore-data` when data availability is unknown ->
  `audit-strategy-spec` -> user confirmation of the full SPEC table ->
  `audit-runtime-semantics` ->
  `run-authorized-backtest` -> `monitor-strategy-run` ->
  `build-report-charts` builds the default professional chart pack ->
  `write-research-report` ->
  `review-research-report`.
- "Compare two experiments":
  `compare-experiments` -> review `comparisons/<comparison_id>/`.
- "Compare strategy versions":
  `audit-artifact-lineage` -> `compare-strategy-versions` -> review
  `comparisons/<comparison_id>/`.
- "Select a final version":
  `audit-artifact-lineage` -> user confirmation of selection policy ->
  `select-final-version` -> review `final/current_final.json`.
- "Generate charts for this run":
  `build-report-charts` -> update report through `write-research-report` ->
  run deterministic `oxq report qa` -> use `review-research-report`.
- "Write the final report":
  `write-research-report` resolves missing chart decisions to
  `chart_decision: default_professional_chart_pack` and writes a blocked result
  if registered chart assets are missing or stale ->
  `build-report-charts` builds the default professional chart pack; do not ask the user whether charts are needed ->
  resume `write-research-report` to draft or update `research_report.md` with
  registered assets -> render HTML from the same Markdown -> deterministic
  `oxq report qa` -> `review-research-report`.
- "Review whether this can be traded":
  `review-performance` and `review-research-report`; route to
  `manage-live-trading` only after the user explicitly asks for broker execution.

## Red Lines

- Do not bypass a matching leaf skill after recognizing it applies.
- Do not treat this router as permission to run commands directly.
- Do not skip `oxq spec validate`, reproducibility audit, research audit, or
  report QA when the routed workflow requires them.
- Do not modify metrics, audit, robustness, or backtest artifacts to improve a
  narrative.
- Do not promote unaudited or failed research to paper/live trading.
