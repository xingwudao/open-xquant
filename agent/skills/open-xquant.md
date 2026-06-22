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
- Do not run `oxq`, import `oxq`, edit specs, create charts, or write report
  files directly from this router.
- Do not write report files directly. Route final report writing to
  `research-report-writer`.

If you catch yourself thinking "I know which command to run directly", stop
and load the leaf skill first.

## Runner And Workspace

Resolve the runner before the leaf skill runs commands:

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

If a research directory lacks `.open-xquant/workspace.yaml`, initialize it with
the resolved runner before creating strategy artifacts. Use
`research init --sdk` when the user will write SDK-based custom research code.

## Task Routing

- Agent install, upgrade, uninstall, cached runner, or target directory
  questions: read `docs/agent-guide.md`. That document is installation
  guidance, not the research workflow.
- New strategy idea, strategy spec creation, spec validation, or audited
  backtest workflow: use `strategy-builder`.
- Complete idea-to-report research workflow: use `quant-research`.
- Universe, symbols, index membership, survivorship, or tradable pool design:
  use `universe-builder`.
- Data availability, parquet inspection, downloads, or data quality: use
  `data-explorer`.
- Costs, slippage, fill price, order timing, lot size, cash return, or broker
  simulation assumptions: use `trade-executor`.
- Stop loss, take profit, drawdown guard, holding limit, rebalance constraint,
  or risk rule: use `rule-builder`.
- Post-backtest audit, reproducibility, research bias, robustness, or
  experiment logging: use `strategy-monitor`.
- Performance interpretation or "did this strategy work?": use
  `performance-reviewer`.
- Factor IC, Rank IC, ICIR, decay, hit rate, or factor predictiveness: use
  `factor-evaluator`.
- Parameter optimization, grid search, walk-forward, or overfitting checks:
  use `parameter-tuner`.
- Indicator overlays or quick chart inspection: use `chart-indicator`.
- Report chart assets, figure requirements, plotting scripts, image QA, or
  registering generated figures: use `report-chart-builder`.
- Final human-readable report writing or editing `research_report.md` /
  `research_report.html`: use `research-report-writer`.
- Semantic review of a completed report, decision consistency, audit fidelity,
  robustness interpretation, or chart narrative quality: use
  `research-report-reviewer`.
- New Indicator, Signal, Rule, or PortfolioOptimizer component: use
  `component-creator`, then follow its routed creation skill.
- Broker connectivity, paper trading, live trading, account checks, or order
  submission: use `live-trader`.

## Common Sequences

- "Build and test this idea":
  `strategy-builder` -> `strategy-monitor` -> `report-chart-builder` when
  figures are needed -> `research-report-writer` ->
  `research-report-reviewer`.
- "Generate charts for this run":
  `report-chart-builder` -> update report through `research-report-writer` ->
  run deterministic `oxq report qa` -> use `research-report-reviewer`.
- "Write the final report":
  `research-report-writer` -> render HTML from the same Markdown ->
  deterministic `oxq report qa` -> `research-report-reviewer`.
- "Review whether this can be traded":
  `performance-reviewer` and `research-report-reviewer`; route to
  `live-trader` only after the user explicitly asks for broker execution.

## Red Lines

- Do not bypass a matching leaf skill after recognizing it applies.
- Do not treat this router as permission to run commands directly.
- Do not skip `oxq spec validate`, reproducibility audit, research audit, or
  report QA when the routed workflow requires them.
- Do not modify metrics, audit, robustness, or backtest artifacts to improve a
  narrative.
- Do not promote unaudited or failed research to paper/live trading.
