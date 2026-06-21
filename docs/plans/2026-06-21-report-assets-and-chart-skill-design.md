# Report Assets And Chart Skill Design

## Purpose

Upgrade open-xquant experiment reports so they can serve as notebook-like
research deliverables without depending on Jupyter. The report should be
artifact-first: it reads deterministic run outputs, presents audit and metrics
evidence, and embeds user-approved chart assets.

The framework should not guess which charts users want, and it should not make
chart generation a core report dependency. Instead, open-xquant should provide
asset management and a skill-driven Agent workflow for discussing chart needs,
writing plotting code when requested, registering generated figures, and
rendering those figures in Markdown and HTML reports.

## User Requirements

- Reports support language selection.
- The default report language is Chinese.
- Reports support chart insertion.
- Reports produce both Markdown and HTML versions.
- HTML reports provide a notebook-like static research narrative.
- Users can ask an Agent to recommend or generate charts from experiment
  artifacts.
- Generated chart files and plotting scripts become experiment assets, not
  temporary files.

## Non-Goals

- Do not add Jupyter or `.ipynb` export in this iteration.
- Do not make `oxq report write` generate charts automatically.
- Do not require matplotlib or other plotting packages for basic report
  rendering.
- Do not treat charts as proof of profitability.
- Do not inline arbitrary HTML assets into reports.
- Do not edit backtest artifacts to make reports or audits look better.

## Core Design

Use a B-3 asset registration model:

1. Users or Agents create chart files outside the report generator.
2. `oxq report asset add` copies or registers those files under the run
   directory.
3. open-xquant records chart metadata in `report_assets/manifest.json`.
4. `oxq report write` reads the manifest and renders registered assets in
   Markdown and HTML.
5. A new `report-chart-builder` skill guides Agents through chart requirement
   discussion, plotting script authoring, asset registration, and final report
   generation.

## Run Directory Additions

Each run can contain a report asset area:

```text
runs/<run_id>/
  report_assets/
    manifest.json
    figures/
      equity_vs_benchmark.png
      drawdown.png
    scripts/
      plot_equity_vs_benchmark.py
      plot_drawdown.py
    attachments/
      notes.pdf
```

`figures/` is for image assets that can be embedded.
`scripts/` is for plotting scripts written by the Agent or user.
`attachments/` is for non-image supporting files that should be linked but not
embedded.

## Manifest Schema

First version:

```json
{
  "schema_version": 1,
  "assets": [
    {
      "id": "equity_vs_benchmark",
      "kind": "figure",
      "path": "figures/equity_vs_benchmark.png",
      "title": "策略净值与基准对比",
      "caption": "由 equity_curve.csv 和 benchmark_curve.csv 生成。",
      "section": "results",
      "order": 10,
      "mime_type": "image/png",
      "sha256": "sha256:...",
      "source": {
        "script": "scripts/plot_equity_vs_benchmark.py",
        "input_artifacts": [
          "equity_curve.csv",
          "benchmark_curve.csv"
        ]
      }
    }
  ]
}
```

Rules:

- `id` must be stable, unique within the manifest, and safe as a filename
  stem.
- `kind` supports `figure` and `attachment` in the first version.
- `path` is relative to `report_assets/`.
- `title` is required.
- `caption` is optional but recommended.
- `section` controls report placement.
- `order` controls ordering within a section.
- `sha256` records the copied asset hash.
- `source.script` records the plotting script if one exists.
- `source.input_artifacts` records run artifacts used to create the figure.

Supported embedded image extensions:

- `png`
- `jpg`
- `jpeg`
- `webp`
- `svg`

Other file types can be registered as attachments and linked from the report.

## CLI Additions

Add a nested asset command group:

```bash
oxq report asset add RUN_DIR FILE \
  --id ID \
  --title TITLE \
  --caption CAPTION \
  --section SECTION \
  --order ORDER \
  --source-script SCRIPT \
  --source-artifact ARTIFACT
```

Behavior:

- Validate `RUN_DIR` exists.
- Validate `FILE` exists.
- Infer `kind` from file extension unless explicitly supplied later.
- Copy figure files into `report_assets/figures/`.
- Copy attachment files into `report_assets/attachments/`.
- Copy source scripts into `report_assets/scripts/` when they are outside that
  directory.
- Compute `sha256` for the asset.
- Upsert one manifest entry by `id`.
- Preserve existing unrelated manifest entries.

Add list support:

```bash
oxq report asset list RUN_DIR
```

Behavior:

- Print registered assets in section/order order.
- Include `id`, `kind`, `title`, `path`, and hash.
- Return a clear empty state when no manifest exists.

Upgrade report writing:

```bash
oxq report write RUN_DIR --lang zh --format all
oxq report write RUN_DIR --lang en --format markdown
oxq report write RUN_DIR --lang zh --format html
```

Defaults:

- `--lang zh`
- `--format all`
- Markdown output: `research_report.md`
- HTML output: `research_report.html`

Backward compatibility:

- `generate_report(run_dir)` still returns a Markdown string.
- Existing callers that only need Markdown can keep using that API.
- `oxq report write RUN_DIR` now writes both Markdown and HTML by default.

## Report Rendering

Markdown:

- Insert figures with relative links:

```markdown
![策略净值与基准对比](report_assets/figures/equity_vs_benchmark.png)

图 1. 由 equity_curve.csv 和 benchmark_curve.csv 生成。
```

- Link attachments rather than embedding them.
- Include asset hash and source script details in a compact asset appendix.

HTML:

- Render a static notebook-like report.
- Use sections similar to:
  - 研究问题
  - 实验配置
  - 数据与执行假设
  - 复现实验命令
  - 实验结果
  - 图表资产
  - 审计与稳健性
  - 结论与下一步
  - 可复现清单
- Render figures with `<figure>`, `<img>`, and `<figcaption>`.
- Show source artifacts, plotting script path, and hash for each figure.
- Keep CSS inline or generated locally so the HTML works offline.
- Do not execute code.

No registered assets:

- The report remains valid.
- It should state that no chart assets were registered.
- It should not silently scan random image files unless a future explicit
  compatibility mode is added.

## Language Support

The first implementation should support:

- `zh`
- `en`

Text labels, section headings, and default explanatory text should be stored in
a small message catalog rather than scattered through renderer code.

Metrics, identifiers, file paths, CLI commands, and artifact names remain in
their original technical form.

Chinese is the default because the primary user workflow expects Chinese
research reports.

## ReportBundle Architecture

Introduce a structured report assembly layer:

- `ReportBundle`
  - Reads run artifacts.
  - Holds spec summary, metrics, audits, robustness, execution assumptions,
    manifest assets, and derived report decision.
  - Performs no Markdown or HTML formatting.

- Markdown renderer
  - Converts `ReportBundle` into Markdown.
  - Uses language catalog.
  - Inserts registered figures and attachment links.

- HTML renderer
  - Converts `ReportBundle` into static notebook-like HTML.
  - Uses the same section order and language catalog.
  - Renders figures with metadata.

- Asset manager
  - Reads and writes `report_assets/manifest.json`.
  - Copies files into asset directories.
  - Computes hashes.
  - Validates safe IDs and relative paths.

This keeps artifact reading, asset management, and output formatting separate.

## Agent Skill

Add `agent/skills/report-chart-builder.md`.

Trigger description:

```yaml
name: report-chart-builder
description: >-
  Design and generate chart assets for open-xquant experiment reports; use when
  users ask to add charts to a report, visualize backtest artifacts for a
  report, decide what figures should appear in a notebook-like experiment
  report, or have an Agent write plotting Python code for report assets.
```

Core instructions:

1. Confirm the run directory.
2. Inspect standard artifacts.
3. If the user requests a specific chart, verify required artifacts exist.
4. If the user does not know which charts to use, recommend chart options based
   on available artifacts and ask for confirmation.
5. Write plotting Python code only after chart requirements are clear.
6. Save plotting scripts under `report_assets/scripts/`.
7. Save generated images under `report_assets/figures/`.
8. Register figures with `oxq report asset add`.
9. Run `oxq report write RUN_DIR --lang zh --format all`.
10. Report Markdown, HTML, manifest, figure, and script paths to the user.

Skill red lines:

- Do not invent charts without user confirmation unless the user explicitly
  asks the Agent to recommend charts.
- Do not use charts to override audit failures.
- Do not hide missing source artifacts by plotting different data.
- Do not leave generated figures outside the run directory.
- Do not edit backtest artifacts.

## Main User Workflows

### Workflow 1: User asks Agent to draw a specific chart

User:

```text
用这次实验数据画策略净值 vs 基准，并放进报告。
```

Agent:

1. Reads `equity_curve.csv` and `benchmark_curve.csv`.
2. Writes `report_assets/scripts/plot_equity_vs_benchmark.py`.
3. Runs the script.
4. Produces `report_assets/figures/equity_vs_benchmark.png`.
5. Registers the figure:

```bash
oxq report asset add runs/<run_id>/ \
  runs/<run_id>/report_assets/figures/equity_vs_benchmark.png \
  --id equity_vs_benchmark \
  --title "策略净值与基准对比" \
  --caption "由 equity_curve.csv 和 benchmark_curve.csv 生成。" \
  --section results \
  --order 10 \
  --source-script runs/<run_id>/report_assets/scripts/plot_equity_vs_benchmark.py \
  --source-artifact equity_curve.csv \
  --source-artifact benchmark_curve.csv
```

6. Runs:

```bash
oxq report write runs/<run_id>/ --lang zh --format all
```

### Workflow 2: User asks for a report but does not know charts

User:

```text
帮我生成实验报告。
```

Agent:

1. Confirms the run directory.
2. Checks standard artifacts.
3. Checks `report_assets/manifest.json`.
4. If no chart assets exist, recommends a small chart set based on available
   artifacts:
   - `equity_curve.csv` plus `benchmark_curve.csv`: strategy equity versus
     benchmark.
   - `equity_curve.csv`: drawdown curve.
   - `trades.csv`: trade distribution.
   - `target_weights.csv`: target weight changes.
   - `robustness.json`: robustness summary visualization.
5. Asks the user to confirm the recommended set.
6. Creates scripts, figures, and asset manifest entries only after confirmation.
7. Writes Markdown and HTML reports.

If the user declines charts, the Agent writes reports without registered chart
assets.

## Error Handling

Missing run directory:

- Fail with a direct error.

Missing required chart inputs:

- Explain which artifact is missing.
- Offer a fallback chart only after asking the user.

Duplicate asset ID:

- Default to upsert when `asset add` is called with the same `id`.
- The updated manifest entry should reflect the new file hash and metadata.

Unsafe asset ID:

- Reject IDs containing path separators, empty components, `.` or `..`.

Unsupported image type:

- Register as attachment if safe, or reject with a clear message.

Malformed manifest:

- Do not overwrite silently.
- Report the parse error and ask the user to repair or move the file.

Plotting dependency missing:

- The skill should tell the Agent to install or use available plotting tools
  only when the user asks the Agent to generate charts.
- Basic report writing must still work without plotting dependencies.

## Testing Strategy

Asset manager tests:

- Adds a PNG figure and writes manifest.
- Adds an attachment and links it.
- Upserts an existing ID.
- Rejects unsafe IDs.
- Preserves existing manifest entries.
- Computes stable `sha256`.

CLI tests:

- `oxq report asset add` copies files and prints output path.
- `oxq report asset list` reports empty and populated states.
- `oxq report write` default creates both Markdown and HTML.
- `--format markdown` only writes Markdown.
- `--format html` only writes HTML.
- `--lang zh` uses Chinese headings.
- `--lang en` uses English headings.

Renderer tests:

- Markdown embeds registered figures with relative paths.
- HTML embeds registered figures with captions.
- Reports include asset hash and source script metadata.
- Reports remain valid when no assets exist.

Skill tests:

- Validate frontmatter and naming.
- Check the skill mentions:
  - requirement discussion before chart generation
  - writing scripts to `report_assets/scripts/`
  - registering assets through `oxq report asset add`
  - not treating charts as proof of profitability

Regression tests:

- Existing `generate_report(run_dir)` callers still receive Markdown.
- Existing report decision behavior remains unchanged.

## Documentation Updates

Update:

- `docs/agent-guide.md`
- `docs/human-guide.md`
- `docs/architecture.md`
- `examples/modules/05_report_and_experiment.py`
- `agent/opencode/commands/quant-report.md`
- `agent/opencode/agents/quant-reporter.md`

Document:

- Chinese report default.
- Markdown and HTML outputs.
- Asset registration workflow.
- Agent chart workflow.
- How chart scripts and figures are stored as experiment assets.

## Rollout Plan

1. Add asset manager and manifest schema.
2. Add `report asset add/list` CLI.
3. Add `ReportBundle` and split Markdown rendering from data assembly.
4. Add HTML renderer.
5. Add language catalog for `zh` and `en`.
6. Update `report write` defaults and compatibility behavior.
7. Add `report-chart-builder` skill.
8. Update docs and examples.
9. Run report, CLI, and skill validation tests.

## Open Decisions Resolved

- Default language: Chinese.
- Default output: Markdown and HTML.
- Chart generation: done by user or Agent-written plotting scripts, not by
  `oxq report write`.
- Chart storage: `report_assets/figures/`.
- Plotting script storage: `report_assets/scripts/`.
- Asset registry: `report_assets/manifest.json`.
- Report presentation: only registered assets are inserted.
