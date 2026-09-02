# open-xquant AI 量化 SEO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立中文优先、可抓取、可验证的 open-xquant SEO 入口，包括仓库文案、VitePress 中文站、Skills 和 CLI 能力索引、GitHub Pages 部署及发布后观测。

**Architecture:** GitHub 元数据和 README 提供仓库入口信号；`website/` 作为独立 VitePress 静态站提供主题集群。Python 生成器从 `agent/skills/*/SKILL.md`、中文扩展数据和 Click 命令树生成事实页面，CI 在部署前验证内容覆盖、静态 HTML 元数据、链接与命名。

**Tech Stack:** Python 3.12、pytest、PyYAML、Click、Node.js 22、npm、VitePress 1.6.4、GitHub Actions、GitHub Pages

**Spec:** `docs/superpowers/specs/2026-09-02-ai-quant-seo-design.md`

## Global Constraints

- 只使用 canonical 名称 `open-xquant`、`equant-py` 和 `ebacktestcraft-py`。
- README 第一段必须包含自然、准确的 `AI 量化研究框架` 定位。
- GitHub description 必须使用规格中批准的中英双语文案。
- 站点生产地址固定为 `https://xingwudao.github.io/open-xquant/`。
- VitePress `base` 固定为 `/open-xquant/`。
- Python 最低版本保持 `>=3.12`。
- Node.js 固定使用 major version `22`。
- VitePress 固定使用稳定版 `1.6.4`，通过 `package-lock.json` 锁定完整依赖树。
- Node 依赖只服务 `website/`，不得进入 open-xquant Python 运行时依赖。
- Skills 数量必须从 `agent/skills/*/SKILL.md` 动态计算。
- CLI 工具只统计公开 Click 命令树的叶子命令，group 节点不计数。
- 首期外部文案不得宣传固定 Skills 数量或工具总数。
- 生成页面必须由事实源和中文扩展数据生成，不得手工修改生成文件。
- 每个可索引页面必须有唯一 title、description、canonical 和 H1。
- 不发布 `docs/superpowers/`、`docs/local/` 或其他内部工作文档。
- Search Console 排名和索引数据是观测结果，不作为构建成功条件。

---

## File Map

`README.md`
: 仓库首屏中文定位和中文站入口。

`pyproject.toml`
: Python distribution 的中英双语描述和英文关键词。

`docs/seo/baseline-2026-09-02.md`
: 上线前仓库元数据、查询样本和可验证数量基线。

`website/data/skills.zh.yaml`
: 35 个 Skills 的人工审阅中文扩展字段。

`website/scripts/seo_docs.py`
: 事实源解析、合并、渲染和漂移检查的纯函数。

`website/scripts/generate.py`
: 生成器命令入口，支持写入模式和 `--check` 模式。

`website/scripts/validate_site.py`
: 对 VitePress 生产构建执行 HTML SEO 契约检查。

`website/.vitepress/config.mts`
: 站点路由、导航、sitemap、主题和构建配置。

`website/.vitepress/seo.mts`
: canonical、Open Graph 和 JSON-LD head 生成逻辑。

`website/.vitepress/theme/`
: 默认主题扩展和紧凑、可访问的文档站样式。

`website/guide/`、`website/workflows/`、`website/faq/`
: 手工审阅的中文主题内容。

`website/skills/`、`website/tools/index.md`
: 生成器维护的 Skills 和 CLI 页面。

`tests/docs/`
: 仓库文案、生成器、站点配置和构建验证测试。

`.github/workflows/docs-pages.yml`
: pull request 构建检查和 `main` 分支 Pages 部署。

`docs/seo/release-runbook.md`
: Pages、GitHub 元数据和 Search Console 发布步骤。

---

### Task 1: Lock Repository Positioning And Baseline

**Files:**
- Create: `tests/docs/test_seo_repository.py`
- Create: `docs/seo/baseline-2026-09-02.md`
- Modify: `README.md:1`
- Modify: `pyproject.toml:8`
- Modify: `pyproject.toml:15`

**Interfaces:**
- Consumes: approved copy in the design spec and current repository metadata.
- Produces: exact positioning strings reused by site content and release checks.

- [ ] **Step 1: Write failing repository SEO contract tests**

Create `tests/docs/test_seo_repository.py`:

```python
from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[2]
POSITIONING = (
    "open-xquant 是中文友好的 AI 量化研究框架，面向 AI Coding Agent "
    "和人类量化研究者，提供策略回测、因子研究、稳健性检验、"
    "审计报告与实盘交易工作流。"
)
PROJECT_DESCRIPTION = (
    "AI 量化研究框架 | Agentic Quant Research Kernel for reproducible "
    "and auditable research"
)
PROJECT_KEYWORDS = {
    "ai-quant",
    "quantitative-finance",
    "quant-research",
    "ai-agents",
    "backtesting",
    "factor-research",
    "algorithmic-trading",
}


def test_readme_leads_with_ai_quant_positioning() -> None:
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# open-xquant"
    assert lines[2] == POSITIONING
    assert "https://xingwudao.github.io/open-xquant/" in "\n".join(lines[:30])


def test_python_project_metadata_matches_positioning() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    assert project["description"] == PROJECT_DESCRIPTION
    assert set(project["keywords"]) == PROJECT_KEYWORDS


def test_baseline_records_reproducible_observations() -> None:
    text = (ROOT / "docs/seo/baseline-2026-09-02.md").read_text(
        encoding="utf-8"
    )
    for heading in (
        "## Repository Metadata",
        "## Search Samples",
        "## Inventory",
        "## Measurement Rules",
    ):
        assert heading in text
```

- [ ] **Step 2: Run the contract test and confirm the expected failure**

Run:

```bash
uv run pytest tests/docs/test_seo_repository.py -v
```

Expected: FAIL because README and `pyproject.toml` still use the old copy and
the baseline file does not exist.

- [ ] **Step 3: Replace the README first-screen copy**

Keep `# open-xquant` as the only H1. Replace the current first paragraph with:

```text
open-xquant 是中文友好的 AI 量化研究框架，面向 AI Coding Agent 和人类量化研究者，提供策略回测、因子研究、稳健性检验、审计报告与实盘交易工作流。
```

Immediately follow it with this technical distinction:

```text
它是面向 AI Agent 的 Agentic Quant Research Kernel：用声明式规格、确定性执行和可审计产物，把交易想法转化为可复现的量化研究流程。
```

Add this link beside the existing Human Guide and Agent Guide links:

```markdown
→ **[中文文档站](https://xingwudao.github.io/open-xquant/)**（AI 量化工作流与 Agent Skills）
```

- [ ] **Step 4: Update Python project metadata**

Set these exact values in `pyproject.toml`:

```toml
description = "AI 量化研究框架 | Agentic Quant Research Kernel for reproducible and auditable research"
keywords = [
    "ai-quant",
    "quantitative-finance",
    "quant-research",
    "ai-agents",
    "backtesting",
    "factor-research",
    "algorithmic-trading",
]
```

Do not remove or repurpose the existing optional dependency groups in this
task.

- [ ] **Step 5: Record the dated baseline**

Create `docs/seo/baseline-2026-09-02.md` with these observations:

```markdown
# AI 量化 SEO Baseline - 2026-09-02

## Repository Metadata

- Repository: `https://github.com/xingwudao/open-xquant`
- Description: empty at observation time
- Homepage: empty at observation time
- Topics: none at observation time

## Search Samples

- Query `AI 量化框架`: open-xquant was not present in the sampled leading results.
- Query `AI Agent 量化研究框架`: open-xquant was not present in the sampled leading results.
- Query `AI 量化 open-xquant`: the GitHub repository was returned.
- Query `site:github.com/xingwudao/open-xquant "AI 量化"`: the GitHub repository was returned.
- These are sampled observations, not a stable rank-tracking dataset.

## Inventory

- Agent Skills: 35 source files under `agent/skills/*/SKILL.md`.
- Public CLI leaf commands: 31 at observation time.
- External copy does not publish either count; CI computes current values.

## Measurement Rules

- Record Search Console impressions, clicks, CTR, average position and indexed pages weekly.
- Use a 6 to 8 week first evaluation window after production deployment.
- Do not treat one browser session or one search screenshot as ranking proof.
```

- [ ] **Step 6: Run focused and naming tests**

Run:

```bash
uv run pytest tests/docs/test_seo_repository.py tests/contracts/test_brand_naming.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit the positioning contract**

```bash
git add README.md pyproject.toml docs/seo/baseline-2026-09-02.md tests/docs/test_seo_repository.py
git commit -m "docs: establish ai quant positioning"
```

---

### Task 2: Build The Deterministic Documentation Generator

**Files:**
- Create: `website/__init__.py`
- Create: `website/scripts/__init__.py`
- Create: `website/scripts/seo_docs.py`
- Create: `website/scripts/generate.py`
- Create: `tests/docs/test_seo_docs.py`

**Interfaces:**
- Consumes: repository root, Skill front matter, `skills.zh.yaml`, and `oxq.cli.main.main`.
- Produces: `SkillRecord`, `CommandRecord`, rendered output mapping, write mode, and drift-check mode.

- [ ] **Step 1: Write failing parser and drift-check tests**

Create `tests/docs/test_seo_docs.py` with temporary fixtures and these cases:

```python
from __future__ import annotations

from pathlib import Path

import click
import pytest

from website.scripts.generate import build_outputs
from website.scripts.seo_docs import (
    ContentError,
    check_outputs,
    collect_leaf_commands,
    load_skill_extensions,
    load_skill_sources,
    merge_skill_records,
    parse_skill_source,
    render_outputs,
)


ROOT = Path(__file__).parents[2]


def _write_skill(root: Path, name: str, description: str) -> Path:
    path = root / "agent" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return path


def test_parse_skill_source_reads_front_matter(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, "audit-sample", "Audit one sample.")
    source = parse_skill_source(path, tmp_path)
    assert source.name == "audit-sample"
    assert source.description == "Audit one sample."
    assert source.source_path == "agent/skills/audit-sample/SKILL.md"


def test_merge_requires_exact_extension_coverage(tmp_path: Path) -> None:
    source = parse_skill_source(
        _write_skill(tmp_path, "audit-sample", "Audit one sample."),
        tmp_path,
    )
    with pytest.raises(ContentError, match="missing Chinese extension"):
        merge_skill_records((source,), {})


def test_collect_leaf_commands_excludes_groups() -> None:
    @click.group()
    def root() -> None:
        pass

    @root.group()
    def audit() -> None:
        pass

    @audit.command(help="Validate one artifact.")
    def validate() -> None:
        pass

    assert collect_leaf_commands(root) == (
        ("audit validate", "Validate one artifact."),
    )


def test_check_outputs_reports_generated_drift(tmp_path: Path) -> None:
    target = tmp_path / "website" / "skills" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_text("stale\n", encoding="utf-8")
    with pytest.raises(ContentError, match="generated content drift"):
        check_outputs(tmp_path, {Path("website/skills/index.md"): "current\n"})


def test_render_outputs_returns_repo_relative_paths() -> None:
    assert all(not path.is_absolute() for path in render_outputs((), ()).keys())
```

- [ ] **Step 2: Run the generator tests and confirm import failure**

Run:

```bash
uv run pytest tests/docs/test_seo_docs.py -v
```

Expected: FAIL because `website.scripts.seo_docs` does not exist.

- [ ] **Step 3: Implement the generator domain model**

Create immutable records in `website/scripts/seo_docs.py`:

```python
@dataclass(frozen=True)
class SkillSource:
    name: str
    description: str
    source_path: str


@dataclass(frozen=True)
class SkillExtension:
    title_zh: str
    summary_zh: str
    group: str
    use_cases_zh: tuple[str, ...]
    inputs_zh: tuple[str, ...]
    outputs_zh: tuple[str, ...]
    constraints_zh: tuple[str, ...]
    related_workflows: tuple[str, ...]


@dataclass(frozen=True)
class SkillRecord:
    source: SkillSource
    extension: SkillExtension


@dataclass(frozen=True)
class CommandRecord:
    path: str
    summary: str


class ContentError(ValueError):
    pass
```

Implement these exact callable interfaces:

```text
parse_skill_source(path: Path, repo_root: Path) -> SkillSource
load_skill_sources(repo_root: Path) -> tuple[SkillSource, ...]
load_skill_extensions(path: Path) -> dict[str, SkillExtension]
merge_skill_records(sources, extensions) -> tuple[SkillRecord, ...]
collect_leaf_commands(root: click.Command) -> tuple[tuple[str, str], ...]
render_outputs(skills, commands) -> dict[Path, str]
write_outputs(repo_root: Path, outputs: dict[Path, str]) -> None
check_outputs(repo_root: Path, outputs: dict[Path, str]) -> None
```

Implementation rules:

- Split Skill front matter only on the first closing `---` delimiter.
- Parse front matter with `yaml.safe_load` and require string `name` and
  `description` values.
- Require the Skill directory name to equal front matter `name`.
- Sort Skills and commands by canonical name/path before rendering.
- Reject duplicate names, missing extensions, orphan extensions and empty
  Chinese fields with `ContentError`.
- Traverse `click.Group.commands` recursively and emit only non-group leaves.
- Normalize rendered text to UTF-8 with one final newline.
- In check mode, compare exact bytes and report missing, stale and unexpected
  generated files without writing to disk.

- [ ] **Step 4: Add the command entry point**

Create `website/scripts/generate.py` with this interface:

```python
def build_outputs(repo_root: Path) -> dict[Path, str]:
    sources = load_skill_sources(repo_root)
    extensions = load_skill_extensions(repo_root / "website/data/skills.zh.yaml")
    skills = merge_skill_records(sources, extensions)
    command_rows = collect_leaf_commands(main)
    commands = tuple(CommandRecord(path=path, summary=summary) for path, summary in command_rows)
    return render_outputs(skills, commands)


def main_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    outputs = build_outputs(repo_root)
    if args.check:
        check_outputs(repo_root, outputs)
    else:
        write_outputs(repo_root, outputs)
    return 0
```

Catch `ContentError` only at the `if __name__ == "__main__"` boundary, print
one concise message to stderr and exit with status 1.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/docs/test_seo_docs.py -v
```

Expected: PASS.

- [ ] **Step 6: Run static checks and commit**

Run:

```bash
uv run ruff check website/scripts tests/docs/test_seo_docs.py
uv run mypy website/scripts
```

Expected: PASS.

Commit:

```bash
git add website/__init__.py website/scripts tests/docs/test_seo_docs.py
git commit -m "feat: add deterministic seo content generator"
```

---

### Task 3: Add Complete Chinese Skill Coverage

**Files:**
- Create: `website/data/skills.zh.yaml`
- Generate: `website/skills/index.md`
- Generate: `website/skills/*.md`
- Modify: `tests/docs/test_seo_docs.py`

**Interfaces:**
- Consumes: `SkillSource` records and exact-name entries in `skills.zh.yaml`.
- Produces: one Skills index and one Chinese detail page per source Skill.

- [ ] **Step 1: Add failing full-coverage tests**

Add the exact current source set to `tests/docs/test_seo_docs.py`:

```python
EXPECTED_SKILLS = {
    "audit-artifact-lineage",
    "audit-runtime-semantics",
    "audit-strategy-idea",
    "audit-strategy-spec",
    "author-component",
    "brainstorm-strategy-idea",
    "build-report-charts",
    "build-rule",
    "build-strategy-spec",
    "build-universe",
    "compare-experiments",
    "compare-strategy-versions",
    "configure-trade-execution",
    "create-component",
    "create-indicator",
    "create-portfolio-optimizer",
    "create-rule",
    "create-signal",
    "evaluate-cross-sectional",
    "evaluate-factor",
    "evaluate-time-series",
    "explore-data",
    "govern-research-workspace",
    "manage-live-trading",
    "manage-strategy-version",
    "monitor-strategy-run",
    "open-xquant",
    "plot-indicators",
    "review-performance",
    "review-research-report",
    "run-authorized-backtest",
    "screen-factors",
    "select-final-version",
    "tune-parameters",
    "write-research-report",
}


def test_production_skill_extensions_cover_sources_exactly() -> None:
    sources = load_skill_sources(ROOT)
    extensions = load_skill_extensions(ROOT / "website/data/skills.zh.yaml")
    assert {source.name for source in sources} == EXPECTED_SKILLS
    assert set(extensions) == EXPECTED_SKILLS
    assert len(merge_skill_records(sources, extensions)) == len(EXPECTED_SKILLS)


def test_every_skill_page_has_search_and_evidence_sections() -> None:
    outputs = build_outputs(ROOT)
    skill_pages = {
        path: text
        for path, text in outputs.items()
        if path.parent == Path("website/skills") and path.name != "index.md"
    }
    assert len(skill_pages) == len(EXPECTED_SKILLS)
    for path, text in skill_pages.items():
        assert "description:" in text, path
        assert "# " in text, path
        assert "## 适用场景" in text, path
        assert "## 输入" in text, path
        assert "## 输出" in text, path
        assert "## 约束" in text, path
        assert "## 源文件" in text, path
```

- [ ] **Step 2: Run the coverage tests and confirm missing-data failure**

Run:

```bash
uv run pytest tests/docs/test_seo_docs.py -v
```

Expected: FAIL because `website/data/skills.zh.yaml` and generated pages do
not exist.

- [ ] **Step 3: Author the complete Chinese extension file**

Use this schema for every exact Skill key:

```yaml
open-xquant:
  title_zh: open-xquant AI 量化研究路由
  summary_zh: 根据量化研究请求选择正确的工作流，并在运行确定性工具前完成边界检查。
  group: 研究治理
  use_cases_zh:
    - 用户提出策略、回测、因子、审计、稳健性、报告或实盘研究请求。
  inputs_zh:
    - 用户目标、研究目录和当前版本状态。
  outputs_zh:
    - 明确的下一阶段 Skill 路由和前置条件。
  constraints_zh:
    - 路由完成前不运行 CLI、SDK 或报告脚本。
  related_workflows:
    - /guide/agentic-quant-research
    - /workflows/strategy-backtest
```

Use these exact title and group assignments for the remaining records:

```yaml
govern-research-workspace: [量化研究工作区治理, 研究治理]
manage-strategy-version: [量化策略版本管理, 研究治理]
compare-strategy-versions: [量化策略版本比较, 研究治理]
select-final-version: [量化策略最终版本选择, 研究治理]
compare-experiments: [量化实验结果比较, 研究治理]
brainstorm-strategy-idea: [量化策略想法梳理, 策略与审计]
audit-strategy-idea: [量化策略想法审计, 策略与审计]
build-strategy-spec: [量化策略规格构建, 策略与审计]
audit-strategy-spec: [量化策略规格审计, 策略与审计]
audit-runtime-semantics: [量化运行语义审计, 策略与审计]
audit-artifact-lineage: [量化产物血缘审计, 策略与审计]
explore-data: [量化数据探索, 数据与因子]
build-universe: [量化标的池构建, 数据与因子]
evaluate-factor: [量化因子评估, 数据与因子]
evaluate-cross-sectional: [横截面因子评估, 数据与因子]
evaluate-time-series: [时间序列因子评估, 数据与因子]
screen-factors: [量化因子筛选, 数据与因子]
tune-parameters: [量化参数调优, 数据与因子]
create-component: [量化组件创建, 组件开发]
author-component: [量化组件实现, 组件开发]
create-indicator: [量化指标创建, 组件开发]
create-rule: [量化规则创建, 组件开发]
build-rule: [量化规则实现, 组件开发]
create-signal: [量化信号创建, 组件开发]
create-portfolio-optimizer: [量化组合优化器创建, 组件开发]
plot-indicators: [量化指标可视化, 组件开发]
run-authorized-backtest: [授权量化回测执行, 执行与报告]
review-performance: [量化绩效复核, 执行与报告]
build-report-charts: [量化报告图表构建, 执行与报告]
write-research-report: [量化研究报告撰写, 执行与报告]
review-research-report: [量化研究报告审阅, 执行与报告]
configure-trade-execution: [量化交易执行配置, 执行与报告]
manage-live-trading: [量化实盘交易管理, 执行与报告]
monitor-strategy-run: [量化策略运行监控, 执行与报告]
```

For each record, read its corresponding `agent/skills/<name>/SKILL.md` and
write non-empty source-backed values for all schema fields. Constraints must
preserve the source Skill's red lines, confirmation gates and path rules.
Do not translate a capability into a guarantee that the source does not make.

- [ ] **Step 4: Implement Skills index and detail rendering**

Render every detail page with the same field order as this concrete
`open-xquant` example:

```markdown
---
title: open-xquant AI 量化研究路由
description: 根据量化研究请求选择正确的工作流，并在运行确定性工具前完成边界检查。
outline: deep
---

# open-xquant AI 量化研究路由

根据量化研究请求选择正确的工作流，并在运行确定性工具前完成边界检查。

## 适用场景

- 用户提出策略、回测、因子、审计、稳健性、报告或实盘研究请求。

## 输入

- 用户目标、研究目录和当前版本状态。

## 输出

- 明确的下一阶段 Skill 路由和前置条件。

## 约束

- 路由完成前不运行 CLI、SDK 或报告脚本。

## 关联工作流

- [AI Agent 量化研究](/guide/agentic-quant-research)
- [AI 量化回测](/workflows/strategy-backtest)

## 源文件

[查看 canonical Skill 定义](https://github.com/xingwudao/open-xquant/blob/main/agent/skills/open-xquant/SKILL.md)
```

Render `website/skills/index.md` grouped in this order:

```text
研究治理
策略与审计
数据与因子
组件开发
执行与报告
```

The index may display the dynamically calculated current Skill count in the
page body. Do not place the count in title or description.

- [ ] **Step 5: Generate and verify all Skill pages**

Run:

```bash
uv run python -m website.scripts.generate
uv run python -m website.scripts.generate --check
uv run pytest tests/docs/test_seo_docs.py -v
```

Expected: all commands exit 0 and exactly one detail page exists for each
source Skill.

- [ ] **Step 6: Commit the Skills topic cluster**

```bash
git add website/data/skills.zh.yaml website/skills tests/docs/test_seo_docs.py
git commit -m "docs: add chinese agent skill topic cluster"
```

---

### Task 4: Generate The Public CLI Capability Index

**Files:**
- Generate: `website/tools/index.md`
- Modify: `website/scripts/seo_docs.py`
- Modify: `tests/docs/test_seo_docs.py`

**Interfaces:**
- Consumes: `oxq.cli.main.main` as a Click command tree.
- Produces: sorted `CommandRecord` rows and a generated CLI capability page.

- [ ] **Step 1: Add failing CLI integration tests**

Add these tests:

```python
from oxq.cli.main import main as oxq_main


def test_production_cli_inventory_contains_only_leaf_commands() -> None:
    rows = collect_leaf_commands(oxq_main)
    paths = {path for path, _summary in rows}
    assert "audit" not in paths
    assert "backtest" not in paths
    assert "spec" not in paths
    assert "audit research" in paths
    assert "backtest run" in paths
    assert "spec validate" in paths
    assert all(summary.strip() for _path, summary in rows)


def test_tool_index_avoids_unstable_marketing_counts() -> None:
    outputs = build_outputs(ROOT)
    text = outputs[Path("website/tools/index.md")]
    assert "# open-xquant CLI 能力索引" in text
    assert "70 个工具" not in text
    assert "公开 CLI 叶子命令" in text
```

- [ ] **Step 2: Run the tests and confirm renderer failure**

Run:

```bash
uv run pytest tests/docs/test_seo_docs.py -v
```

Expected: FAIL because the tool index renderer is absent or incomplete.

- [ ] **Step 3: Render commands by top-level group**

Render `website/tools/index.md` from `CommandRecord` values with:

```markdown
---
title: open-xquant CLI 能力索引
description: 可验证的 open-xquant 公开 CLI 叶子命令，覆盖策略规格、回测、审计、稳健性、报告和研究治理。
---

# open-xquant CLI 能力索引

本页从当前 Click 命令树生成。统计对象是公开 CLI 叶子命令，命令组本身不计入。

## audit

### `oxq audit research`

Run research bias audit on a backtest run directory.
```

Normalize multiline Click help into one readable paragraph, preserve command
paths verbatim, and sort groups and commands lexicographically.

- [ ] **Step 4: Regenerate and verify drift mode**

Run:

```bash
uv run python -m website.scripts.generate
uv run python -m website.scripts.generate --check
uv run pytest tests/docs/test_seo_docs.py -v
```

Expected: PASS. The current source tree produces 31 command entries, but the
page title and description do not publish that number.

- [ ] **Step 5: Commit the CLI index**

```bash
git add website/tools/index.md website/scripts/seo_docs.py tests/docs/test_seo_docs.py
git commit -m "docs: generate public cli capability index"
```

---

### Task 5: Author The Chinese Search-Intent Pages

**Files:**
- Create: `website/index.md`
- Create: `website/guide/ai-quant-framework.md`
- Create: `website/guide/agentic-quant-research.md`
- Create: `website/guide/reproducible-quant-research.md`
- Create: `website/workflows/strategy-backtest.md`
- Create: `website/workflows/factor-research.md`
- Create: `website/workflows/research-audit.md`
- Create: `website/workflows/robustness-testing.md`
- Create: `website/workflows/live-trading.md`
- Create: `website/examples/index.md`
- Create: `website/faq/index.md`
- Create: `website/404.md`
- Create: `tests/docs/test_seo_content.py`

**Interfaces:**
- Consumes: approved keyword clusters, existing guides, contracts, examples and runtime behavior.
- Produces: one landing page per primary Chinese search intent with unique metadata and evidence links.

- [ ] **Step 1: Write failing content-contract tests**

Create `tests/docs/test_seo_content.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
PAGES = {
    "website/index.md": ("AI 量化框架 | open-xquant", "AI 量化研究框架"),
    "website/guide/ai-quant-framework.md": ("什么是 AI 量化框架 | open-xquant", "AI 量化框架"),
    "website/guide/agentic-quant-research.md": ("AI Agent 量化研究 | open-xquant", "AI Agent 量化研究"),
    "website/guide/reproducible-quant-research.md": ("可复现量化研究 | open-xquant", "可复现量化研究"),
    "website/workflows/strategy-backtest.md": ("AI 量化回测 | open-xquant", "AI 量化回测"),
    "website/workflows/factor-research.md": ("AI 因子研究 | open-xquant", "AI 因子研究"),
    "website/workflows/research-audit.md": ("量化回测审计 | open-xquant", "量化回测审计"),
    "website/workflows/robustness-testing.md": ("量化策略稳健性检验 | open-xquant", "量化策略稳健性检验"),
    "website/workflows/live-trading.md": ("AI 量化实盘交易 | open-xquant", "AI 量化实盘交易"),
    "website/examples/index.md": ("量化研究示例 | open-xquant", "量化研究示例"),
    "website/faq/index.md": ("AI 量化常见问题 | open-xquant", "AI 量化常见问题"),
}


def _front_matter(text: str) -> dict[str, object]:
    assert text.startswith("---\n")
    _empty, raw, _body = text.split("---", 2)
    payload = yaml.safe_load(raw)
    assert isinstance(payload, dict)
    return payload


def test_search_intent_pages_have_unique_metadata_and_h1() -> None:
    titles: set[str] = set()
    descriptions: set[str] = set()
    for relative, (title, h1) in PAGES.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        metadata = _front_matter(text)
        assert metadata["title"] == title
        assert isinstance(metadata["description"], str)
        assert len(metadata["description"]) >= 45
        assert f"# {h1}" in text
        titles.add(str(metadata["title"]))
        descriptions.add(str(metadata["description"]))
    assert len(titles) == len(PAGES)
    assert len(descriptions) == len(PAGES)


def test_homepage_uses_real_architecture_asset_and_source_links() -> None:
    text = (ROOT / "website/index.md").read_text(encoding="utf-8")
    assert "/images/open-xquant-subagent-collaboration.png" in text
    assert "https://github.com/xingwudao/open-xquant" in text
    assert "/workflows/strategy-backtest" in text
    assert "/skills/" in text
```

- [ ] **Step 2: Run the content tests and confirm missing-page failure**

Run:

```bash
uv run pytest tests/docs/test_seo_content.py -v
```

Expected: FAIL because the search-intent pages do not exist.

- [ ] **Step 3: Author homepage and three guide pages**

Use the exact title/H1 pairs from the test. Give each page a unique Chinese
description of 45 to 90 characters.

Homepage sections, in order:

```text
open-xquant definition and boundary
idea -> audit -> spec -> confirmation -> backtest -> report -> final
real architecture image
strategy backtest, factor research, audit, robustness and live workflow links
Agent Skills grouped entry
quick start links to Human Guide, Agent Guide and GitHub
```

The homepage first paragraph must use the approved README positioning. Embed
the architecture image with stable dimensions:

```html
<img
  src="/images/open-xquant-subagent-collaboration.png"
  width="1920"
  height="1080"
  alt="open-xquant 中协调 Agent、专业 SubAgent 与确定性量化研究内核的协作关系"
>
```

Guide page evidence sources:

```text
guide/ai-quant-framework -> README.md, docs/architecture.md
guide/agentic-quant-research -> docs/agent-guide.md, agent/skills/open-xquant/SKILL.md
guide/reproducible-quant-research -> docs/human-guide.md, docs/strategy-workflow-artifact-governance.md
```

Each guide must include `适合谁`, `解决什么问题`, `工作边界`, `下一步` and
at least two contextual internal links.

- [ ] **Step 4: Author five workflow pages**

Use these required sections and evidence sources:

```text
strategy-backtest:
  sections: 输入, 授权门槛, 确定性执行, 回测产物, 常见失败
  sources: agent/skills/run-authorized-backtest/SKILL.md, docs/human-guide.md

factor-research:
  sections: 因子假设, 数据要求, 横截面评估, 时间序列评估, 筛选与调优
  sources: agent/skills/evaluate-factor/SKILL.md, agent/skills/screen-factors/SKILL.md

research-audit:
  sections: 想法审计, 规格审计, 运行语义审计, 偏差审计, 产物血缘
  sources: docs/strategy-workflow-artifact-governance.md, agent/skills/audit-artifact-lineage/SKILL.md

robustness-testing:
  sections: 为什么需要稳健性检验, 成本压力, 参数扰动, 样本外验证, 结果解释
  sources: README.md, src/oxq/robustness.py

live-trading:
  sections: 研究与实盘边界, 认证要求, Broker 配置, 监控, 停止条件
  sources: agent/skills/configure-trade-execution/SKILL.md, agent/skills/manage-live-trading/SKILL.md
```

Do not claim automatic profitability, risk elimination or unattended live
trading. Every page must link to at least one relevant Skill detail page.

- [ ] **Step 5: Author examples and FAQ pages**

The examples index must link to existing repository examples and state data
dependency, command entry, expected artifact type and limitation for each
listed example.

The FAQ must answer these exact questions in visible H2 headings:

```text
open-xquant 是 AI 交易机器人吗？
AI 生成的量化策略为什么还需要审计？
相同策略为什么必须能够复现？
open-xquant 能直接用于实盘交易吗？
Agent Skills 和 CLI 分别负责什么？
如何开始第一次量化研究？
```

Create `website/404.md` with `title: 页面未找到`, one H1, a `noindex` robots
meta tag, and links back to `/`, `/guide/ai-quant-framework` and `/skills/`.

- [ ] **Step 6: Run content and naming tests**

Run:

```bash
uv run pytest tests/docs/test_seo_content.py tests/contracts/test_brand_naming.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit the Chinese topic pages**

```bash
git add website/index.md website/404.md website/guide website/workflows website/examples website/faq tests/docs/test_seo_content.py
git commit -m "docs: add ai quant search intent pages"
```

---

### Task 6: Build VitePress And Technical SEO

**Files:**
- Create: `website/package.json`
- Create: `website/package-lock.json`
- Create: `website/.node-version`
- Create: `website/.vitepress/config.mts`
- Create: `website/.vitepress/seo.mts`
- Create: `website/.vitepress/theme/index.ts`
- Create: `website/.vitepress/theme/custom.css`
- Create: `website/public/robots.txt`
- Create: `website/public/images/open-xquant-subagent-collaboration.png`
- Create: `tests/docs/test_site_config.py`

**Interfaces:**
- Consumes: all hand-authored and generated Markdown pages.
- Produces: static HTML under `website/.vitepress/dist` with canonical, social metadata, JSON-LD and sitemap.

- [ ] **Step 1: Write failing site configuration contracts**

Create `tests/docs/test_site_config.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_node_and_vitepress_are_pinned() -> None:
    package = json.loads((ROOT / "website/package.json").read_text(encoding="utf-8"))
    assert package["private"] is True
    assert package["type"] == "module"
    assert package["engines"]["node"] == ">=22 <23"
    assert package["devDependencies"]["vitepress"] == "1.6.4"
    assert (ROOT / "website/.node-version").read_text(encoding="utf-8") == "22\n"


def test_site_config_locks_production_urls() -> None:
    config = (ROOT / "website/.vitepress/config.mts").read_text(encoding="utf-8")
    seo = (ROOT / "website/.vitepress/seo.mts").read_text(encoding="utf-8")
    assert "base: '/open-xquant/'" in config
    assert "titleTemplate: false" in config
    assert "provider: 'local'" in config
    assert "https://xingwudao.github.io/open-xquant/" in config
    assert "https://xingwudao.github.io" in seo
    assert "rel: 'canonical'" in seo
    assert "application/ld+json" in seo


def test_robots_declares_production_sitemap() -> None:
    robots = (ROOT / "website/public/robots.txt").read_text(encoding="utf-8")
    assert robots == (
        "User-agent: *\n"
        "Allow: /\n"
        "Sitemap: https://xingwudao.github.io/open-xquant/sitemap.xml\n"
    )
```

- [ ] **Step 2: Run contracts and confirm missing-config failure**

Run:

```bash
uv run pytest tests/docs/test_site_config.py -v
```

Expected: FAIL because the Node and VitePress files do not exist.

- [ ] **Step 3: Add the isolated Node project**

Create `website/package.json`:

```json
{
  "name": "open-xquant-docs",
  "private": true,
  "type": "module",
  "engines": {
    "node": ">=22 <23"
  },
  "scripts": {
    "docs:generate": "cd .. && uv run python -m website.scripts.generate",
    "docs:check": "cd .. && uv run python -m website.scripts.generate --check",
    "docs:dev": "npm run docs:generate && vitepress dev .",
    "docs:build": "npm run docs:generate && vitepress build .",
    "docs:preview": "vitepress preview .",
    "docs:validate": "cd .. && uv run python -m website.scripts.validate_site website/.vitepress/dist"
  },
  "devDependencies": {
    "vitepress": "1.6.4"
  }
}
```

Write `22` plus one newline to `website/.node-version`, then run:

```bash
npm install --prefix website --package-lock-only
npm ci --prefix website
```

Expected: dependency installation succeeds and `package-lock.json` records
VitePress 1.6.4.

- [ ] **Step 4: Implement canonical and structured-data head generation**

In `website/.vitepress/seo.mts`, export:

```typescript
export const SITE_ORIGIN = 'https://xingwudao.github.io'
export const SITE_BASE = '/open-xquant/'

export function canonicalFor(relativePath: string): string
export function seoHead(context: TransformContext): HeadConfig[]
```

`canonicalFor` must remove `index.md` and `.md`, preserve nested paths, prepend
`SITE_BASE`, and return an absolute URL. `seoHead` must emit:

```text
link rel=canonical
og:title
og:description
og:type
og:url
og:locale=zh_CN
og:image
application/ld+json
```

For `index.md`, JSON-LD is an array containing `WebSite` and
`SoftwareSourceCode`. For every other page, JSON-LD is `BreadcrumbList` with
the homepage and current page. Use only title, description, production URL,
repository URL and MIT license facts visible on the site.

- [ ] **Step 5: Configure VitePress**

Use `defineConfig` with these exact core values:

```typescript
export default defineConfig({
  lang: 'zh-CN',
  title: 'open-xquant',
  description: '面向 AI Agent 和量化研究者的中文友好 AI 量化研究框架',
  base: '/open-xquant/',
  titleTemplate: false,
  cleanUrls: true,
  lastUpdated: true,
  sitemap: {
    hostname: 'https://xingwudao.github.io/open-xquant/'
  },
  transformHead: seoHead,
  ignoreDeadLinks: false
})
```

Configure nav entries for 首页、AI 量化框架、工作流、Agent Skills、CLI
能力 and GitHub. Configure sidebars for `/guide/`, `/workflows/`, `/skills/`
and `/tools/`. Enable VitePress local search with `provider: 'local'` and
Chinese button, empty-result, reset and keyboard-navigation labels. Keep
sidebar labels concise and use the titles from Task 5.

- [ ] **Step 6: Extend the default theme and reuse the architecture image**

Import `DefaultTheme` and `custom.css` in
`website/.vitepress/theme/index.ts`. Use these design tokens:

```css
:root {
  --vp-c-brand-1: #087f5b;
  --vp-c-brand-2: #0b6b50;
  --vp-c-brand-3: #095c45;
  --vp-c-tip-1: #087f5b;
  --vp-c-warning-1: #b45309;
  --vp-c-danger-1: #b42318;
  --vp-border-radius: 6px;
}
```

Keep body text at a fixed readable size, use neutral backgrounds, set visible
focus styles, constrain content width, and prevent long command names from
overflowing on mobile. Do not add gradients, decorative blobs, nested cards or
oversized marketing typography.

Copy the existing tracked image
`docs/images/open-xquant-subagent-collaboration.png` to
`website/public/images/open-xquant-subagent-collaboration.png`. Do not create
a replacement illustration.

- [ ] **Step 7: Add robots and build the production site**

Create `website/public/robots.txt` with the exact content asserted by the
test, then run:

```bash
uv run pytest tests/docs/test_site_config.py -v
npm --prefix website run docs:check
npm --prefix website run docs:build
```

Expected: PASS; VitePress reports a successful build with no dead internal
links.

- [ ] **Step 8: Verify generated production artifacts**

Run:

```bash
test -s website/.vitepress/dist/index.html
test -s website/.vitepress/dist/guide/ai-quant-framework.html
test -s website/.vitepress/dist/sitemap.xml
test -s website/.vitepress/dist/robots.txt
rg -n "AI 量化框架 \| open-xquant|rel=\"canonical\"|application/ld\+json" website/.vitepress/dist/index.html
```

Expected: every file is non-empty and the homepage contains all three SEO
signals.

- [ ] **Step 9: Verify desktop and mobile rendering**

Start the local server:

```bash
npm --prefix website run docs:dev -- --host 127.0.0.1
```

Inspect the homepage, one workflow page and one generated Skill page at
`1440x900` and `390x844`. Verify the architecture image is visible, navigation
and local search work, long command names wrap, focus states are visible, and
no text or controls overlap.

- [ ] **Step 10: Commit the VitePress site**

```bash
git add website/package.json website/package-lock.json website/.node-version website/.vitepress website/public tests/docs/test_site_config.py
git commit -m "feat: build vitepress seo site"
```

---

### Task 7: Validate Built HTML And Deploy Through CI

**Files:**
- Create: `website/scripts/validate_site.py`
- Create: `tests/docs/test_validate_site.py`
- Create: `.github/workflows/docs-pages.yml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `website/.vitepress/dist` and committed generated Markdown.
- Produces: deterministic validation exit status and a Pages deployment artifact.

- [ ] **Step 1: Write failing HTML validator tests**

Create `tests/docs/test_validate_site.py` with temporary HTML fixtures:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from website.scripts.validate_site import SiteValidationError, validate_site


VALID_HTML = """<!doctype html>
<html lang="zh-CN"><head>
<title>AI 量化框架 | open-xquant</title>
<meta name="description" content="中文友好的 AI 量化研究框架，覆盖回测、因子、审计和稳健性工作流。">
<link rel="canonical" href="https://xingwudao.github.io/open-xquant/">
<meta property="og:title" content="AI 量化框架 | open-xquant">
<meta property="og:description" content="中文友好的 AI 量化研究框架，覆盖回测、因子、审计和稳健性工作流。">
<meta property="og:url" content="https://xingwudao.github.io/open-xquant/">
<meta property="og:image" content="https://xingwudao.github.io/open-xquant/images/open-xquant-subagent-collaboration.png">
<script type="application/ld+json">{"@type":"WebSite"}</script>
</head><body><h1>AI 量化研究框架</h1></body></html>"""


def _write_site(tmp_path: Path, html: str) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(html, encoding="utf-8")
    (dist / "sitemap.xml").write_text(
        "<urlset><url><loc>https://xingwudao.github.io/open-xquant/</loc></url></urlset>",
        encoding="utf-8",
    )
    (dist / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nSitemap: https://xingwudao.github.io/open-xquant/sitemap.xml\n",
        encoding="utf-8",
    )
    return dist


def test_validate_site_accepts_complete_metadata(tmp_path: Path) -> None:
    validate_site(_write_site(tmp_path, VALID_HTML))


def test_validate_site_rejects_missing_canonical(tmp_path: Path) -> None:
    html = VALID_HTML.replace(
        '<link rel="canonical" href="https://xingwudao.github.io/open-xquant/">',
        "",
    )
    with pytest.raises(SiteValidationError, match="canonical"):
        validate_site(_write_site(tmp_path, html))


def test_validate_site_rejects_duplicate_titles(tmp_path: Path) -> None:
    dist = _write_site(tmp_path, VALID_HTML)
    nested = dist / "guide"
    nested.mkdir()
    (nested / "duplicate.html").write_text(VALID_HTML, encoding="utf-8")
    with pytest.raises(SiteValidationError, match="duplicate title"):
        validate_site(dist)
```

- [ ] **Step 2: Run validator tests and confirm import failure**

Run:

```bash
uv run pytest tests/docs/test_validate_site.py -v
```

Expected: FAIL because `validate_site.py` does not exist.

- [ ] **Step 3: Implement the production HTML validator**

Use `html.parser.HTMLParser` from the Python standard library. Export:

```python
class SiteValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PageSignals:
    path: Path
    title: str
    description: str
    canonical: str
    h1_count: int
    og_title: str
    og_description: str
    og_url: str
    og_image: str
    json_ld_count: int
```

Implement these exact callable interfaces:

```text
inspect_page(path: Path) -> PageSignals
validate_site(dist: Path) -> tuple[PageSignals, ...]
main(argv: Sequence[str] | None = None) -> int
```

Validation must fail when:

- no HTML files exist;
- title, description, canonical, Open Graph fields or JSON-LD are missing;
- title, description or canonical duplicates another indexable page;
- H1 count is not exactly one;
- canonical or sitemap URL leaves the production `/open-xquant/` prefix;
- sitemap entries do not equal validated HTML canonical URLs;
- robots does not declare the production sitemap;
- a generated page has an empty body.

Exclude only `404.html` from unique metadata and sitemap equality checks.

- [ ] **Step 4: Run unit and production validation**

Run:

```bash
uv run pytest tests/docs/test_validate_site.py -v
npm --prefix website run docs:build
npm --prefix website run docs:validate
```

Expected: PASS.

- [ ] **Step 5: Add build-output ignores**

Append these exact lines to `.gitignore`:

```text
website/node_modules/
website/.vitepress/cache/
website/.vitepress/dist/
```

Do not ignore generated Markdown, `package-lock.json`, source images or site
configuration.

- [ ] **Step 6: Add the Pages workflow**

Create `.github/workflows/docs-pages.yml` with:

```yaml
name: Docs and GitHub Pages

on:
  pull_request:
    paths:
      - "README.md"
      - "pyproject.toml"
      - "agent/skills/**"
      - "website/**"
      - "tests/docs/**"
      - ".github/workflows/docs-pages.yml"
  push:
    branches:
      - main
    paths:
      - "README.md"
      - "pyproject.toml"
      - "agent/skills/**"
      - "website/**"
      - "tests/docs/**"
      - ".github/workflows/docs-pages.yml"
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0
        with:
          enable-cache: true
      - uses: actions/setup-node@v6
        with:
          node-version-file: website/.node-version
          cache: npm
          cache-dependency-path: website/package-lock.json
      - run: uv sync --extra dev
      - run: npm ci --prefix website
      - run: uv run pytest tests/docs tests/contracts/test_brand_naming.py -v
      - run: npm --prefix website run docs:check
      - run: npm --prefix website run docs:build
      - run: npm --prefix website run docs:validate
      - if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: actions/configure-pages@v5
      - if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: actions/upload-pages-artifact@v4
        with:
          path: website/.vitepress/dist

  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 7: Validate the workflow and run the full local gate**

Run:

```bash
uv run python -c "from pathlib import Path; import yaml; yaml.safe_load(Path('.github/workflows/docs-pages.yml').read_text())"
uv run pytest tests/docs tests/contracts/test_brand_naming.py -v
uv run ruff check website/scripts tests/docs
uv run mypy website/scripts
npm ci --prefix website
npm --prefix website run docs:check
npm --prefix website run docs:build
npm --prefix website run docs:validate
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 8: Commit CI and validation**

```bash
git add .gitignore .github/workflows/docs-pages.yml website/scripts/validate_site.py tests/docs/test_validate_site.py
git commit -m "ci: validate and deploy seo documentation"
```

---

### Task 8: Publish Pages And Update External Repository Signals

**Files:**
- Create: `docs/seo/release-runbook.md`
- Modify: `tests/docs/test_seo_repository.py`
- External: GitHub Pages source, description, homepage, topics, Search Console

**Interfaces:**
- Consumes: a merged `main` commit whose Pages workflow passed.
- Produces: live production pages, verified GitHub metadata and a repeatable indexing handoff.

- [ ] **Step 1: Add a failing release-runbook contract**

Extend `tests/docs/test_seo_repository.py`:

```python
def test_release_runbook_contains_exact_external_contract() -> None:
    text = (ROOT / "docs/seo/release-runbook.md").read_text(encoding="utf-8")
    assert "https://xingwudao.github.io/open-xquant/" in text
    assert "AI 量化研究框架：AI Agent 驱动策略回测、因子研究、稳健性检验、审计报告与实盘交易 | Agentic Quant Research Kernel" in text
    assert "sitemap.xml" in text
    assert "Google Search Console" in text
    assert "gh repo view xingwudao/open-xquant" in text
```

- [ ] **Step 2: Run the contract and confirm missing-runbook failure**

Run:

```bash
uv run pytest tests/docs/test_seo_repository.py -v
```

Expected: FAIL because the runbook does not exist.

- [ ] **Step 3: Write the exact release runbook**

Create `docs/seo/release-runbook.md` with these ordered gates:

```text
1. Confirm the SEO branch is merged to main.
2. Enable GitHub Pages with GitHub Actions as the source.
3. Watch the Docs and GitHub Pages workflow until both jobs pass.
4. Smoke-test homepage, three core pages, robots.txt and sitemap.xml.
5. Update GitHub description, homepage and ten approved topics.
6. Read GitHub metadata back and compare exact values.
7. Verify Google Search Console ownership and submit sitemap.xml.
8. Record production date and begin weekly observation.
```

Include these exact commands:

```bash
seo_run_id="$(gh run list --workflow docs-pages.yml --branch main --limit 1 --json databaseId --jq '.[0].databaseId')"
gh run watch "$seo_run_id" --exit-status

curl -fsS https://xingwudao.github.io/open-xquant/
curl -fsS https://xingwudao.github.io/open-xquant/guide/ai-quant-framework
curl -fsS https://xingwudao.github.io/open-xquant/workflows/strategy-backtest
curl -fsS https://xingwudao.github.io/open-xquant/skills/
curl -fsS https://xingwudao.github.io/open-xquant/robots.txt
curl -fsS https://xingwudao.github.io/open-xquant/sitemap.xml

gh repo edit xingwudao/open-xquant \
  --description "AI 量化研究框架：AI Agent 驱动策略回测、因子研究、稳健性检验、审计报告与实盘交易 | Agentic Quant Research Kernel" \
  --homepage "https://xingwudao.github.io/open-xquant/" \
  --add-topic ai-quant \
  --add-topic quantitative-finance \
  --add-topic quant-research \
  --add-topic ai-agents \
  --add-topic agentic-ai \
  --add-topic backtesting \
  --add-topic factor-research \
  --add-topic algorithmic-trading \
  --add-topic trading-strategy \
  --add-topic python

gh repo view xingwudao/open-xquant \
  --json description,homepageUrl,repositoryTopics,url
```

Document that the repository owner must enable Pages source and complete
Search Console ownership in authenticated web interfaces when CLI access is
not available. Do not update homepage before all production smoke tests pass.

- [ ] **Step 4: Run repository and naming tests**

Run:

```bash
uv run pytest tests/docs/test_seo_repository.py tests/contracts/test_brand_naming.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the release runbook**

```bash
git add docs/seo/release-runbook.md tests/docs/test_seo_repository.py
git commit -m "docs: add seo release and indexing runbook"
```

- [ ] **Step 6: Complete the post-merge production gates**

After merge, execute the runbook in order. Capture:

```text
merged main commit SHA
successful Pages workflow run URL
production deployment URL
GitHub metadata readback
Search Console verification status
sitemap submission date
```

If Pages, smoke checks or metadata readback fails, stop at that gate and leave
the previous production metadata unchanged.

---

## Final Verification

Run the complete local gate from repository root:

```bash
uv run pytest tests/docs tests/contracts/test_brand_naming.py -v
uv run ruff check website/scripts tests/docs
uv run mypy website/scripts
npm ci --prefix website
npm --prefix website run docs:check
npm --prefix website run docs:build
npm --prefix website run docs:validate
git diff --check
git status --short
```

Expected local result:

```text
all pytest tests pass
ruff reports no errors
mypy reports success
npm ci exits 0
generated content has no drift
VitePress production build succeeds
built HTML validation succeeds
git diff has no whitespace errors
working tree has no uncommitted implementation changes
```

Post-merge production verification:

```bash
curl -fsS https://xingwudao.github.io/open-xquant/
curl -fsS https://xingwudao.github.io/open-xquant/sitemap.xml
gh repo view xingwudao/open-xquant --json description,homepageUrl,repositoryTopics,url
```

Do not claim production SEO deployment complete until the live URL, sitemap
and GitHub metadata readback all match the approved contract. Search ranking
improvement remains an observed outcome over the following 6 to 8 weeks.
