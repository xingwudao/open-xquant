from __future__ import annotations

from pathlib import Path

import click
import pytest
from website.scripts.generate import build_outputs
from website.scripts.seo_docs import (
    ContentError,
    check_outputs,
    collect_leaf_commands,
    load_skill_sources,
    load_skill_extensions,
    merge_skill_records,
    parse_skill_source,
    render_outputs,
)

ROOT = Path(__file__).parents[2]

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


def test_collect_leaf_commands_skips_hidden_commands_and_groups() -> None:
    @click.group()
    def root() -> None:
        pass

    @root.command(help="Public command.")
    def public() -> None:
        pass

    @root.command(help="Hidden command.", hidden=True)
    def hidden_leaf() -> None:
        pass

    @root.group(hidden=True)
    def hidden_group() -> None:
        pass

    @hidden_group.command(help="Hidden nested command.")
    def nested() -> None:
        pass

    assert collect_leaf_commands(root) == (("public", "Public command."),)


def test_check_outputs_reports_generated_drift(tmp_path: Path) -> None:
    target = tmp_path / "website" / "skills" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_text("stale\n", encoding="utf-8")
    with pytest.raises(ContentError, match="generated content drift"):
        check_outputs(tmp_path, {Path("website/skills/index.md"): "current\n"})


def test_check_outputs_compares_expected_text_bytes_exactly(tmp_path: Path) -> None:
    target = tmp_path / "website" / "skills" / "index.md"
    target.parent.mkdir(parents=True)
    target.write_text("current\n", encoding="utf-8")
    with pytest.raises(ContentError, match="generated content drift"):
        check_outputs(tmp_path, {Path("website/skills/index.md"): "current"})


def test_load_skill_extensions_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    path = tmp_path / "skills.zh.yaml"
    extension = """\
  title_zh: 示例 Skill
  summary_zh: 示例摘要。
  group: 示例分组
  use_cases_zh:
    - 示例场景。
  inputs_zh:
    - 示例输入。
  outputs_zh:
    - 示例输出。
  constraints_zh:
    - 示例约束。
  related_workflows:
    - /guide/example
"""
    path.write_text(
        f"audit-sample:\n{extension}audit-sample:\n{extension}",
        encoding="utf-8",
    )
    with pytest.raises(ContentError, match="duplicate Chinese extension"):
        load_skill_extensions(path)


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


def test_skills_index_uses_required_group_order() -> None:
    outputs = build_outputs(ROOT)
    text = outputs[Path("website/skills/index.md")]
    headings = [
        line.removeprefix("## ")
        for line in text.splitlines()
        if line.startswith("## ")
    ]
    assert headings == [
        "研究治理",
        "策略与审计",
        "数据与因子",
        "组件开发",
        "执行与报告",
    ]


def test_render_outputs_returns_repo_relative_paths() -> None:
    assert all(not path.is_absolute() for path in render_outputs((), ()).keys())
