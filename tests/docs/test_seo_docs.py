from __future__ import annotations

from pathlib import Path

import click
import pytest
from website.scripts.seo_docs import (
    ContentError,
    check_outputs,
    collect_leaf_commands,
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
