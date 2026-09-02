from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import click

from website.scripts.seo_docs import (
    CommandRecord,
    ContentError,
    check_outputs,
    collect_leaf_commands,
    load_skill_extensions,
    load_skill_sources,
    merge_skill_records,
    render_outputs,
    write_outputs,
)


def build_outputs(repo_root: Path) -> dict[Path, str]:
    sources = load_skill_sources(repo_root)
    extensions = load_skill_extensions(repo_root / "website/data/skills.zh.yaml")
    skills = merge_skill_records(sources, extensions)
    command_rows = collect_leaf_commands(_load_oxq_main())
    commands = tuple(
        CommandRecord(path=path, summary=summary) for path, summary in command_rows
    )
    return render_outputs(skills, commands)


def _load_oxq_main() -> click.Command:
    module = importlib.import_module("oxq.cli.main")
    return cast(click.Command, module.main)


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


if __name__ == "__main__":
    try:
        raise SystemExit(main_cli())
    except ContentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
