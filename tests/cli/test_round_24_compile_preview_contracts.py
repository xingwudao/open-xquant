from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from click.testing import CliRunner

from oxq.cli.main import main
from oxq.spec.schema import StrategySpec

_CRASH_AFTER_PREVIEW_BACKUP = r"""
import json
import os
from pathlib import Path

from oxq.cli import main as main_module

target = Path(os.environ["OXQ_PREVIEW_TARGET"])
original_replace = main_module.os.replace


def exit_after_backup(source, destination, *args, **kwargs):
    result = original_replace(source, destination, *args, **kwargs)
    source_path = Path(source)
    destination_path = Path(destination)
    if (
        source_path.name == target.name
        and destination_path.name.startswith(f".{target.name}.oxq-preview-old-")
    ):
        os._exit(73)
    return result


main_module.os.replace = exit_after_backup
main_module.main.main(
    args=json.loads(os.environ["OXQ_PREVIEW_ARGS"]),
    prog_name="oxq",
    standalone_mode=False,
)
"""


def _write_spec(path: Path, hypothesis: str) -> None:
    spec = StrategySpec.template(
        strategy_id="round_24_compile_preview",
        hypothesis=hypothesis,
    )
    path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")


def _compile(spec_path: Path, out_dir: Path):
    return CliRunner().invoke(
        main,
        ["strategy", "compile", str(spec_path), "--out", str(out_dir)],
    )


def _crash_after_backup(spec_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env.update(
        {
            "OXQ_PREVIEW_ARGS": json.dumps(
                ["strategy", "compile", str(spec_path), "--out", str(out_dir)]
            ),
            "OXQ_PREVIEW_TARGET": str(out_dir),
            "PYTHONPATH": os.pathsep.join(
                [str(repo / "src"), existing_pythonpath]
                if existing_pythonpath
                else [str(repo / "src")]
            ),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", _CRASH_AFTER_PREVIEW_BACKUP],
        cwd=spec_path.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _preview_transaction_path(out_dir: Path) -> Path:
    return out_dir.with_name(f".{out_dir.name}.oxq-preview-transaction.json")


def test_compile_preview_recovers_exit_immediately_after_backup_rename(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    out_dir = tmp_path / "compile_preview"
    _write_spec(spec_path, "initial durable preview")
    initial = _compile(spec_path, out_dir)
    assert initial.exit_code == 0, initial.output
    initial_hash = (out_dir / "spec_hash.txt").read_text(encoding="utf-8")

    _write_spec(spec_path, "replacement durable preview")
    crashed = _crash_after_backup(spec_path, out_dir)

    assert crashed.returncode == 73, (crashed.stdout, crashed.stderr)
    assert _preview_transaction_path(out_dir).is_file()
    assert not out_dir.exists()
    assert len(list(tmp_path.glob(".compile_preview.oxq-preview-old-*"))) == 1

    recovered = _compile(spec_path, out_dir)

    assert recovered.exit_code == 0, recovered.output
    assert (out_dir / "spec_hash.txt").read_text(encoding="utf-8") != initial_hash
    assert not _preview_transaction_path(out_dir).exists()
    assert not list(tmp_path.glob(".compile_preview.oxq-preview-old-*"))
    assert not list(tmp_path.glob(".compile_preview.stage-*"))


def test_compile_preview_recovery_preserves_unrecognized_replacement_target(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    out_dir = tmp_path / "compile_preview"
    _write_spec(spec_path, "initial owner-protected preview")
    initial = _compile(spec_path, out_dir)
    assert initial.exit_code == 0, initial.output

    _write_spec(spec_path, "replacement owner-protected preview")
    crashed = _crash_after_backup(spec_path, out_dir)
    assert crashed.returncode == 73, (crashed.stdout, crashed.stderr)
    transaction = _preview_transaction_path(out_dir)
    assert transaction.is_file()
    backup = next(iter(tmp_path.glob(".compile_preview.oxq-preview-old-*")))

    out_dir.mkdir()
    sentinel = out_dir / "user-owned.txt"
    sentinel.write_text("do not delete\n", encoding="utf-8")
    recovery = _compile(spec_path, out_dir)

    assert recovery.exit_code == 1
    assert sentinel.read_text(encoding="utf-8") == "do not delete\n"
    assert transaction.is_file()
    assert backup.is_dir()
