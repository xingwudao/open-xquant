from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from oxq.cli import main as main_module
from oxq.cli.main import main
from oxq.spec.schema import StrategySpec

_CRASH_DURING_BACKUP_CLEANUP = r"""
import json
import os
from pathlib import Path

from oxq.cli import main as main_module

original_unlink = main_module.os.unlink


def exit_after_marker_unlink(path, *args, **kwargs):
    result = original_unlink(path, *args, **kwargs)
    if Path(path).name == main_module._COMPILE_PREVIEW_MARKER_NAME:
        os._exit(86)
    return result


main_module.os.unlink = exit_after_marker_unlink
main_module.main.main(
    args=json.loads(os.environ["OXQ_PREVIEW_ARGS"]),
    prog_name="oxq",
    standalone_mode=False,
)
"""


_CRASH_AFTER_BACKUP_RENAME = r"""
import json
import os
from pathlib import Path

from oxq.cli import main as main_module

target = Path(os.environ["OXQ_PREVIEW_TARGET"])
original_replace = main_module.os.replace


def exit_after_backup_rename(source, destination, *args, **kwargs):
    result = original_replace(source, destination, *args, **kwargs)
    source_path = Path(source)
    destination_path = Path(destination)
    if (
        source_path.name == target.name
        and destination_path.name.startswith(f".{target.name}.oxq-preview-old-")
    ):
        os._exit(87)
    return result


main_module.os.replace = exit_after_backup_rename
main_module.main.main(
    args=json.loads(os.environ["OXQ_PREVIEW_ARGS"]),
    prog_name="oxq",
    standalone_mode=False,
)
"""


def _write_spec(path: Path, hypothesis: str) -> None:
    spec = StrategySpec.template(
        strategy_id="round_25_compile_preview",
        hypothesis=hypothesis,
    )
    path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")


def _compile(spec_path: Path, out_dir: Path):
    return CliRunner().invoke(
        main,
        ["strategy", "compile", str(spec_path), "--out", str(out_dir)],
    )


def _run_crashing_compile(
    script: str,
    *,
    spec_path: Path,
    out_dir: Path,
) -> subprocess.CompletedProcess[str]:
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
        [sys.executable, "-c", script],
        cwd=spec_path.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _transaction_path(out_dir: Path) -> Path:
    return out_dir.with_name(f".{out_dir.name}.oxq-preview-transaction.json")


def _transaction_sibling(transaction_path: Path, field: str) -> Path:
    payload = json.loads(transaction_path.read_text(encoding="utf-8"))
    recorded = Path(payload[field])
    return recorded if recorded.is_absolute() else transaction_path.parent / recorded


def _prepare_cleanup_crash(tmp_path: Path) -> tuple[Path, Path, Path]:
    spec_path = tmp_path / "strategy_spec.yaml"
    out_dir = tmp_path / "compile_preview"
    _write_spec(spec_path, "initial cleanup generation")
    initial = _compile(spec_path, out_dir)
    assert initial.exit_code == 0, initial.output

    _write_spec(spec_path, "replacement cleanup generation")
    crashed = _run_crashing_compile(
        _CRASH_DURING_BACKUP_CLEANUP,
        spec_path=spec_path,
        out_dir=out_dir,
    )
    assert crashed.returncode == 86, (crashed.stdout, crashed.stderr)

    transaction = _transaction_path(out_dir)
    assert transaction.is_file()
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    assert payload["phase"] == "backup_cleanup"
    assert payload["backup_cleanup_kind"] == "managed"
    assert isinstance(payload["backup_cleanup_identity"], str)
    assert payload["backup_cleanup_identity"]
    cleanup = _transaction_sibling(transaction, "backup_quarantine")
    assert cleanup.is_dir()
    assert not (cleanup / main_module._COMPILE_PREVIEW_MARKER_NAME).exists()
    return spec_path, out_dir, transaction


def test_compile_preview_retries_after_exit_during_recursive_backup_cleanup(
    tmp_path: Path,
) -> None:
    spec_path, out_dir, transaction = _prepare_cleanup_crash(tmp_path)

    retried = _compile(spec_path, out_dir)

    assert retried.exit_code == 0, retried.output
    assert not transaction.exists()
    assert not list(tmp_path.glob(".compile_preview.oxq-preview-old-*"))
    assert not list(tmp_path.glob(".compile_preview.oxq-preview-cleanup-*"))
    assert not list(tmp_path.glob(".compile_preview.stage-*"))


def test_compile_preview_cleanup_preserves_replacement_at_recorded_backup_name(
    tmp_path: Path,
) -> None:
    spec_path, out_dir, transaction = _prepare_cleanup_crash(tmp_path)
    backup = _transaction_sibling(transaction, "backup")
    cleanup = _transaction_sibling(transaction, "backup_quarantine")
    assert not backup.exists()

    backup.mkdir()
    (backup / main_module._COMPILE_PREVIEW_MARKER_NAME).write_text(
        json.dumps(main_module._COMPILE_PREVIEW_MARKER, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sentinel = backup / "external-owner.txt"
    sentinel.write_text("preserve replacement\n", encoding="utf-8")

    retried = _compile(spec_path, out_dir)

    assert retried.exit_code == 1
    assert "identity" in retried.output.lower()
    assert sentinel.read_text(encoding="utf-8") == "preserve replacement\n"
    assert cleanup.is_dir()
    assert transaction.is_file()


def test_compile_preview_crash_retries_through_alternate_case_alias(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "CaseSensitiveSpelling"
    parent.mkdir()
    spec_path = parent / "strategy_spec.yaml"
    out_dir = parent / "CompilePreview"
    _write_spec(spec_path, "initial case spelling")
    initial = _compile(spec_path, out_dir)
    assert initial.exit_code == 0, initial.output

    _write_spec(spec_path, "replacement case spelling")
    crashed = _run_crashing_compile(
        _CRASH_AFTER_BACKUP_RENAME,
        spec_path=spec_path,
        out_dir=out_dir,
    )
    assert crashed.returncode == 87, (crashed.stdout, crashed.stderr)

    transaction = _transaction_path(out_dir)
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    for field in ("target", "staging", "backup"):
        recorded = Path(payload[field])
        assert not recorded.is_absolute()
        assert recorded == Path(recorded.name)

    alternate_parent = parent.with_name(parent.name.swapcase())
    alternate_out = alternate_parent / out_dir.name.swapcase()
    alternate_transaction = _transaction_path(alternate_out)
    if not alternate_transaction.exists() or not alternate_transaction.samefile(transaction):
        pytest.skip("test requires a case-insensitive filesystem")

    retried = _compile(spec_path, alternate_out)

    assert retried.exit_code == 0, retried.output
    assert alternate_out.is_dir()
    assert not transaction.exists()
    assert not list(parent.glob(".CompilePreview.oxq-preview-old-*"))
