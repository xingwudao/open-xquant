from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from oxq.cli import main as main_module
from oxq.cli.main import main
from oxq.spec.schema import StrategySpec

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="round 26 exercises POSIX directory-descriptor publication",
)


_CRASH_AFTER_STAGING_INSTALL = r"""
import json
import os
from pathlib import Path

from oxq.cli import main as main_module

target = Path(os.environ["OXQ_PREVIEW_TARGET"])
original_replace = main_module.os.replace


def exit_after_staging_install(source, destination, *args, **kwargs):
    result = original_replace(source, destination, *args, **kwargs)
    if (
        Path(source).name.startswith(f".{target.name}.stage-")
        and Path(destination).name == target.name
    ):
        os._exit(88)
    return result


main_module.os.replace = exit_after_staging_install
main_module.main.main(
    args=json.loads(os.environ["OXQ_PREVIEW_ARGS"]),
    prog_name="oxq",
    standalone_mode=False,
)
"""


def _write_spec(path: Path, hypothesis: str) -> None:
    spec = StrategySpec.template(
        strategy_id="round_26_compile_preview",
        hypothesis=hypothesis,
    )
    path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")


def _compile(spec_path: Path, out_dir: Path):
    return CliRunner().invoke(
        main,
        ["strategy", "compile", str(spec_path), "--out", str(out_dir)],
    )


def _write_managed_marker(directory: Path) -> None:
    directory.mkdir(exist_ok=True)
    (directory / main_module._COMPILE_PREVIEW_MARKER_NAME).write_text(
        json.dumps(main_module._COMPILE_PREVIEW_MARKER, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _transaction_path(out_dir: Path) -> Path:
    return out_dir.with_name(f".{out_dir.name}.oxq-preview-transaction.json")


def _run_crashing_compile(
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
        [sys.executable, "-c", _CRASH_AFTER_STAGING_INSTALL],
        cwd=spec_path.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_compile_preview_parent_swap_before_backup_rename_is_not_redirected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    parent = tmp_path / "preview-parent"
    parent.mkdir()
    out_dir = parent / "compile_preview"
    _write_spec(spec_path, "initial parent-pinned generation")
    initial = _compile(spec_path, out_dir)
    assert initial.exit_code == 0, initial.output
    initial_hash = (out_dir / "spec_hash.txt").read_text(encoding="utf-8")

    _write_spec(spec_path, "replacement parent-pinned generation")
    displaced_parent = tmp_path / "displaced-preview-parent"
    original_replace = main_module.os.replace
    swapped = False

    def swap_before_backup(source, destination, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and Path(source).name == out_dir.name
            and Path(destination).name.startswith(f".{out_dir.name}.oxq-preview-old-")
        ):
            swapped = True
            original_replace(parent, displaced_parent)
            parent.mkdir()
            replacement_target = parent / out_dir.name
            replacement_target.mkdir()
            (replacement_target / "external-owner.txt").write_text(
                "preserve replacement parent\n",
                encoding="utf-8",
            )
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(main_module.os, "replace", swap_before_backup)

    result = _compile(spec_path, out_dir)

    assert swapped
    assert result.exit_code == 1
    assert (parent / out_dir.name / "external-owner.txt").read_text(
        encoding="utf-8"
    ) == "preserve replacement parent\n"
    displaced_target = displaced_parent / out_dir.name
    assert (displaced_target / "spec_hash.txt").read_text(encoding="utf-8") == initial_hash
    assert not _transaction_path(displaced_target).exists()
    assert not list(displaced_parent.glob(".compile_preview.oxq-preview-old-*"))
    assert not list(displaced_parent.glob(".compile_preview.stage-*"))


def test_compile_preview_parent_swap_before_staging_rename_is_not_redirected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    parent = tmp_path / "preview-parent"
    parent.mkdir()
    out_dir = parent / "compile_preview"
    _write_spec(spec_path, "initial staging-parent generation")
    initial = _compile(spec_path, out_dir)
    assert initial.exit_code == 0, initial.output
    initial_hash = (out_dir / "spec_hash.txt").read_text(encoding="utf-8")

    _write_spec(spec_path, "replacement staging-parent generation")
    displaced_parent = tmp_path / "displaced-preview-parent"
    original_replace = main_module.os.replace
    swapped = False

    def swap_before_install(source, destination, *args, **kwargs):
        nonlocal swapped
        source_name = Path(source).name
        if (
            not swapped
            and source_name.startswith(f".{out_dir.name}.stage-")
            and Path(destination).name == out_dir.name
        ):
            swapped = True
            original_replace(parent, displaced_parent)
            parent.mkdir()
            (parent / out_dir.name).mkdir()
            replacement_staging = parent / source_name
            _write_managed_marker(replacement_staging)
            (replacement_staging / "external-owner.txt").write_text(
                "preserve replacement staging\n",
                encoding="utf-8",
            )
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(main_module.os, "replace", swap_before_install)

    result = _compile(spec_path, out_dir)

    assert swapped
    assert result.exit_code == 1
    assert not any((parent / out_dir.name).iterdir())
    assert (parent / next(parent.glob(".compile_preview.stage-*")).name / "external-owner.txt").read_text(
        encoding="utf-8"
    ) == "preserve replacement staging\n"
    displaced_target = displaced_parent / out_dir.name
    assert (displaced_target / "spec_hash.txt").read_text(encoding="utf-8") == initial_hash
    assert not _transaction_path(displaced_target).exists()
    assert not list(displaced_parent.glob(".compile_preview.oxq-preview-old-*"))
    assert not list(displaced_parent.glob(".compile_preview.stage-*"))


def test_compile_preview_cleanup_preserves_replacement_after_identity_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    out_dir = tmp_path / "compile_preview"
    _write_spec(spec_path, "initial cleanup barrier generation")
    initial = _compile(spec_path, out_dir)
    assert initial.exit_code == 0, initial.output

    _write_spec(spec_path, "replacement cleanup barrier generation")
    original_identity = main_module._compile_preview_directory_identity
    identity_calls = 0
    replacement_path: Path | None = None
    displaced_backup = tmp_path / "validated-backup-displaced"

    def validate_then_replace(path: Path) -> str:
        nonlocal identity_calls, replacement_path
        identity = original_identity(path)
        identity_calls += 1
        if identity_calls == 2:
            path.replace(displaced_backup)
            _write_managed_marker(path)
            (path / "external-owner.txt").write_text(
                "preserve post-validation replacement\n",
                encoding="utf-8",
            )
            replacement_path = path
        return identity

    monkeypatch.setattr(
        main_module,
        "_compile_preview_directory_identity",
        validate_then_replace,
    )

    result = _compile(spec_path, out_dir)

    assert identity_calls >= 2
    assert result.exit_code == 1
    assert replacement_path is not None
    assert (replacement_path / "external-owner.txt").read_text(
        encoding="utf-8"
    ) == "preserve post-validation replacement\n"
    assert displaced_backup.is_dir()
    assert _transaction_path(out_dir).is_file()


def test_compile_preview_unsupported_platform_fails_before_output_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "unsupported-parent"
    out_dir = parent / "compile_preview"
    monkeypatch.setattr(main_module.os, "name", "nt")

    with pytest.raises(click.ClickException, match="stable relative directory operations"):
        main_module._publish_compile_preview(out_dir, {"compiled_plan.json": b"{}\n"}, [])

    assert not parent.exists()


def test_compile_preview_legacy_journal_fails_closed_for_ambiguous_installed_target(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    out_dir = tmp_path / "compile_preview"
    _write_spec(spec_path, "legacy ambiguous installed generation")
    crashed = _run_crashing_compile(spec_path, out_dir)
    assert crashed.returncode == 88, (crashed.stdout, crashed.stderr)

    transaction = _transaction_path(out_dir)
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    del payload["staging_identity"]
    transaction.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    recovered = _compile(spec_path, out_dir)

    assert recovered.exit_code == 1
    assert "invalid pending compile preview transaction" not in recovered.output.lower()
    assert out_dir.is_dir()
    assert transaction.is_file()


@pytest.mark.parametrize("had_target", [False, True], ids=["no-target", "had-target"])
def test_compile_preview_recovers_exit_after_staging_install_before_phase_update(
    tmp_path: Path,
    had_target: bool,
) -> None:
    spec_path = tmp_path / "strategy_spec.yaml"
    out_dir = tmp_path / "compile_preview"
    if had_target:
        _write_spec(spec_path, "initial pre-crash generation")
        initial = _compile(spec_path, out_dir)
        assert initial.exit_code == 0, initial.output

    _write_spec(spec_path, "installed pre-phase-update generation")
    crashed = _run_crashing_compile(spec_path, out_dir)

    assert crashed.returncode == 88, (crashed.stdout, crashed.stderr)
    transaction = _transaction_path(out_dir)
    assert transaction.is_file()
    payload = json.loads(transaction.read_text(encoding="utf-8"))
    assert payload["phase"] == ("backup_created" if had_target else "prepared")
    assert isinstance(payload.get("staging_identity"), str)
    assert payload["staging_identity"]
    assert out_dir.is_dir()
    assert not (tmp_path / payload["staging"]).exists()

    recovered = _compile(spec_path, out_dir)

    assert recovered.exit_code == 0, recovered.output
    assert not transaction.exists()
    assert not list(tmp_path.glob(".compile_preview.oxq-preview-old-*"))
    assert not list(tmp_path.glob(".compile_preview.stage-*"))
