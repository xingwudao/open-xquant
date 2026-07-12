from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from oxq.cli import research


def _directory_symlink_or_skip(alias: Path, target: Path) -> None:
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


def test_research_init_lock_is_shared_by_filesystem_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "workspace-alias"
    _directory_symlink_or_skip(alias, workspace)

    assert research._workspace_init_lock_path(alias) == research._workspace_init_lock_path(workspace)


def test_prepared_recovery_preserves_external_destination_and_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research.initialize_workspace(workspace, name="external-sentinel")
    destination = workspace / "versions/v001/phase_state.json"
    destination.unlink()
    journal_path = workspace / ".open-xquant/governance-transaction.json"
    original_write_journal = research._write_governance_transaction_journal

    def crash_after_prepared(root: Path, journal: dict[str, object]) -> None:
        original_write_journal(root, journal)
        if journal.get("state") == "prepared":
            destination.write_text("external sentinel\n", encoding="utf-8")
            raise RuntimeError("simulated crash after prepared journal")

    monkeypatch.setattr(
        research,
        "_write_governance_transaction_journal",
        crash_after_prepared,
    )

    with pytest.raises(Exception, match="unrecognized.*journal preserved"):
        research._write_governance_files_atomically(
            workspace,
            {destination: "transaction replacement\n"},
        )

    assert destination.read_text(encoding="utf-8") == "external sentinel\n"
    assert journal_path.is_file()


def test_prepared_recovery_uses_persisted_replacement_evidence_then_retries(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    research.initialize_workspace(workspace, name="crash-retry")
    destination = workspace / "versions/v001/phase_state.json"
    destination.unlink()
    replacement = "transaction replacement\n"
    code = """
import os
import pathlib
import sys

from oxq.cli import research

workspace = pathlib.Path(sys.argv[1])
destination = workspace / "versions/v001/phase_state.json"
replacement = sys.argv[2]
original_replace = research.os.replace

def crash_after_install(source, target, *args, **kwargs):
    original_replace(source, target, *args, **kwargs)
    if pathlib.Path(source).name.startswith(f".{destination.name}.stage-") and pathlib.Path(target).name == destination.name:
        os._exit(73)

research.os.replace = crash_after_install
research._write_governance_files_atomically(workspace, {destination: replacement})
"""

    crashed = subprocess.run(
        [sys.executable, "-c", code, str(workspace), replacement],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=15,
        check=False,
    )

    assert crashed.returncode == 73, crashed.stdout + crashed.stderr
    journal_path = workspace / ".open-xquant/governance-transaction.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    entry = journal["entries"][0]
    assert entry["progress"] == "staged"
    assert entry["replacement_identity"] == research._governance_identity_payload(destination.stat())
    assert entry["replacement_sha256"] == hashlib.sha256(replacement.encode("utf-8")).hexdigest()

    research._recover_governance_transaction(workspace)

    assert not destination.exists()
    assert not journal_path.exists()
    research._write_governance_files_atomically(
        workspace,
        {destination: replacement},
    )
    assert destination.read_text(encoding="utf-8") == replacement
    assert not journal_path.exists()
