from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import oxq.report.qa as qa_module
from oxq.report.assets import ReportPublicationError, publish_report_artifacts
from oxq.report.qa import run_report_qa
from oxq.selection_lock import final_selection_lock_path, hold_final_selection_lock

_CRASH_EXIT_CODE = 87
_JOURNAL_NAME = ".oxq-report-transaction.json"


def test_first_publication_creates_missing_report_directory_from_existing_ancestor(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    config_dir = workspace / ".open-xquant"
    config_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        "workflow:\n  layout: version_governed\npaths:\n  versions_dir: versions\n",
        encoding="utf-8",
    )
    report_dir = workspace / "versions/v001/10_reports/run-24"

    publish_report_artifacts(report_dir, {"research_report.md": b"first report\n"})

    assert (report_dir / "research_report.md").read_bytes() == b"first report\n"


def test_first_publication_honors_explicit_lock_subject(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    config_dir = workspace / ".open-xquant"
    config_dir.mkdir(parents=True)
    (config_dir / "workspace.yaml").write_text(
        "workflow:\n  layout: version_governed\npaths:\n  versions_dir: versions\n",
        encoding="utf-8",
    )
    source_run = workspace / "versions/v001/09_backtests/run-24"
    source_run.mkdir(parents=True)
    report_dir = workspace / "versions/v001/10_reports/run-24"
    attempted = threading.Event()
    completed = threading.Event()
    failures: list[BaseException] = []

    def publish() -> None:
        attempted.set()
        try:
            publish_report_artifacts(
                report_dir,
                {"research_report.md": b"first report\n"},
                lock_subject=source_run,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            completed.set()

    with hold_final_selection_lock(final_selection_lock_path(source_run)):
        worker = threading.Thread(target=publish, name="round-24-first-publisher")
        worker.start()
        assert attempted.wait(timeout=5)
        assert not completed.wait(timeout=0.1)
        assert not report_dir.exists()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []
    assert (report_dir / "research_report.md").read_bytes() == b"first report\n"


@pytest.mark.parametrize(
    ("boundary", "missing_relative"),
    [
        ("research_report.md.replace", "research_report.md"),
        ("report_assets/manifest.json.replace", "report_assets/manifest.json"),
    ],
)
def test_publication_recovers_crash_before_building_next_batch(
    tmp_path,
    boundary: str,
    missing_relative: str,
) -> None:
    report_dir = _write_report_pair(tmp_path)
    manifest_path = report_dir / "report_assets/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    old_manifest = b'{"schema_version": 1, "assets": []}\n'
    manifest_path.write_bytes(old_manifest)
    unrelated = report_dir / "researcher-notes.txt"
    unrelated.write_bytes(b"keep me\n")

    _crash_during_replacement(report_dir, boundary=boundary)

    assert not (report_dir / missing_relative).exists()
    observed_baseline: list[tuple[bytes, bytes]] = []

    def build_publication() -> dict[str, bytes]:
        observed_baseline.append(
            (
                (report_dir / "research_report.md").read_bytes(),
                manifest_path.read_bytes(),
            )
        )
        return {"research_report.html": b"next html\n"}

    publish_report_artifacts(report_dir, build_publication)

    assert observed_baseline == [(b"old markdown\n", old_manifest)]
    assert (report_dir / "research_report.md").read_bytes() == b"old markdown\n"
    assert manifest_path.read_bytes() == old_manifest
    assert (report_dir / "research_report.html").read_bytes() == b"next html\n"
    assert unrelated.read_bytes() == b"keep me\n"
    assert not (report_dir / _JOURNAL_NAME).exists()


def test_report_qa_recovers_crash_idempotently_before_inspection(monkeypatch, tmp_path) -> None:
    report_dir = _write_report_pair(tmp_path)
    manifest_path = report_dir / "report_assets/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    old_manifest = b'{"schema_version": 1, "assets": []}\n'
    manifest_path.write_bytes(old_manifest)

    _crash_during_replacement(report_dir)

    monkeypatch.setattr(
        qa_module,
        "_resolve_report_qa_paths",
        lambda path, *, source_run_dir=None: (report_dir, report_dir, False),
    )

    def inspect_recovered_snapshot(*args, **kwargs):
        return (
            (report_dir / "research_report.md").read_bytes(),
            manifest_path.read_bytes(),
        )

    monkeypatch.setattr(qa_module, "_run_report_qa_reads", inspect_recovered_snapshot)

    expected = (b"old markdown\n", old_manifest)
    assert run_report_qa(report_dir) == expected
    assert run_report_qa(report_dir) == expected
    assert not (report_dir / _JOURNAL_NAME).exists()


def test_publication_rejects_unsafe_journal_target_without_touching_other_files(tmp_path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside\n")
    unrelated = report_dir / "notes.txt"
    unrelated.write_bytes(b"notes\n")
    journal_path = report_dir / _JOURNAL_NAME
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": "0" * 32,
                "recovery": "rollback",
                "report_root": str(report_dir.resolve()),
                "targets": [
                    {
                        "path": "../outside.txt",
                        "old": None,
                        "new": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReportPublicationError, match="journal.*unsafe|unsafe.*journal"):
        publish_report_artifacts(report_dir, {"research_report.md": b"new report\n"})

    assert outside.read_bytes() == b"outside\n"
    assert unrelated.read_bytes() == b"notes\n"
    assert journal_path.is_file()
    assert not (report_dir / "research_report.md").exists()


def test_publication_rejects_windows_absolute_target_before_starting_transaction(tmp_path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()

    with pytest.raises(ReportPublicationError, match="safe relative path"):
        publish_report_artifacts(report_dir, {r"C:\outside.txt": b"outside\n"})

    assert list(report_dir.iterdir()) == []


def test_standalone_qa_blocks_publication_and_reads_one_snapshot(monkeypatch, tmp_path) -> None:
    report_dir = _write_report_pair(tmp_path)
    qa_paused = threading.Event()
    allow_qa = threading.Event()
    publisher_attempted = threading.Event()
    publisher_completed = threading.Event()
    qa_results: list[tuple[bytes, bytes]] = []
    qa_failures: list[BaseException] = []
    publisher_failures: list[BaseException] = []

    def inspect_snapshot(*args, **kwargs):
        markdown = (report_dir / "research_report.md").read_bytes()
        qa_paused.set()
        assert allow_qa.wait(timeout=5)
        html = (report_dir / "research_report.html").read_bytes()
        return markdown, html

    def inspect_with_qa() -> None:
        try:
            qa_results.append(run_report_qa(report_dir))
        except BaseException as exc:
            qa_failures.append(exc)

    def publish_replacement() -> None:
        publisher_attempted.set()
        try:
            publish_report_artifacts(
                report_dir,
                {
                    "research_report.md": b"new markdown\n",
                    "research_report.html": b"new html\n",
                },
            )
        except BaseException as exc:
            publisher_failures.append(exc)
        finally:
            publisher_completed.set()

    monkeypatch.setattr(qa_module, "_run_report_qa_reads", inspect_snapshot)
    qa_worker = threading.Thread(target=inspect_with_qa, name="round-24-standalone-qa")
    publisher = threading.Thread(target=publish_replacement, name="round-24-report-publisher")
    qa_worker.start()
    try:
        assert qa_paused.wait(timeout=5)
        publisher.start()
        assert publisher_attempted.wait(timeout=5)
        assert not publisher_completed.wait(timeout=0.1)
    finally:
        allow_qa.set()
    qa_worker.join(timeout=5)
    publisher.join(timeout=5)

    assert not qa_worker.is_alive()
    assert not publisher.is_alive()
    assert qa_failures == []
    assert publisher_failures == []
    assert qa_results == [(b"old markdown\n", b"old html\n")]
    assert (report_dir / "research_report.md").read_bytes() == b"new markdown\n"
    assert (report_dir / "research_report.html").read_bytes() == b"new html\n"


def _write_report_pair(tmp_path: Path) -> Path:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "research_report.md").write_bytes(b"old markdown\n")
    (report_dir / "research_report.html").write_bytes(b"old html\n")
    return report_dir


def _crash_during_replacement(
    report_dir: Path,
    *,
    boundary: str = "research_report.md.replace",
) -> None:
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        import oxq.report.assets as assets_module

        def crash_at_replacement(label: str) -> None:
            if label == sys.argv[2]:
                os._exit(87)

        assets_module._report_publication_boundary = crash_at_replacement
        assets_module.publish_report_artifacts(
            Path(sys.argv[1]),
            {
                "research_report.md": b"new markdown\\n",
                "report_assets/manifest.json": b'{"schema_version": 1, "assets": [{"id": "new"}]}\\n',
            },
        )
        """
    )
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source_root), environment.get("PYTHONPATH")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(report_dir), boundary],
        check=False,
        env=environment,
    )
    assert completed.returncode == _CRASH_EXIT_CODE
