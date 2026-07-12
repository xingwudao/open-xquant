from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
import yaml

import oxq.report.assets as assets_module
import oxq.report.qa as qa_module
from oxq.report.assets import (
    ReportPublicationError,
    list_report_assets,
    publish_report_artifacts,
    report_publication_read_transaction,
)
from oxq.report.generator import generate_report
from oxq.report.html import render_html_report, render_markdown_html_report
from oxq.report.qa import run_report_qa
from oxq.run_digests import run_digest_transaction
from oxq.spec.schema import StrategySpec

_CRASH_EXIT_CODE = 87
_JOURNAL_NAME = ".oxq-report-transaction.json"
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR4nGNgYGAAAAAEAAHIiY1AAAAAAElFTkSuQmCC"
)


def test_first_publication_rejects_existing_symlink_ancestor_before_lock_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    _symlink_directory(alias, outside)
    report_dir = alias / "first-report"

    def reject_lock_discovery(path: Path) -> None:
        raise AssertionError(f"selection lock discovery escaped through {path}")

    monkeypatch.setattr(assets_module, "final_selection_lock_path", reject_lock_discovery)

    with pytest.raises(ReportPublicationError, match="symlink|reparse"):
        publish_report_artifacts(report_dir, {"research_report.md": b"outside\n"})

    assert not (outside / "first-report").exists()


def test_publication_lstats_symlink_component_before_parent_segment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    _symlink_directory(alias, outside)
    supplied = alias / ".." / "report"

    def reject_lock_discovery(path: Path) -> None:
        raise AssertionError(f"selection lock discovery ran for unsafe path {path}")

    monkeypatch.setattr(assets_module, "final_selection_lock_path", reject_lock_discovery)

    with pytest.raises(ReportPublicationError, match="symlink|reparse"):
        publish_report_artifacts(supplied, {"research_report.md": b"unsafe\n"})


def test_report_read_rejects_symlink_ancestor_without_recovering_outside_tree(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    report_dir = _write_report_pair(outside)
    _crash_at_publication_boundary(report_dir, "research_report.md.replace")
    journal = report_dir / _JOURNAL_NAME
    assert journal.is_file()
    assert not (report_dir / "research_report.md").exists()

    alias = tmp_path / "alias"
    _symlink_directory(alias, outside)

    with pytest.raises(ReportPublicationError, match="symlink|reparse"):
        with report_publication_read_transaction(alias / report_dir.name):
            pass

    assert journal.is_file()
    assert not (report_dir / "research_report.md").exists()


def test_report_write_rejects_symlink_ancestor_without_recovering_outside_tree(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    report_dir = _write_report_pair(outside)
    _crash_at_publication_boundary(report_dir, "research_report.md.replace")
    journal = report_dir / _JOURNAL_NAME
    assert journal.is_file()
    assert not (report_dir / "research_report.md").exists()

    alias = tmp_path / "alias"
    _symlink_directory(alias, outside)

    with pytest.raises(ReportPublicationError, match="symlink|reparse"):
        publish_report_artifacts(
            alias / report_dir.name,
            {"research_report.html": b"must not publish\n"},
        )

    assert journal.is_file()
    assert not (report_dir / "research_report.md").exists()
    assert (report_dir / "research_report.html").read_bytes() == b"old html\n"


def test_report_lock_key_uses_stable_location_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        assets_module,
        "stable_path_location_identity",
        lambda path: "same-filesystem-location",
        raising=False,
    )

    first = assets_module._report_publication_lock_path(tmp_path / "Report")
    second = assets_module._report_publication_lock_path(tmp_path / "report")

    assert first == second


def test_report_journal_root_binding_uses_identity_instead_of_path_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-spelling"
    second_root = tmp_path / "second-spelling"
    first_root.mkdir()
    second_root.mkdir()
    monkeypatch.setattr(
        assets_module,
        "stable_filesystem_identity",
        lambda path: "same-root-identity",
        raising=False,
    )
    transaction_id = "1" * 32
    target = assets_module._PublicationTarget(
        relative="research_report.md",
        target=first_root / "research_report.md",
        content=b"new\n",
        existed=False,
        baseline=None,
    )
    payload = assets_module._publication_journal_payload(
        first_root,
        [target],
        transaction_id,
        recovery="rollback",
        staging_complete=False,
    )

    recovery, targets = assets_module._validate_report_transaction_journal(
        second_root,
        second_root / _JOURNAL_NAME,
        payload,
    )

    assert recovery == "rollback"
    assert [item.relative for item in targets] == ["research_report.md"]
    assert payload["report_root_identity"] == "same-root-identity"
    assert "report_root" not in payload


def test_first_publication_holds_pre_and_post_materialization_locks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "first-report"
    first_paused = threading.Event()
    allow_first = threading.Event()
    second_done = threading.Event()
    failures: list[BaseException] = []

    def pause_after_journal(label: str) -> None:
        if threading.current_thread().name == "round-25-first" and label == "journal.created":
            first_paused.set()
            assert allow_first.wait(timeout=5)

    def publish(content: bytes) -> None:
        try:
            publish_report_artifacts(report_dir, {"research_report.md": content})
        except BaseException as exc:
            failures.append(exc)

    def publish_second() -> None:
        try:
            publish(b"second\n")
        finally:
            second_done.set()

    monkeypatch.setattr(
        assets_module,
        "_report_publication_precommit_boundary",
        pause_after_journal,
    )
    first = threading.Thread(target=publish, args=(b"first\n",), name="round-25-first")
    second = threading.Thread(target=publish_second, name="round-25-second")
    first.start()
    try:
        assert first_paused.wait(timeout=5)
        assert report_dir.is_dir()
        second.start()
        second_blocked = not second_done.wait(timeout=1)
    finally:
        allow_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_blocked
    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert (report_dir / "research_report.md").read_bytes() == b"second\n"


def test_first_publication_closes_materialization_lock_handoff_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "missing" / "nested" / "report"
    first_materialized = threading.Event()
    allow_first = threading.Event()
    second_done = threading.Event()
    failures: list[BaseException] = []
    original_materialize = assets_module._materialize_report_root

    def pause_after_materialization(plan, *, create: bool, hold_location_lock=None):
        root = original_materialize(
            plan,
            create=create,
            hold_location_lock=hold_location_lock,
        )
        if threading.current_thread().name == "round-25-materializer":
            first_materialized.set()
            assert allow_first.wait(timeout=5)
        return root

    def publish(content: bytes) -> None:
        try:
            publish_report_artifacts(report_dir, {"research_report.md": content})
        except BaseException as exc:
            failures.append(exc)

    def publish_second() -> None:
        try:
            publish(b"second\n")
        finally:
            second_done.set()

    monkeypatch.setattr(assets_module, "_materialize_report_root", pause_after_materialization)
    first = threading.Thread(target=publish, args=(b"first\n",), name="round-25-materializer")
    second = threading.Thread(target=publish_second, name="round-25-after-materialization")
    first.start()
    try:
        assert first_materialized.wait(timeout=5)
        second.start()
        second_blocked = not second_done.wait(timeout=1)
    finally:
        allow_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_blocked
    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert (report_dir / "research_report.md").read_bytes() == b"second\n"


@pytest.mark.parametrize(
    "entrypoint",
    [generate_report, render_html_report],
    ids=["markdown", "html"],
)
def test_public_report_generation_reads_one_run_generation(
    tmp_path: Path,
    entrypoint,
) -> None:
    run_dir = _write_generation_run(tmp_path)
    spec_published = threading.Event()
    allow_metrics = threading.Event()
    reader_done = threading.Event()
    writer_failures: list[BaseException] = []
    reader_failures: list[BaseException] = []
    reports: list[str] = []

    def publish_generation() -> None:
        try:
            with run_digest_transaction(run_dir):
                _write_generation_spec(run_dir, strategy_id="generation_b")
                spec_published.set()
                assert allow_metrics.wait(timeout=5)
                _write_generation_metrics(run_dir, run_id="generation-b")
        except BaseException as exc:
            writer_failures.append(exc)

    def read_report() -> None:
        try:
            reports.append(entrypoint(run_dir, lang="en"))
        except BaseException as exc:
            reader_failures.append(exc)
        finally:
            reader_done.set()

    writer = threading.Thread(target=publish_generation, name="round-25-run-writer")
    reader = threading.Thread(target=read_report, name="round-25-report-reader")
    writer.start()
    try:
        assert spec_published.wait(timeout=5)
        reader.start()
        reader_blocked = not reader_done.wait(timeout=1)
    finally:
        allow_metrics.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert reader_blocked
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_failures == []
    assert reader_failures == []
    assert len(reports) == 1
    assert "generation_b" in reports[0]
    assert "generation-b" in reports[0]


@pytest.mark.parametrize(
    "relative_path",
    [
        "research_report.md",
        "research_report.html",
        "report_assets/manifest.json",
        "report_assets/figures/equity.png",
        "report_assets/scripts/plot.py",
    ],
    ids=["markdown", "html", "manifest", "asset", "script"],
)
def test_standalone_qa_rejects_symlinked_report_inputs(
    tmp_path: Path,
    relative_path: str,
) -> None:
    run_dir = _write_standalone_qa_run(tmp_path)
    path = run_dir / relative_path
    outside = tmp_path / f"outside-{path.name}"
    path.replace(outside)
    try:
        path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="non-symlink regular|symlink|reparse|unsafe"):
        run_report_qa(run_dir)


def test_standalone_qa_rejects_report_file_replaced_during_descriptor_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_standalone_qa_run(tmp_path)
    markdown = run_dir / "research_report.md"
    opened_copy = tmp_path / "opened-report.md"
    replaced = False

    def replace_after_open(path: Path, stage: str) -> None:
        nonlocal replaced
        if path == markdown and stage == "opened" and not replaced:
            replaced = True
            path.replace(opened_copy)
            path.write_text("# replacement\n", encoding="utf-8")

    monkeypatch.setattr(
        qa_module,
        "_report_input_read_boundary",
        replace_after_open,
        raising=False,
    )

    with pytest.raises(ValueError, match="changed during read|coherent"):
        run_report_qa(run_dir)


def test_standalone_qa_rejects_report_file_mutated_during_descriptor_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_standalone_qa_run(tmp_path)
    markdown = run_dir / "research_report.md"
    mutated = False

    def mutate_after_read(path: Path, stage: str) -> None:
        nonlocal mutated
        if path == markdown and stage == "read" and not mutated:
            mutated = True
            path.write_text("# mutated in place\n", encoding="utf-8")

    monkeypatch.setattr(qa_module, "_report_input_read_boundary", mutate_after_read)

    with pytest.raises(ValueError, match="changed during read|coherent"):
        run_report_qa(run_dir)


def test_list_report_assets_waits_for_publication_and_reads_committed_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "AssetRun"
    run_dir.mkdir()
    read_alias = _case_alias_or_same(run_dir)
    _write_asset_snapshot(run_dir, asset_id="old", content=b"old")
    writer_paused = threading.Event()
    allow_writer = threading.Event()
    reader_done = threading.Event()
    writer_failures: list[BaseException] = []
    reader_failures: list[BaseException] = []
    listed_ids: list[list[str]] = []

    def pause_before_manifest_replace(label: str) -> None:
        if label == "report_assets/manifest.json.replace":
            writer_paused.set()
            assert allow_writer.wait(timeout=5)

    def publish() -> None:
        try:
            publish_report_artifacts(
                run_dir,
                _asset_publication(asset_id="new", content=b"new"),
            )
        except BaseException as exc:
            writer_failures.append(exc)

    def read_assets() -> None:
        try:
            listed_ids.append([asset.id for asset in list_report_assets(read_alias)])
        except BaseException as exc:
            reader_failures.append(exc)
        finally:
            reader_done.set()

    monkeypatch.setattr(assets_module, "_report_publication_boundary", pause_before_manifest_replace)
    writer = threading.Thread(target=publish, name="round-25-asset-writer")
    reader = threading.Thread(target=read_assets, name="round-25-asset-reader")
    writer.start()
    try:
        assert writer_paused.wait(timeout=5)
        reader.start()
        reader_blocked = not reader_done.wait(timeout=1)
    finally:
        allow_writer.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert reader_blocked
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_failures == []
    assert reader_failures == []
    assert listed_ids == [["new"]]


def test_list_report_assets_recovers_committed_crash_before_journal_clear(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "AssetRun"
    run_dir.mkdir()
    read_alias = _case_alias_or_same(run_dir)
    _write_asset_snapshot(run_dir, asset_id="old", content=b"old")

    _crash_before_journal_clear(run_dir)

    assert (run_dir / _JOURNAL_NAME).is_file()
    assert [asset.id for asset in list_report_assets(read_alias)] == ["new"]
    assert not (run_dir / _JOURNAL_NAME).exists()


@pytest.mark.parametrize(
    "boundary",
    [
        "research_report.md.stage-created",
        "baseline.validated",
        "journal.staged",
    ],
    ids=["partial-stage", "baseline", "journal-transition"],
)
def test_pre_staging_journal_recovers_crashes_at_precommit_boundaries(
    tmp_path: Path,
    boundary: str,
) -> None:
    report_dir = _write_report_pair(tmp_path)

    _crash_at_publication_boundary(report_dir, boundary)

    assert (report_dir / _JOURNAL_NAME).is_file()
    with report_publication_read_transaction(report_dir):
        assert (report_dir / "research_report.md").read_bytes() == b"old markdown\n"
        assert (report_dir / "research_report.html").read_bytes() == b"old html\n"
    assert not (report_dir / _JOURNAL_NAME).exists()
    assert list(report_dir.glob(".*.oxq-report-*")) == []


def _write_report_pair(parent: Path) -> Path:
    report_dir = parent / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "research_report.md").write_bytes(b"old markdown\n")
    (report_dir / "research_report.html").write_bytes(b"old html\n")
    return report_dir


def _symlink_directory(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


def _case_alias_or_same(path: Path) -> Path:
    alias = path.with_name(path.name.swapcase())
    try:
        return alias if alias != path and os.path.samefile(path, alias) else path
    except OSError:
        return path


def _write_generation_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "generation-run"
    run_dir.mkdir()
    _write_generation_spec(run_dir, strategy_id="generation_a")
    _write_generation_metrics(run_dir, run_id="generation-a")
    return run_dir


def _write_generation_spec(run_dir: Path, *, strategy_id: str) -> None:
    spec = StrategySpec.template(
        strategy_id=strategy_id,
        hypothesis=f"{strategy_id} hypothesis",
    )
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )


def _write_generation_metrics(run_dir: Path, *, run_id: str) -> None:
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "trade_count": 1,
                "max_drawdown": -0.01,
                "total_return": 0.02,
                "annualized_return": 0.02,
                "annualized_volatility": 0.01,
                "sharpe_ratio": 1.0,
            }
        ),
        encoding="utf-8",
    )


def _write_standalone_qa_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "qa-run"
    run_dir.mkdir()
    spec = StrategySpec.template(
        strategy_id="round_25_qa",
        hypothesis="QA must read stable report inputs",
    )
    spec.validation.train_period = ["2024-01-02", "2024-01-31"]
    spec.validation.test_period = ["2024-02-01", "2024-03-31"]
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "trade_count": 2,
                "oos_trade_count": 1,
                "total_return": 0.2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n"
        "2024-01-02,100\n"
        "2024-01-31,110\n"
        "2024-02-29,99\n"
        "2024-03-29,120\n",
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text(
        "symbol,side,shares,filled_price,filled_at,fee\n"
        "AAA,BUY,1,10,2024-01-15,0\n"
        "AAA,SELL,1,11,2024-02-15,0\n",
        encoding="utf-8",
    )
    figure = run_dir / "report_assets/figures/equity.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(_PNG_BYTES)
    script = run_dir / "report_assets/scripts/plot.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('plot')\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "assets": [
            {
                "id": "equity",
                "kind": "figure",
                "path": "figures/equity.png",
                "title": "Equity",
                "caption": "",
                "section": "results",
                "order": 10,
                "mime_type": "image/png",
                "sha256": _sha256_bytes(_PNG_BYTES),
                "source": {"script": "scripts/plot.py"},
            }
        ],
    }
    (run_dir / "report_assets/manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Equity](report_assets/figures/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(
        render_markdown_html_report(markdown, lang="en"),
        encoding="utf-8",
    )
    return run_dir


def _write_asset_snapshot(run_dir: Path, *, asset_id: str, content: bytes) -> None:
    publication = _asset_publication(asset_id=asset_id, content=content)
    for relative, data in publication.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        assert data is not None
        path.write_bytes(data)


def _asset_publication(*, asset_id: str, content: bytes) -> dict[str, bytes]:
    manifest = {
        "schema_version": 1,
        "assets": [
            {
                "id": asset_id,
                "kind": "figure",
                "path": f"figures/{asset_id}.png",
                "title": asset_id.title(),
                "caption": "",
                "section": "results",
                "order": 10,
                "mime_type": "image/png",
                "sha256": _sha256_bytes(content),
            }
        ],
    }
    return {
        f"report_assets/figures/{asset_id}.png": content,
        "report_assets/manifest.json": (json.dumps(manifest) + "\n").encode(),
    }


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _crash_at_publication_boundary(report_dir: Path, boundary: str) -> None:
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        import oxq.report.assets as assets_module

        def crash(label: str) -> None:
            if label == sys.argv[2]:
                os._exit(87)

        assets_module._report_publication_boundary = crash
        assets_module._report_publication_precommit_boundary = crash
        assets_module.publish_report_artifacts(
            Path(sys.argv[1]),
            {
                "research_report.md": b"new markdown\\n",
                "research_report.html": b"new html\\n",
            },
        )
        """
    )
    completed = _run_crash_script(script, report_dir, boundary)
    assert completed.returncode == _CRASH_EXIT_CODE


def _crash_before_journal_clear(run_dir: Path) -> None:
    script = textwrap.dedent(
        """
        import json
        import os
        import sys
        from pathlib import Path

        import oxq.report.assets as assets_module

        content = b"new"
        manifest = {
            "schema_version": 1,
            "assets": [{
                "id": "new",
                "kind": "figure",
                "path": "figures/new.png",
                "title": "New",
                "caption": "",
                "section": "results",
                "order": 10,
                "mime_type": "image/png",
                "sha256": "sha256:" + __import__("hashlib").sha256(content).hexdigest(),
            }],
        }

        def crash(path: Path) -> None:
            os._exit(87)

        assets_module._clear_report_transaction_journal = crash
        assets_module.publish_report_artifacts(
            Path(sys.argv[1]),
            {
                "report_assets/figures/new.png": content,
                "report_assets/manifest.json": (json.dumps(manifest) + "\\n").encode(),
            },
        )
        """
    )
    completed = _run_crash_script(script, run_dir)
    assert completed.returncode == _CRASH_EXIT_CODE


def _run_crash_script(script: str, *args: object) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source_root), environment.get("PYTHONPATH")) if item
    )
    return subprocess.run(
        [sys.executable, "-c", script, *(str(arg) for arg in args)],
        check=False,
        env=environment,
    )
