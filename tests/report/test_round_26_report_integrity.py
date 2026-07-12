from __future__ import annotations

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
import oxq.run_digests as run_digests_module
from oxq.report.assets import (
    ReportPublicationError,
    add_report_assets,
    list_report_assets,
    publish_report_artifacts,
    report_publication_read_transaction,
    safe_asset_id,
)
from oxq.report.generator import write_report_files
from oxq.report.html import render_markdown_html_report
from oxq.report.qa import run_report_qa
from oxq.run_digests import publish_run_artifacts
from oxq.spec.schema import StrategySpec

_JOURNAL_NAME = ".oxq-report-transaction.json"


def test_direct_qa_waits_for_governed_run_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_qa_run(tmp_path / "runs", run_id="governed-run")
    _write_legacy_run_integrity(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_return"] = 0.9
    publication_paused = threading.Event()
    allow_publication = threading.Event()
    reader_done = threading.Event()
    publisher_failures: list[BaseException] = []
    reader_failures: list[BaseException] = []
    reader_results = []
    original_boundary = run_digests_module._publication_boundary

    def pause_after_metrics_replace(label: str) -> None:
        if threading.current_thread().name == "round-26-run-publisher" and label == "artifact:metrics.json.replace":
            publication_paused.set()
            assert allow_publication.wait(timeout=5)
        original_boundary(label)

    def publish_metrics() -> None:
        try:
            publish_run_artifacts(run_dir, {"metrics.json": json.dumps(metrics).encode()})
        except BaseException as exc:
            publisher_failures.append(exc)

    def inspect_run() -> None:
        try:
            reader_results.append(run_report_qa(run_dir))
        except BaseException as exc:
            reader_failures.append(exc)
        finally:
            reader_done.set()

    monkeypatch.setattr(run_digests_module, "_publication_boundary", pause_after_metrics_replace)
    publisher = threading.Thread(target=publish_metrics, name="round-26-run-publisher")
    reader = threading.Thread(target=inspect_run, name="round-26-run-qa")
    publisher.start()
    try:
        assert publication_paused.wait(timeout=5)
        reader.start()
        reader_completed_while_uncommitted = reader_done.wait(timeout=1)
    finally:
        allow_publication.set()
    publisher.join(timeout=5)
    reader.join(timeout=5)

    assert not publisher.is_alive()
    assert not reader.is_alive()
    assert not reader_completed_while_uncommitted
    assert publisher_failures == []
    assert reader_failures == []
    known_numbers = {item["name"]: item["value"] for item in reader_results[0].facts.known_numbers}
    assert known_numbers["metric.total_return"] == 0.9


def test_reciprocal_custom_exports_do_not_deadlock(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import json
        import sys
        import threading
        from contextlib import contextmanager
        from pathlib import Path

        import yaml

        import oxq.report.assets as assets_module
        from oxq.report.generator import write_report_files
        from oxq.spec.schema import StrategySpec

        root = Path(sys.argv[1])

        def write_run(parent: Path, run_id: str) -> Path:
            run_dir = parent / "run"
            run_dir.mkdir(parents=True)
            spec = StrategySpec.template(strategy_id=run_id, hypothesis=f"{run_id} hypothesis")
            spec.validation.train_period = []
            spec.validation.test_period = ["2024-01-02", "2024-01-03"]
            spec.validation.required_oos = False
            (run_dir / "strategy_spec.yaml").write_text(
                yaml.safe_dump(spec.to_dict(), sort_keys=False),
                encoding="utf-8",
            )
            (run_dir / "metrics.json").write_text(
                json.dumps({
                    "run_id": run_id,
                    "trade_count": 1,
                    "max_drawdown": -0.01,
                    "total_return": 0.02,
                    "annualized_return": 0.02,
                    "annualized_volatility": 0.01,
                    "sharpe_ratio": 1.0,
                }),
                encoding="utf-8",
            )
            return run_dir

        run_a = write_run(root / "a", "run-a")
        run_b = write_run(root / "b", "run-b")
        barrier = threading.Barrier(2)
        local = threading.local()
        original_hold = assets_module._hold_report_publication_root

        @contextmanager
        def synchronize_first_report_lock(plan, *, create: bool):
            with original_hold(plan, create=create) as report_root:
                if not getattr(local, "synchronized", False):
                    local.synchronized = True
                    barrier.wait(timeout=5)
                yield report_root

        assets_module._hold_report_publication_root = synchronize_first_report_lock
        failures = []

        def export(source: Path, destination: Path, name: str) -> None:
            try:
                write_report_files(
                    source,
                    lang="en",
                    output_format="markdown",
                    out=destination / name,
                )
            except BaseException as exc:
                failures.append(repr(exc))

        workers = [
            threading.Thread(target=export, args=(run_a, run_b, "from-a.md"), daemon=True),
            threading.Thread(target=export, args=(run_b, run_a, "from-b.md"), daemon=True),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=6)
        if any(worker.is_alive() for worker in workers):
            raise SystemExit(3)
        if failures:
            print("\\n".join(failures), file=sys.stderr)
            raise SystemExit(4)
        if not (run_a / "from-b.md").is_file() or not (run_b / "from-a.md").is_file():
            raise SystemExit(5)
        """
    )

    completed = _run_script(script, tmp_path, timeout=15)

    assert completed.returncode == 0, completed.stderr.decode()


def test_custom_export_snapshots_assets_before_destination_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = _write_generation_run(tmp_path / "source")
    _write_asset_snapshot(run_dir, content=b"old-generation")
    export_dir = tmp_path / "export"
    first_source_file_read = threading.Event()
    allow_export = threading.Event()
    publisher_started = threading.Event()
    publisher_done = threading.Event()
    export_failures: list[BaseException] = []
    publisher_failures: list[BaseException] = []
    original_read_bytes = Path.read_bytes
    paused = False

    def pause_after_first_source_file(path: Path) -> bytes:
        nonlocal paused
        content = original_read_bytes(path)
        if threading.current_thread().name == "round-26-custom-export" and not paused and path.is_relative_to(run_dir / "report_assets"):
            paused = True
            first_source_file_read.set()
            assert allow_export.wait(timeout=5)
        return content

    def export_report() -> None:
        try:
            write_report_files(
                run_dir,
                lang="en",
                output_format="markdown",
                out=export_dir / "research_report.md",
            )
        except BaseException as exc:
            export_failures.append(exc)

    def replace_assets() -> None:
        publisher_started.set()
        try:
            publish_report_artifacts(run_dir, _asset_publication(content=b"new-generation"))
        except BaseException as exc:
            publisher_failures.append(exc)
        finally:
            publisher_done.set()

    monkeypatch.setattr(Path, "read_bytes", pause_after_first_source_file)
    exporter = threading.Thread(target=export_report, name="round-26-custom-export")
    publisher = threading.Thread(target=replace_assets, name="round-26-asset-publisher")
    exporter.start()
    try:
        assert first_source_file_read.wait(timeout=5)
        publisher.start()
        assert publisher_started.wait(timeout=5)
        publisher_completed_while_snapshot_paused = publisher_done.wait(timeout=1)
    finally:
        allow_export.set()
    exporter.join(timeout=5)
    publisher.join(timeout=5)

    assert not exporter.is_alive()
    assert not publisher.is_alive()
    assert not publisher_completed_while_snapshot_paused
    assert export_failures == []
    assert publisher_failures == []
    manifest = json.loads((export_dir / "report_assets/manifest.json").read_text(encoding="utf-8"))
    exported_asset = export_dir / "report_assets/figures/chart.png"
    assert manifest["assets"][0]["sha256"] == _sha256_bytes(exported_asset.read_bytes())


@pytest.mark.parametrize("use_builder", [False, True], ids=["mapping", "builder"])
@pytest.mark.parametrize(
    "targets",
    [
        {"parent/child.txt": b"child", "parent": b"parent"},
        {"Chart.txt": b"upper", "chart.txt": b"lower"},
        {"caf\u00e9.txt": b"composed", "cafe\u0301.txt": b"decomposed"},
        {".OXQ-REPORT-TRANSACTION.JSON": b"alias"},
        {".asset.oxq-report-11111111111111111111111111111111.new": b"internal"},
        {".oxq-report-transaction.json/child": b"descendant"},
    ],
    ids=["ancestor", "casefold", "unicode", "journal-alias", "internal", "journal-descendant"],
)
def test_publication_rejects_overlapping_or_reserved_targets_before_mkdir(
    tmp_path: Path,
    targets: dict[str, bytes],
    use_builder: bool,
) -> None:
    report_dir = tmp_path / "missing" / "report"
    publication = (lambda: targets) if use_builder else targets

    with pytest.raises(ReportPublicationError, match="overlap|collision|portable|reserved"):
        publish_report_artifacts(report_dir, publication)

    assert not report_dir.exists()


@pytest.mark.parametrize(
    "target",
    ["CON.txt", "nested/AUX.log", "COM1.csv", "bad<name>.txt", "trailing.", "trailing "],
)
def test_publication_rejects_nonportable_targets_before_mkdir(tmp_path: Path, target: str) -> None:
    report_dir = tmp_path / "missing" / "report"

    with pytest.raises(ReportPublicationError, match="portable|safe relative path"):
        publish_report_artifacts(report_dir, {target: b"invalid"})

    assert not report_dir.exists()


def test_publication_preserves_valid_posix_nested_targets(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"

    publish_report_artifacts(report_dir, {"results/2026/\u56fe\u8868.txt": b"valid"})

    assert (report_dir / "results/2026/\u56fe\u8868.txt").read_bytes() == b"valid"


def test_publication_rejects_existing_portable_alias_without_staging(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    existing = report_dir / "chart.txt"
    existing.write_bytes(b"original")

    with pytest.raises(ReportPublicationError, match="alias|collision|portable"):
        publish_report_artifacts(report_dir, {"Chart.txt": b"replacement"})

    assert existing.read_bytes() == b"original"
    assert not (report_dir / _JOURNAL_NAME).exists()
    assert list(report_dir.glob(".*.oxq-report-*")) == []


def test_publication_rejects_existing_journal_alias_without_mutation(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    journal_alias = report_dir / ".OXQ-REPORT-TRANSACTION.JSON"
    journal_alias.write_bytes(b"reserved alias")

    with pytest.raises(ReportPublicationError, match="journal.*alias|portable.*collision"):
        publish_report_artifacts(report_dir, {"research_report.md": b"report"})

    assert journal_alias.read_bytes() == b"reserved alias"
    assert _JOURNAL_NAME not in os.listdir(report_dir)
    assert not (report_dir / "research_report.md").exists()


@pytest.mark.parametrize("linked_entry", ["manifest", "asset"])
def test_list_report_assets_rejects_symlinked_inputs(tmp_path: Path, linked_entry: str) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_asset_snapshot(run_dir, content=b"asset")
    linked_path = run_dir / "report_assets/manifest.json" if linked_entry == "manifest" else run_dir / "report_assets/figures/chart.png"
    outside = tmp_path / f"outside-{linked_entry}"
    linked_path.replace(outside)
    try:
        linked_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink|reparse|non-symlink regular|changed"):
        list_report_assets(run_dir)


@pytest.mark.parametrize("replaced_entry", ["manifest", "asset"])
def test_list_report_assets_rejects_descriptor_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replaced_entry: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_asset_snapshot(run_dir, content=b"asset")
    target = run_dir / "report_assets/manifest.json" if replaced_entry == "manifest" else run_dir / "report_assets/figures/chart.png"
    opened_copy = tmp_path / f"opened-{target.name}"
    replaced = False

    def replace_after_open(path: Path, stage: str) -> None:
        nonlocal replaced
        if path == target and stage == "opened" and not replaced:
            replaced = True
            path.replace(opened_copy)
            path.write_bytes(opened_copy.read_bytes())

    monkeypatch.setattr(
        assets_module,
        "_report_asset_input_read_boundary",
        replace_after_open,
        raising=False,
    )

    with pytest.raises(ValueError, match="changed during read|coherent"):
        list_report_assets(run_dir)


@pytest.mark.parametrize(
    "asset_id",
    ["CON", "nul.txt", "COM1.csv", "bad<id>", "trailing.", "trailing ", "\uff23\uff2f\uff2e.txt"],
)
def test_asset_ids_reject_nonportable_components(asset_id: str) -> None:
    with pytest.raises(ValueError, match="invalid asset id"):
        safe_asset_id(asset_id)


def test_asset_ids_accept_unicode_and_reject_normalized_batch_aliases(tmp_path: Path) -> None:
    assert safe_asset_id("caf\u00e9") == "caf\u00e9"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with pytest.raises(ValueError, match="asset id.*collision|portable.*collision"):
        add_report_assets(
            run_dir,
            [
                {"id": "caf\u00e9", "file_path": first, "title": "First"},
                {"id": "cafe\u0301", "file_path": second, "title": "Second"},
            ],
        )

    assert not (run_dir / "report_assets").exists()


def test_list_report_assets_rejects_manifest_aliases_before_asset_reads(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = run_dir / "report_assets/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    _manifest_asset(asset_id="Chart", path="figures/Chart.png", content=b"first"),
                    _manifest_asset(asset_id="chart", path="figures/chart.png", content=b"second"),
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="asset id.*collision|asset path.*collision|portable.*collision"):
        list_report_assets(run_dir)


def test_report_qa_reports_nonportable_manifest_names_and_aliases(tmp_path: Path) -> None:
    run_dir = _write_qa_run(tmp_path, run_id="qa-portable")
    manifest = run_dir / "report_assets/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    _manifest_asset(asset_id="Chart", path="figures/CON.png", content=b"first"),
                    _manifest_asset(asset_id="chart", path="figures/con.png", content=b"second"),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_report_qa(run_dir)

    finding_ids = {finding.id for finding in result.findings}
    assert "asset_id_collision" in finding_ids
    assert "asset_path_invalid" in finding_ids


@pytest.mark.parametrize(
    "journal_targets",
    [["CON.txt"], ["Chart.txt", "chart.txt"], ["parent", "parent/child.txt"]],
    ids=["device", "casefold", "ancestor"],
)
def test_recovery_rejects_nonportable_or_overlapping_journal_targets(
    tmp_path: Path,
    journal_targets: list[str],
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    journal = report_dir / _JOURNAL_NAME
    journal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": "1" * 32,
                "recovery": "rollback",
                "staging_complete": False,
                "report_root_identity": assets_module.stable_filesystem_identity(report_dir),
                "targets": [{"path": target, "old": None, "new": None} for target in journal_targets],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReportPublicationError, match="unsafe|portable|overlap|collision"):
        with report_publication_read_transaction(report_dir):
            pass

    assert journal.is_file()


def _write_generation_run(parent: Path) -> Path:
    run_dir = parent / "run"
    run_dir.mkdir(parents=True)
    spec = StrategySpec.template(strategy_id="round_26", hypothesis="snapshot one report generation")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
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
    return run_dir


def _write_qa_run(parent: Path, *, run_id: str) -> Path:
    run_dir = parent / run_id
    run_dir.mkdir(parents=True)
    spec = StrategySpec.template(strategy_id="round_26_qa", hypothesis="QA must read one generation")
    spec.validation.train_period = ["2024-01-02", "2024-01-31"]
    spec.validation.test_period = ["2024-02-01", "2024-03-31"]
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": run_id, "trade_count": 2, "oos_trade_count": 1, "total_return": 0.2}),
        encoding="utf-8",
    )
    (run_dir / "equity_curve.csv").write_text(
        "date,value\n2024-01-02,100\n2024-01-31,110\n2024-02-29,99\n2024-03-29,120\n",
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text(
        "symbol,side,shares,filled_price,filled_at,fee\nAAA,BUY,1,10,2024-01-15,0\nAAA,SELL,1,11,2024-02-15,0\n",
        encoding="utf-8",
    )
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(
        render_markdown_html_report(markdown, lang="en"),
        encoding="utf-8",
    )
    return run_dir


def _write_legacy_run_integrity(run_dir: Path) -> None:
    (run_dir / "data_manifest.json").write_text("{}\n", encoding="utf-8")
    artifact_hashes = {
        "data_manifest.json": _canonical_json_hash(run_dir / "data_manifest.json"),
        "equity_curve.csv": _short_file_hash(run_dir / "equity_curve.csv"),
        "trades.csv": _short_file_hash(run_dir / "trades.csv"),
        "metrics.json": _canonical_json_hash(run_dir / "metrics.json", exclude_keys={"run_id"}),
    }
    manifest = run_dir / "artifact_hashes.json"
    manifest.write_text(json.dumps(artifact_hashes, indent=2) + "\n", encoding="utf-8")
    digest = {
        "run_id": run_dir.name,
        "artifact_hashes": _canonical_json_hash(manifest),
    }
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps(digest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_asset_snapshot(run_dir: Path, *, content: bytes) -> None:
    for relative, data in _asset_publication(content=content).items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _asset_publication(*, content: bytes) -> dict[str, bytes]:
    manifest = {
        "schema_version": 1,
        "assets": [_manifest_asset(asset_id="chart", path="figures/chart.png", content=content)],
    }
    return {
        "report_assets/figures/chart.png": content,
        "report_assets/manifest.json": (json.dumps(manifest) + "\n").encode(),
    }


def _manifest_asset(*, asset_id: str, path: str, content: bytes) -> dict[str, object]:
    return {
        "id": asset_id,
        "kind": "figure",
        "path": path,
        "title": asset_id,
        "caption": "",
        "section": "results",
        "order": 10,
        "mime_type": "image/png",
        "sha256": _sha256_bytes(content),
    }


def _canonical_json_hash(path: Path, *, exclude_keys: set[str] | None = None) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and exclude_keys:
        payload = {key: value for key, value in payload.items() if key not in exclude_keys}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _short_file_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _run_script(script: str, *args: object, timeout: float) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(item for item in (str(source_root), environment.get("PYTHONPATH")) if item)
    return subprocess.run(
        [sys.executable, "-c", script, *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        env=environment,
        timeout=timeout,
    )
