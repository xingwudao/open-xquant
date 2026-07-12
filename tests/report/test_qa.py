from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import oxq.report.qa as qa_module
import oxq.run_digests as run_digests_module
from oxq.report.assets import add_report_asset, publish_report_artifacts
from oxq.report.html import render_markdown_html_report
from oxq.report.qa import run_report_qa as _run_report_qa
from oxq.run_digests import publish_run_artifacts, run_digest_transaction
from oxq.spec.schema import StrategySpec


def run_report_qa(run_dir, **kwargs):
    kwargs.setdefault("include_advisory_checks", True)
    return _run_report_qa(run_dir, **kwargs)


def test_report_qa_passes_complete_registered_report(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    script = run_dir / "report_assets/scripts/plot.py"
    script.parent.mkdir(parents=True)
    script.write_text('import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\n', encoding="utf-8")
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(
        run_dir,
        figure,
        asset_id="equity",
        title="策略净值",
        caption="由 equity_curve.csv 生成。",
        section="results",
        order=10,
        source_script=script,
        source_artifacts=["equity_curve.csv"],
    )
    markdown = (
        "# 研究报告\n\n"
        "有效数据最后交易日：2024-03-29\n\n"
        "配置结束日：2024-03-31\n\n"
        "总收益为 20.00%，正收益月份 2 个，负收益月份 1 个。\n\n"
        "![策略净值](report_assets/figures/equity.png)\n\n"
        "图 1. 由 equity_curve.csv 生成。\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"
    assert result.fatal_count == 0
    assert result.warning_count == 0
    assert result.facts.configured_end_date == "2024-03-31"
    assert result.facts.effective_last_trading_day == "2024-03-29"


def test_report_qa_preserves_non_governed_direct_qa_without_spec_hash(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    (run_dir / "spec_hash.txt").unlink()
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(
        render_markdown_html_report(markdown, lang="en"),
        encoding="utf-8",
    )

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_preserves_legacy_direct_qa_under_directory_named_10_reports(tmp_path) -> None:
    reports_dir = tmp_path / "arbitrary" / "10_reports"
    reports_dir.mkdir(parents=True)
    run_dir = _write_qa_run(reports_dir)
    (run_dir / "spec_hash.txt").unlink()
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(
        render_markdown_html_report(markdown, lang="en"),
        encoding="utf-8",
    )

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_accepts_version_governed_report_package(tmp_path) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_qa_case"
    version_dir = tmp_path / "versions" / "v001"
    run_dir = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, run_dir)
    _set_metrics_run_id(run_dir, run_id)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )
    markdown = "# 研究报告\n\n有效数据最后交易日：2024-03-29\n\n配置结束日：2024-03-31\n"
    (report_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    result = run_report_qa(report_dir)

    assert result.status == "pass"
    assert result.fatal_count == 0
    assert result.facts.configured_end_date == "2024-03-31"
    assert result.facts.effective_last_trading_day == "2024-03-29"


def test_report_qa_rejects_symlinked_governed_report_package(tmp_path) -> None:
    _, report_dir = _write_governed_qa_package(tmp_path, run_id="symlinked_package")
    outside_report = tmp_path / "outside" / report_dir.name
    outside_report.parent.mkdir()
    report_dir.rename(outside_report)
    report_dir.symlink_to(outside_report, target_is_directory=True)

    with pytest.raises(ValueError, match="report package.*symlink"):
        run_report_qa(report_dir)


def test_report_qa_rejects_symlinked_governed_report_file(tmp_path) -> None:
    _, report_dir = _write_governed_qa_package(tmp_path, run_id="symlinked_file")
    markdown_path = report_dir / "research_report.md"
    outside_markdown = tmp_path / "outside_report.md"
    markdown_path.rename(outside_markdown)
    markdown_path.symlink_to(outside_markdown)

    with pytest.raises(ValueError, match="report package.*symlink"):
        run_report_qa(report_dir)


def test_report_qa_rejects_symlinked_governed_metrics_before_reading_target(
    monkeypatch,
    tmp_path,
) -> None:
    source_run, report_dir = _write_governed_qa_package(
        tmp_path,
        run_id="symlinked_metrics",
    )
    metrics_path = source_run / "metrics.json"
    external_metrics = tmp_path / "external_metrics.json"
    metrics_path.rename(external_metrics)
    metrics_path.symlink_to(external_metrics)
    original_read_text = Path.read_text

    def reject_external_read(path: Path, *args, **kwargs) -> str:
        if path == metrics_path:
            raise AssertionError("governed QA read a symlinked external metrics target")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_external_read)

    with pytest.raises(ValueError, match="metrics.json.*symlink"):
        run_report_qa(report_dir)


@pytest.mark.parametrize(
    ("workspace_text", "current_text", "manifest_text", "message"),
    [
        ("workflow: [\n", '{"active_version": "v001"}', "{}", "workspace.yaml"),
        (
            "workflow:\n  layout: version_governed\npaths:\n  versions_dir: versions\n",
            "{not-json",
            "{}",
            "current.json",
        ),
        (
            "workflow:\n  layout: version_governed\npaths:\n  versions_dir: versions\n",
            '{"active_version": "v001"}',
            "{not-json",
            "version_manifest.json",
        ),
    ],
    ids=["workspace", "current", "version-manifest"],
)
def test_report_qa_fails_closed_for_malformed_governed_workspace_context(
    tmp_path,
    workspace_text: str,
    current_text: str,
    manifest_text: str,
    message: str,
) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    (config_dir / "workspace.yaml").write_text(workspace_text, encoding="utf-8")
    (tmp_path / "current.json").write_text(current_text, encoding="utf-8")
    version_dir = tmp_path / "versions/v001"
    report_dir = version_dir / "10_reports/run_1"
    report_dir.mkdir(parents=True)
    (version_dir / "version_manifest.json").write_text(manifest_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_report_qa(report_dir)


def test_report_qa_rejects_present_malformed_versions_dir_before_nearest_manifest_fallback(tmp_path) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump({"paths": {"versions_dir": []}}, sort_keys=False),
        encoding="utf-8",
    )
    version_dir = tmp_path / "versions/v001"
    report_dir = version_dir / "10_reports/run_1"
    report_dir.mkdir(parents=True)
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "phase_paths": {
                    "09_backtests": "versions/v001/09_backtests",
                    "10_reports": "versions/v001/10_reports",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paths.versions_dir must be a safe relative path"):
        run_report_qa(report_dir)


def test_report_qa_rejects_report_outside_configured_governed_root_before_nearest_manifest_fallback(
    tmp_path,
) -> None:
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump({"paths": {"versions_dir": "versions"}}, sort_keys=False),
        encoding="utf-8",
    )
    version_dir = tmp_path / "outside/v001"
    report_dir = version_dir / "10_reports/run_1"
    report_dir.mkdir(parents=True)
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "phase_paths": {
                    "09_backtests": "outside/v001/09_backtests",
                    "10_reports": "outside/v001/10_reports",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configured versions root"):
        run_report_qa(report_dir)


@pytest.mark.parametrize("symlink_component", ["phase", "run"], ids=["phase", "direct-child"])
def test_report_qa_rejects_legacy_canonical_source_symlink_components(tmp_path, symlink_component: str) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    outside_run = _write_qa_run(fixture)
    run_id = "run_1"
    _set_metrics_run_id(outside_run, run_id)
    version_dir = tmp_path / "versions/v001"
    backtest_phase = version_dir / "09_backtests"
    if symlink_component == "phase":
        outside_phase = tmp_path / "outside_backtests"
        outside_phase.mkdir()
        outside_run.rename(outside_phase / run_id)
        backtest_phase.parent.mkdir(parents=True)
        backtest_phase.symlink_to(outside_phase, target_is_directory=True)
    else:
        backtest_phase.mkdir(parents=True)
        (backtest_phase / run_id).symlink_to(outside_run, target_is_directory=True)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )

    with pytest.raises(ValueError, match="source_run_dir.*symlink"):
        run_report_qa(report_dir)


def test_report_qa_accepts_governed_canonical_spec_hash_after_yaml_reformat(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="canonical_hash")
    spec_path = source_run / "strategy_spec.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=True), encoding="utf-8")

    result = run_report_qa(report_dir)

    assert result.status == "pass"


def test_report_qa_accepts_canonically_equivalent_artifact_hash_manifest(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="canonical_artifact_hashes")
    manifest_path = source_run / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps(manifest, indent=4, sort_keys=True) + "\n", encoding="utf-8")

    result = run_report_qa(report_dir)

    assert result.status == "pass"


def test_report_qa_rejects_tampered_governed_metrics(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="tampered_metrics")
    metrics_path = source_run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["total_return"] = 9.99
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics_hash"):
        run_report_qa(report_dir)


def test_report_qa_requires_metrics_hash_entry_for_governed_source(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="missing_metrics_hash")
    manifest_path = source_run / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["metrics.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_current_run_digest(source_run)

    with pytest.raises(ValueError, match="artifact_hashes.*metrics.json"):
        run_report_qa(report_dir)


def test_report_qa_rejects_stale_governed_artifact_hashes_digest(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="stale_hash_manifest_digest")
    manifest_path = source_run / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics.json"] = "sha256:" + "0" * 16
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="run_digest"):
        run_report_qa(report_dir)


def test_report_qa_rejects_tampered_governed_run_digest(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="tampered_run_digest")
    digest_path = source_run.parent / "run_digests.jsonl"
    digest_path.write_text(
        json.dumps({"run_id": source_run.name, "artifact_hashes": "sha256:" + "0" * 16}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run_digest"):
        run_report_qa(report_dir)


@pytest.mark.parametrize("boundary", ["artifact:metrics.json.replace", "manifest.replace"])
def test_report_qa_waits_for_paused_source_publication(
    monkeypatch,
    tmp_path,
    boundary: str,
) -> None:
    boundary_id = boundary.partition(".")[0].replace(":", "_")
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id=f"paused_{boundary_id}")
    metrics = json.loads((source_run / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_return"] = 0.9
    publication_paused = threading.Event()
    allow_publication = threading.Event()
    reader_attempted = threading.Event()
    reader_completed = threading.Event()
    publisher_failures: list[BaseException] = []
    reader_results = []
    reader_failures: list[BaseException] = []
    original_boundary = run_digests_module._publication_boundary

    def pause_publication(label: str) -> None:
        if threading.current_thread().name == "qa-publisher" and label == boundary:
            publication_paused.set()
            assert allow_publication.wait(timeout=5)
        original_boundary(label)

    @contextmanager
    def observed_transaction(run_path):
        if threading.current_thread().name == "qa-reader":
            reader_attempted.set()
        with run_digest_transaction(run_path):
            yield

    def publish_metrics() -> None:
        try:
            publish_run_artifacts(source_run, {"metrics.json": json.dumps(metrics).encode()})
        except BaseException as exc:
            publisher_failures.append(exc)

    def read_report() -> None:
        try:
            reader_results.append(run_report_qa(report_dir))
        except BaseException as exc:
            reader_failures.append(exc)
        finally:
            reader_completed.set()

    monkeypatch.setattr(run_digests_module, "_publication_boundary", pause_publication)
    monkeypatch.setattr("oxq.report.qa.run_digest_transaction", observed_transaction, raising=False)
    publisher = threading.Thread(target=publish_metrics, name="qa-publisher")
    reader = threading.Thread(target=read_report, name="qa-reader")
    publisher.start()
    try:
        assert publication_paused.wait(timeout=5)
        reader.start()
        assert reader_attempted.wait(timeout=5)
        assert not reader_completed.is_set()
    finally:
        allow_publication.set()
    publisher.join(timeout=5)
    reader.join(timeout=5)

    assert not publisher.is_alive()
    assert not reader.is_alive()
    assert publisher_failures == []
    assert reader_failures == []
    assert {item["name"]: item["value"] for item in reader_results[0].facts.known_numbers}["metric.total_return"] == 0.9


def test_governed_report_qa_holds_final_lock_through_all_report_package_reads(
    monkeypatch,
    tmp_path,
) -> None:
    _, report_dir = _write_governed_qa_package(tmp_path, run_id="paused_report_read")
    markdown_read = threading.Event()
    allow_qa = threading.Event()
    publisher_attempted = threading.Event()
    publisher_completed = threading.Event()
    qa_failures: list[BaseException] = []
    publisher_failures: list[BaseException] = []
    original_read_text = qa_module._read_text

    def pause_after_markdown(path, findings, label):
        content = original_read_text(path, findings, label)
        if threading.current_thread().name == "package-qa" and label == "research_report.md":
            markdown_read.set()
            assert allow_qa.wait(timeout=5)
        return content

    def read_package() -> None:
        try:
            run_report_qa(report_dir)
        except BaseException as exc:
            qa_failures.append(exc)

    def publish_report() -> None:
        publisher_attempted.set()
        try:
            publish_report_artifacts(
                report_dir,
                {"research_report.html": b"replacement html\n"},
            )
        except BaseException as exc:
            publisher_failures.append(exc)
        finally:
            publisher_completed.set()

    monkeypatch.setattr(qa_module, "_read_text", pause_after_markdown)
    qa_worker = threading.Thread(target=read_package, name="package-qa")
    publisher = threading.Thread(target=publish_report, name="package-publisher")
    qa_worker.start()
    try:
        assert markdown_read.wait(timeout=5)
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
    assert (report_dir / "research_report.html").read_bytes() == b"replacement html\n"


def test_governed_report_qa_re_resolves_package_after_lock_acquisition(
    monkeypatch,
    tmp_path,
) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="reresolve_package")
    version_manifest = report_dir.parent.parent / "version_manifest.json"
    transaction_attempted = threading.Event()
    failures: list[BaseException] = []
    original_transaction = run_digest_transaction

    @contextmanager
    def observed_transaction(run_path):
        transaction_attempted.set()
        with original_transaction(run_path):
            yield

    def read_package() -> None:
        try:
            run_report_qa(report_dir)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(qa_module, "run_digest_transaction", observed_transaction)
    from oxq.selection_lock import final_selection_lock_path, hold_final_selection_lock

    with hold_final_selection_lock(final_selection_lock_path(source_run)):
        worker = threading.Thread(target=read_package)
        worker.start()
        assert transaction_attempted.wait(timeout=5)
        version_manifest.write_text("{not-json", encoding="utf-8")
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert "version_manifest.json must contain a valid JSON object" in str(failures[0])


@pytest.mark.parametrize("corruption", ["zero", "duplicate", "malformed"])
def test_report_qa_requires_exactly_one_valid_source_run_digest(tmp_path, corruption: str) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id=f"digest_{corruption}")
    digest_path = source_run.parent / "run_digests.jsonl"
    original_line = digest_path.read_text(encoding="utf-8").strip()
    if corruption == "zero":
        digest_path.write_text(
            json.dumps({"run_id": "other-run", "artifact_hashes": "sha256:" + "0" * 16}) + "\n",
            encoding="utf-8",
        )
    elif corruption == "duplicate":
        digest_path.write_text(f"{original_line}\n{original_line}\n", encoding="utf-8")
    else:
        digest_path.write_text(
            original_line + "\n" + json.dumps({"run_id": "other-run", "artifact_hashes": 7}) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="run_digests.jsonl"):
        run_report_qa(report_dir)


@pytest.mark.parametrize(
    "artifact_name",
    ["reproducibility_audit.json", "research_bias_audit.json"],
)
def test_report_qa_rejects_unmanifested_source_artifact_before_loading_facts(
    tmp_path,
    artifact_name: str,
) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="unmanifested_source_artifact")
    manifest_path = source_run / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest[artifact_name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_current_run_digest(source_run)

    with pytest.raises(ValueError, match=rf"artifact_hashes.*{re.escape(artifact_name)}"):
        run_report_qa(report_dir)


def test_report_qa_rejects_tampered_monitor_audit_before_loading_facts(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="tampered_monitor_audit")
    audit_path = source_run / "reproducibility_audit.json"
    audit_path.write_text(
        json.dumps({"status": "pass", "fatal_count": 0, "warning_count": 999}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reproducibility_audit.json.*hash mismatch"):
        run_report_qa(report_dir)


def test_report_qa_requires_every_hashed_governed_source_artifact(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="missing_source_artifact")
    (source_run / "trades.csv").unlink()

    with pytest.raises(ValueError, match="missing_files.*trades.csv"):
        run_report_qa(report_dir)


def test_report_qa_rejects_changed_governed_spec_with_same_strategy_id(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="changed_same_id")
    spec_path = source_run / "strategy_spec.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["research"]["hypothesis"] = "materially changed after the governed run"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="spec_hash.txt does not match strategy_spec.yaml"):
        run_report_qa(report_dir)


def test_report_qa_rejects_unparseable_governed_source_spec(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="unparseable_spec")
    (source_run / "strategy_spec.yaml").write_text("research: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="strategy_spec.yaml must contain a valid StrategySpec"):
        run_report_qa(report_dir)


def test_report_qa_requires_governed_source_spec_hash(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="missing_spec_hash")
    (source_run / "spec_hash.txt").unlink()

    with pytest.raises(ValueError, match="spec_hash.txt is required"):
        run_report_qa(report_dir)


@pytest.mark.parametrize(
    "stored_hash",
    ["", "sha256:not-hex", "sha256:0123456789abcdef0123456789abcdef"],
    ids=["empty", "non-hex", "wrong-length"],
)
def test_report_qa_rejects_malformed_governed_source_spec_hash(tmp_path, stored_hash: str) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="malformed_spec_hash")
    (source_run / "spec_hash.txt").write_text(stored_hash + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="spec_hash.txt must contain a canonical StrategySpec hash"):
        run_report_qa(report_dir)


def test_report_qa_rejects_stale_governed_source_spec_hash(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="stale_spec_hash")
    (source_run / "spec_hash.txt").write_text("sha256:0000000000000000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="spec_hash.txt does not match strategy_spec.yaml"):
        run_report_qa(report_dir)


@pytest.mark.parametrize("artifact_name", ["environment.json", "compiled_plan.json"])
def test_report_qa_rejects_governed_run_provenance_spec_hash_mismatch(tmp_path, artifact_name: str) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="stale_run_provenance")
    artifact_path = source_run / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["spec_hash"] = "sha256:0000000000000000"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{artifact_name} spec_hash does not match strategy_spec.yaml"):
        run_report_qa(report_dir)


@pytest.mark.parametrize("artifact_name", ["environment.json", "compiled_plan.json"])
def test_report_qa_requires_governed_run_provenance_spec_hash(tmp_path, artifact_name: str) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="missing_run_provenance_hash")
    artifact_path = source_run / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact.pop("spec_hash")
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{artifact_name} spec_hash is required"):
        run_report_qa(report_dir)


def test_report_qa_rejects_governed_metrics_spec_hash_mismatch_when_present(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="stale_metrics_hash")
    metrics_path = source_run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["spec_hash"] = "sha256:0000000000000000"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="metrics.json spec_hash does not match strategy_spec.yaml"):
        run_report_qa(report_dir)


@pytest.mark.parametrize("artifact_name", ["spec_audit.json", "runtime_audit.json"])
def test_report_qa_rejects_governed_audit_spec_hash_mismatch(tmp_path, artifact_name: str) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="stale_audit_provenance")
    (source_run / artifact_name).write_text(
        json.dumps({"spec_hash": "sha256:0000000000000000"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"{artifact_name} spec_hash does not match strategy_spec.yaml"):
        run_report_qa(report_dir)


def test_report_qa_accepts_matching_copied_report_spec_evidence(tmp_path) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="copied_spec_evidence")
    shutil.copyfile(source_run / "strategy_spec.yaml", report_dir / "strategy_spec.yaml")
    shutil.copyfile(source_run / "spec_hash.txt", report_dir / "spec_hash.txt")

    result = run_report_qa(report_dir)

    assert result.status == "pass"


@pytest.mark.parametrize("evidence_name", ["strategy_spec.yaml", "spec_hash.txt"])
def test_report_qa_rejects_stale_copied_report_spec_evidence(tmp_path, evidence_name: str) -> None:
    source_run, report_dir = _write_governed_qa_package(tmp_path, run_id="stale_copied_evidence")
    shutil.copyfile(source_run / "strategy_spec.yaml", report_dir / "strategy_spec.yaml")
    shutil.copyfile(source_run / "spec_hash.txt", report_dir / "spec_hash.txt")
    if evidence_name == "strategy_spec.yaml":
        copied_spec_path = report_dir / evidence_name
        copied_spec = yaml.safe_load(copied_spec_path.read_text(encoding="utf-8"))
        copied_spec["research"]["hypothesis"] = "stale report copy"
        copied_spec_path.write_text(yaml.safe_dump(copied_spec, sort_keys=False), encoding="utf-8")
    else:
        (report_dir / evidence_name).write_text("sha256:0000000000000000\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"report package {re.escape(evidence_name)}"):
        run_report_qa(report_dir)


def test_report_qa_requires_governed_writer_strategy_id(tmp_path) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_missing_strategy"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    _set_metrics_run_id(governed_source_run, run_id)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    (report_dir / "writer_result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "version_id": "v001",
                "run_id": run_id,
                "source_run_dir": f"versions/v001/09_backtests/{run_id}",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="writer_result.json strategy_id is required"):
        run_report_qa(report_dir)


@pytest.mark.parametrize("strategy_id", [None, "", 123], ids=["null", "empty", "integer"])
def test_report_qa_rejects_malformed_governed_writer_strategy_id(tmp_path, strategy_id) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_malformed_strategy"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    _set_metrics_run_id(governed_source_run, run_id)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id=strategy_id,
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )

    with pytest.raises(ValueError, match="writer_result.json strategy_id must be a non-empty string"):
        run_report_qa(report_dir)


def test_report_qa_rejects_governed_writer_strategy_id_mismatch(tmp_path) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_wrong_strategy"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    _set_metrics_run_id(governed_source_run, run_id)
    metrics_path = governed_source_run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["strategy_id"] = "wrong_strategy"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="wrong_strategy",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )

    with pytest.raises(ValueError, match="writer_result.json strategy_id does not match strategy_spec.yaml"):
        run_report_qa(report_dir)


def test_report_qa_requires_governed_metrics_strategy_id(tmp_path) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_missing_metrics_strategy"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    _set_metrics_run_id(governed_source_run, run_id)
    metrics_path = governed_source_run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.pop("strategy_id")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )

    with pytest.raises(ValueError, match="metrics.json strategy_id is required"):
        run_report_qa(report_dir)


@pytest.mark.parametrize("strategy_id", [None, "", 123], ids=["null", "empty", "integer"])
def test_report_qa_rejects_malformed_governed_metrics_strategy_id(tmp_path, strategy_id) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_malformed_metrics_strategy"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    _set_metrics_run_id(governed_source_run, run_id)
    metrics_path = governed_source_run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["strategy_id"] = strategy_id
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )

    with pytest.raises(ValueError, match="metrics.json strategy_id must be a non-empty string"):
        run_report_qa(report_dir)


def test_report_qa_rejects_governed_metrics_strategy_id_mismatch(tmp_path) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_wrong_metrics_strategy"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    _set_metrics_run_id(governed_source_run, run_id)
    metrics_path = governed_source_run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["strategy_id"] = "wrong_strategy"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )

    with pytest.raises(ValueError, match="metrics.json strategy_id does not match strategy_spec.yaml"):
        run_report_qa(report_dir)


def test_report_qa_rejects_governed_metrics_run_id_mismatch(tmp_path) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_qa_case"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )

    with pytest.raises(ValueError, match="metrics.json run_id must match the resolved run directory name"):
        run_report_qa(report_dir)


def test_report_qa_requires_writer_result_for_governed_package(tmp_path) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_missing_writer"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    _set_metrics_run_id(governed_source_run, run_id)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="writer_result.json is required"):
        run_report_qa(report_dir)


def test_report_qa_rejects_malformed_governed_writer_result(tmp_path) -> None:
    report_dir = tmp_path / "versions/v001/10_reports/run_1"
    report_dir.mkdir(parents=True)
    (report_dir / "writer_result.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="writer_result.json must contain a valid JSON object"):
        run_report_qa(report_dir)


@pytest.mark.parametrize(
    "writer_status",
    ["blocked", "fail", "unknown", None],
    ids=["blocked", "fail", "unknown", "missing"],
)
def test_report_qa_requires_successful_writer_result(tmp_path, writer_status) -> None:
    source_run = _write_qa_run(tmp_path)
    run_id = "20240101_000000_writer_status"
    version_dir = tmp_path / "versions" / "v001"
    governed_source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source_run, governed_source_run)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    writer_result = {
        "version_id": "v001",
        "run_id": run_id,
        "strategy_id": "qa_case",
        "source_run_dir": f"versions/v001/09_backtests/{run_id}",
    }
    if writer_status is not None:
        writer_result["status"] = writer_status
    (report_dir / "writer_result.json").write_text(json.dumps(writer_result), encoding="utf-8")
    markdown = "# 研究报告\n\n有效数据最后交易日：2024-03-29\n\n配置结束日：2024-03-31\n"
    (report_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    with pytest.raises(ValueError, match="writer_result.json status"):
        run_report_qa(report_dir)


@pytest.mark.parametrize("missing_field", ["version_id", "run_id", "strategy_id", "source_run_dir"])
def test_report_qa_requires_complete_governed_writer_identity(tmp_path, missing_field: str) -> None:
    report_dir = tmp_path / "versions/v001/10_reports/run_1"
    report_dir.mkdir(parents=True)
    writer_result = {
        "status": "pass",
        "version_id": "v001",
        "run_id": "run_1",
        "strategy_id": "qa_case",
        "source_run_dir": "versions/v001/09_backtests/run_1",
    }
    writer_result.pop(missing_field)
    (report_dir / "writer_result.json").write_text(json.dumps(writer_result), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"writer_result.json {missing_field} is required"):
        run_report_qa(report_dir)


def test_report_qa_requires_writer_result_with_explicit_manifest_source(tmp_path) -> None:
    version_dir = tmp_path / "versions/v001"
    run_id = "run_1"
    source_run = version_dir / "09_backtests" / run_id
    source_run.mkdir(parents=True)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "phase_paths": {
                    "09_backtests": "versions/v001/09_backtests",
                    "10_reports": "versions/v001/10_reports",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="writer_result.json is required"):
        run_report_qa(report_dir, source_run_dir=source_run)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version_id", "v002", "version_id does not match"),
        ("run_id", "run_2", "run_id does not match"),
    ],
)
def test_report_qa_rejects_mismatched_governed_writer_identity(
    tmp_path,
    field: str,
    value: str,
    message: str,
) -> None:
    report_dir = tmp_path / "versions/v001/10_reports/run_1"
    report_dir.mkdir(parents=True)
    writer_result = {
        "status": "pass",
        "version_id": "v001",
        "run_id": "run_1",
        "strategy_id": "qa_case",
        "source_run_dir": "versions/v001/09_backtests/run_1",
    }
    writer_result[field] = value
    (report_dir / "writer_result.json").write_text(json.dumps(writer_result), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_report_qa(report_dir)


@pytest.mark.parametrize(
    "source_reference",
    ["../outside/run_1", "versions/v001/09_backtests/different_run"],
)
def test_report_qa_rejects_unsafe_or_mismatched_canonical_writer_source(
    tmp_path,
    source_reference: str,
) -> None:
    run_id = "run_1"
    version_dir = tmp_path / "versions/v001"
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    (report_dir / "writer_result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "version_id": "v001",
                "run_id": run_id,
                "strategy_id": "qa_case",
                "source_run_dir": source_reference,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_run_dir"):
        run_report_qa(report_dir)


def test_report_qa_rejects_mismatched_explicit_canonical_source(tmp_path) -> None:
    run_id = "run_1"
    version_dir = tmp_path / "versions/v001"
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )
    outside_source = tmp_path / "outside" / run_id
    outside_source.mkdir(parents=True)

    with pytest.raises(ValueError, match="source_run_dir"):
        run_report_qa(report_dir, source_run_dir=outside_source)


def test_report_qa_rejects_missing_canonical_writer_source(tmp_path) -> None:
    run_id = "run_1"
    report_dir = tmp_path / "versions/v001/10_reports" / run_id
    report_dir.mkdir(parents=True)
    (report_dir / "writer_result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "version_id": "v001",
                "run_id": run_id,
                "strategy_id": "qa_case",
                "source_run_dir": f"versions/v001/09_backtests/{run_id}",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_run_dir does not exist"):
        run_report_qa(report_dir)


def test_report_qa_accepts_manifest_defined_report_and_backtest_paths(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    source = _write_qa_run(fixture)
    workspace = tmp_path / "workspace"
    version_dir = workspace / "research_versions/v003"
    run_id = "20240101_000000_custom_paths"
    source_run = version_dir / "artifacts/backtests" / run_id
    report_dir = version_dir / "artifacts/reports" / run_id
    shutil.copytree(source, source_run)
    _set_metrics_run_id(source_run, run_id)
    report_dir.mkdir(parents=True)
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v003",
                "phase_paths": {
                    "09_backtests": "research_versions/v003/artifacts/backtests",
                    "10_reports": "research_versions/v003/artifacts/reports",
                },
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "writer_result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "version_id": "v003",
                "run_id": run_id,
                "strategy_id": "qa_case",
                "source_run_dir": f"research_versions/v003/artifacts/backtests/{run_id}",
            }
        ),
        encoding="utf-8",
    )
    markdown = "# 研究报告\n\n有效数据最后交易日：2024-03-29\n\n配置结束日：2024-03-31\n"
    (report_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "research_report.html").write_text(
        render_markdown_html_report(markdown),
        encoding="utf-8",
    )

    result = run_report_qa(report_dir)

    assert result.status == "pass"
    assert result.facts.configured_end_date == "2024-03-31"
    assert result.facts.effective_last_trading_day == "2024-03-29"


@pytest.mark.parametrize(
    "source_reference",
    [
        "../outside/run_1",
        "research_versions/v003/artifacts/backtests/different_run",
    ],
)
def test_report_qa_rejects_unsafe_or_mismatched_writer_source(
    tmp_path,
    source_reference: str,
) -> None:
    workspace = tmp_path / "workspace"
    version_dir = workspace / "research_versions/v003"
    run_id = "run_1"
    report_dir = version_dir / "artifacts/reports" / run_id
    report_dir.mkdir(parents=True)
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v003",
                "phase_paths": {
                    "09_backtests": "research_versions/v003/artifacts/backtests",
                    "10_reports": "research_versions/v003/artifacts/reports",
                },
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "writer_result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "version_id": "v003",
                "run_id": run_id,
                "strategy_id": "qa_case",
                "source_run_dir": source_reference,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_run_dir"):
        run_report_qa(report_dir)


def test_report_qa_accepts_explicit_manifest_source_path(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    source = _write_qa_run(fixture)
    workspace = tmp_path / "workspace"
    version_dir = workspace / "research_versions/v003"
    run_id = "run_explicit"
    source_run = version_dir / "artifacts/backtests" / run_id
    report_dir = version_dir / "artifacts/reports" / run_id
    shutil.copytree(source, source_run)
    _set_metrics_run_id(source_run, run_id)
    report_dir.mkdir(parents=True)
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v003",
                "phase_paths": {
                    "09_backtests": "research_versions/v003/artifacts/backtests",
                    "10_reports": "research_versions/v003/artifacts/reports",
                },
            }
        ),
        encoding="utf-8",
    )
    _write_writer_result(
        report_dir,
        version_id="v003",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"research_versions/v003/artifacts/backtests/{run_id}",
    )
    markdown = "# 研究报告\n\n有效数据最后交易日：2024-03-29\n\n配置结束日：2024-03-31\n"
    (report_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "research_report.html").write_text(
        render_markdown_html_report(markdown),
        encoding="utf-8",
    )

    result = run_report_qa(report_dir, source_run_dir=source_run)

    assert result.status == "pass"


def test_report_qa_flags_report_image_manifest_hash_and_number_problems(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(run_dir, figure, asset_id="equity", title="Equity", section="results", order=10)
    (run_dir / "report_assets/figures/equity.png").write_bytes(b"changed")
    markdown = (
        "# Report\n\n"
        "有效数据最后交易日：2024-03-29\n\n"
        "配置结束日：2024-03-31\n\n"
        "总收益为 99.00%。\n\n"
        "![Unregistered](report_assets/figures/unregistered.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(
        '<!doctype html><html><body><img src="../outside.png"></body></html>',
        encoding="utf-8",
    )

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    finding_ids = {finding.id for finding in result.findings}
    assert "asset_hash_mismatch" in finding_ids
    assert "markdown_image_unregistered" in finding_ids
    assert "html_image_path" in finding_ids
    assert "numeric_claim_unverified" in finding_ids


def test_report_qa_does_not_validate_chart_text_rendering(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    script = run_dir / "report_assets/scripts/plot.py"
    script.parent.mkdir(parents=True)
    script.write_text('import matplotlib.pyplot as plt\nplt.title("策略净值")\n', encoding="utf-8")
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(
        run_dir,
        figure,
        asset_id="equity",
        title="策略净值",
        caption="由 equity_curve.csv 生成。",
        source_script=script,
    )
    markdown = (
        "# 研究报告\n\n"
        "有效数据最后交易日：2024-03-29\n\n"
        "配置结束日：2024-03-31\n\n"
        "![策略净值](report_assets/figures/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"
    assert result.warning_count == 0


def test_report_qa_flags_missing_html_date_disclosures(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = "# 研究报告\n\n有效数据最后交易日：2024-03-29\n\n配置结束日：2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text("<!doctype html><html><body><h1>研究报告</h1></body></html>", encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    finding_ids = {finding.id for finding in result.findings}
    assert "html_effective_last_trading_day_missing" in finding_ids
    assert "html_configured_end_date_missing" in finding_ids


def test_report_qa_allows_table_date_disclosures(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "| Field | Value |\n"
        "| --- | --- |\n"
        "| Effective last trading day | 2024-03-29 |\n"
        "| Configured end date | 2024-03-31 |\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_flags_same_count_different_html_image_sources(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    equity = tmp_path / "equity.png"
    drawdown = tmp_path / "drawdown.png"
    _write_png(equity)
    _write_png(drawdown)
    add_report_asset(run_dir, equity, asset_id="equity", title="Equity", section="results", order=10)
    add_report_asset(run_dir, drawdown, asset_id="drawdown", title="Drawdown", section="results", order=20)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Equity](report_assets/figures/equity.png)\n"
    )
    html = '<!doctype html><html><body><p>2024-03-29 2024-03-31</p><img src="report_assets/figures/drawdown.png"></body></html>'
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(html, encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "image_source_mismatch" for finding in result.findings)


def test_report_qa_rejects_embedded_attachment_images(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    attachment = tmp_path / "notes.pdf"
    attachment.write_bytes(b"%PDF-1.4")
    add_report_asset(run_dir, attachment, asset_id="notes", title="Notes", section="appendix", order=10)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Notes](report_assets/attachments/notes.pdf)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "embedded_image_not_figure" for finding in result.findings)


def test_report_qa_requires_available_date_facts(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    (run_dir / "equity_curve.csv").unlink()
    (run_dir / "research_report.md").write_text("# Report\n\nConfigured end date: 2024-03-31\n", encoding="utf-8")
    (run_dir / "research_report.html").write_text(
        "<!doctype html><html><body>Configured end date: 2024-03-31</body></html>",
        encoding="utf-8",
    )

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "effective_last_trading_day_unavailable" for finding in result.findings)


def test_report_qa_flags_non_percent_numeric_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The report claims 99 OOS trades, 10 positive months, and Sharpe 9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "99" in messages
    assert "10" in messages
    assert "9.99" in messages


def test_report_qa_defaults_to_deterministic_checks(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The report claims 99 OOS trades and Sharpe 9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = _run_report_qa(run_dir)

    assert result.status == "pass"
    assert not any(finding.id == "numeric_claim_unverified" for finding in result.findings)


def test_report_qa_does_not_match_percent_claims_against_counts(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "trade_count": 2, "oos_trade_count": 1})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The invented total return was 200.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "200.00%" in finding.message for finding in result.findings)


def test_report_qa_matches_percent_claims_to_metric_context(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "max_drawdown": -0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The max drawdown was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_treats_unscoped_drawdown_as_overall(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"max_drawdown": -0.05, "oos_max_drawdown": -0.1})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Max drawdown was 10.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "10.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_annualized_return_context_exclusive(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"annualized_return": 0.1, "total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Annualized return was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_excess_return_context_exclusive(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "excess_total_return": 0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Excess return was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_total_return_context_exclusive(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "benchmark_total_return": 0.07, "excess_total_return": 0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 5.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "5.00%" in finding.message for finding in result.findings)


def test_report_qa_treats_unscoped_total_return_as_overall(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.1, "oos_total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_binds_strategy_and_benchmark_returns_per_claim(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "benchmark_total_return": 0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Strategy total return 5.00%; benchmark total return 5.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "5.00%" in finding.message for finding in result.findings)


def test_report_qa_binds_generic_strategy_return_to_strategy_total_return(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "benchmark_total_return": 0.05, "excess_total_return": 0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The strategy returned 5.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "5.00%" in finding.message for finding in result.findings)


def test_report_qa_parses_signed_positive_percentage_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was +99.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "+99.00%" in finding.message for finding in result.findings)


def test_report_qa_excludes_benchmark_window_strategy_return_from_generic_total(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "benchmark_curve.csv").write_text(
        "date,value\n"
        "2024-01-31,100\n"
        "2024-03-29,100\n",
        encoding="utf-8",
    )
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 9.09%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "9.09%" in finding.message for finding in result.findings)


def test_report_qa_matches_monthly_return_claims_to_mentioned_month(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "February return was 10.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "10.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_cost_rate_claims_field_specific(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    spec["cost"] = {"fee_rate": 0.001, "slippage_rate": 0.0005}
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Fee: 0.050%, Slippage: 0.100%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "0.050%" in messages
    assert "0.100%" in messages


def test_report_qa_excludes_fee_min_from_fee_rate_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    spec["cost"] = {"fee_rate": 0.001, "fee_min": 0.0}
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Fee: 0.000%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "0.000%" in finding.message for finding in result.findings)


def test_report_qa_respects_oos_scope_for_percent_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "oos_total_return": -0.1})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The OOS total return was 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_keeps_is_oos_percent_claims_scoped_on_mixed_lines(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"is_total_return": 0.1, "oos_total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "IS total return 20.00%, OOS total return 10.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "20.00%" in messages
    assert "10.00%" in messages


def test_report_qa_does_not_scope_prior_total_return_to_later_oos_label(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.1, "oos_total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return 10.00%, OOS total return 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_binds_scope_markers_after_percent_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "oos_total_return": -0.1})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "20.00% OOS total return.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_restricts_prior_unscoped_return_on_mixed_scope_line(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.1, "oos_total_return": 0.2})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return 20.00%, OOS total return 20.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "20.00%" in finding.message for finding in result.findings)


def test_report_qa_checks_numbers_inside_ordered_list_items(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "1. Sharpe ratio was 9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "9.99" in finding.message for finding in result.findings)


def test_report_qa_skips_generated_timestamp_clock_components(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "**Generated**: 2024-03-31 12:34:56 UTC\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_accepts_registered_webp_dimensions(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    figure = tmp_path / "equity.webp"
    _write_webp_vp8x(figure, width=2, height=3)
    add_report_asset(run_dir, figure, asset_id="equity", title="Equity", section="results", order=10)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Equity](report_assets/figures/equity.webp)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"
    assert not any(finding.id == "image_dimensions_unreadable" for finding in result.findings)


def test_report_qa_requires_date_labels_when_required_dates_are_equal(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    spec["validation"]["test_period"][1] = "2024-03-29"
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    finding_ids = {finding.id for finding in result.findings}
    assert result.status == "fail"
    assert "markdown_configured_end_date_missing" in finding_ids
    assert "html_configured_end_date_missing" in finding_ids


def test_report_qa_allows_strategy_spec_cost_and_cash_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = StrategySpec.from_yaml(str(run_dir / "strategy_spec.yaml"))
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.0005
    spec.execution.initial_cash = 100000
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Fee: 0.100%.\n\n"
        "Slippage: 0.050%.\n\n"
        "Initial Cash: $100,000.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_checks_numeric_claims_in_html_text(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The total return was 20.00%.\n"
    )
    html = (
        "<!doctype html><html><body>"
        "<p>Effective last trading day: 2024-03-29</p>"
        "<p>Configured end date: 2024-03-31</p>"
        "<p>The total return was 99.00%.</p>"
        "</body></html>"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(html, encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(
        finding.id == "numeric_claim_unverified" and "HTML" in finding.message and "99.00%" in finding.message
        for finding in result.findings
    )


def test_report_qa_preserves_html_table_row_context_for_numbers(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"total_return": 0.2, "max_drawdown": -0.05})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The total return was 20.00%.\n"
    )
    html = (
        "<!doctype html><html><body>"
        "<p>Effective last trading day: 2024-03-29</p>"
        "<p>Configured end date: 2024-03-31</p>"
        "<table><tr><td>Max drawdown</td><td>20.00%</td></tr></table>"
        "</body></html>"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(html, encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(
        finding.id == "numeric_claim_unverified" and "HTML" in finding.message and "20.00%" in finding.message
        for finding in result.findings
    )


def test_report_qa_rejects_figure_kind_outside_figures_dir(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    asset_path = run_dir / "report_assets/attachments/equity.png"
    asset_path.parent.mkdir(parents=True)
    _write_png(asset_path)
    _write_manifest(
        run_dir,
        [
            {
                "id": "equity",
                "kind": "figure",
                "path": "attachments/equity.png",
                "title": "Equity",
                "caption": "",
                "section": "results",
                "order": 10,
                "mime_type": "image/png",
                "sha256": "",
            }
        ],
    )
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "![Equity](report_assets/attachments/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "asset_kind_path_mismatch" for finding in result.findings)


def test_report_qa_requires_manifest_asset_hash(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    asset_path = run_dir / "report_assets/attachments/notes.txt"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text("notes", encoding="utf-8")
    _write_manifest(
        run_dir,
        [
            {
                "id": "notes",
                "kind": "attachment",
                "path": "attachments/notes.txt",
                "title": "Notes",
                "caption": "",
                "section": "appendix",
                "order": 10,
                "mime_type": "text/plain",
                "sha256": "",
            }
        ],
    )
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "asset_hash_missing" for finding in result.findings)


def test_report_qa_rejects_directory_manifest_asset_paths(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    (run_dir / "report_assets/attachments").mkdir(parents=True)
    _write_manifest(
        run_dir,
        [
            {
                "id": "notes",
                "kind": "attachment",
                "path": "attachments",
                "title": "Notes",
                "caption": "",
                "section": "appendix",
                "order": 10,
                "mime_type": "text/plain",
                "sha256": "sha256:unused",
            }
        ],
    )
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "asset_file_not_regular" for finding in result.findings)


def test_report_qa_rejects_non_object_manifest_entries(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    _write_manifest(run_dir, ["bad"])
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "manifest_asset_invalid" for finding in result.findings)


def test_report_qa_fails_when_metrics_are_missing(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    (run_dir / "metrics.json").unlink()
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "metrics_unreadable" for finding in result.findings)


def test_report_qa_allows_total_and_oos_trade_counts_on_same_line(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The run had 2 total trades and 1 OOS trade.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_keeps_total_and_oos_trade_counts_claim_specific(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The run had 1 total trades and 2 OOS trades.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "1" in messages
    assert "2" in messages


def test_report_qa_keeps_positive_and_negative_month_counts_claim_specific(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Positive months 1, negative months 2.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "1" in messages
    assert "2" in messages


def test_report_qa_does_not_validate_generic_trade_claims_with_oos_count(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The run had 1 trade overall.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "1" in finding.message for finding in result.findings)


def test_report_qa_allows_rounded_plain_number_ratio_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 1.234})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Sharpe Ratio | 1.23\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_parses_signed_positive_plain_number_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 1.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Sharpe +9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "+9.99" in finding.message for finding in result.findings)


def test_report_qa_checks_leading_decimal_metric_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 1.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "9.99 Sharpe ratio.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "9.99" in finding.message for finding in result.findings)


def test_report_qa_treats_unscoped_ratio_claims_as_overall(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 0.5, "oos_sharpe_ratio": 2.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Sharpe 2.00.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "2.00" in finding.message for finding in result.findings)


def test_report_qa_matches_win_rate_claims_only_to_win_rate_metrics(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    spec = yaml.safe_load((run_dir / "strategy_spec.yaml").read_text(encoding="utf-8"))
    spec["cost"] = {"fee_rate": 0.001, "slippage_rate": 0.0005}
    (run_dir / "strategy_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Win rate was 0.10%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "0.10%" in finding.message for finding in result.findings)


def test_report_qa_binds_same_line_ratio_claims_to_labels(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"sharpe_ratio": 1.0, "calmar_ratio": 2.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Sharpe 2.00, Calmar 1.00.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    messages = "\n".join(finding.message for finding in result.findings if finding.id == "numeric_claim_unverified")
    assert "2.00" in messages
    assert "1.00" in messages


def test_report_qa_keeps_trade_counts_out_of_ratio_claims(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"trade_count": 2, "sharpe_ratio": 1.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "The run had 2 total trades and Sharpe 2.00.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "2.00" in finding.message for finding in result.findings)


def test_report_qa_checks_plain_numbers_inside_figure_captions(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"oos_sharpe_ratio": 1.0})
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Figure 1. OOS Sharpe 9.99.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "warn"
    assert any(finding.id == "numeric_claim_unverified" and "9.99" in finding.message for finding in result.findings)


def test_report_qa_skips_numbered_markdown_headings(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    markdown = (
        "# Report\n\n"
        "## 7. Executive Decision\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_parses_comma_formatted_percentage_claims_as_single_value(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_return"] = 10.0
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    markdown = (
        "# Report\n\n"
        "Effective last trading day: 2024-03-29\n\n"
        "Configured end date: 2024-03-31\n\n"
        "Total return was 1,000.00%.\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown, lang="en"), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "pass"


def test_report_qa_reports_unsafe_source_script_path(tmp_path) -> None:
    run_dir = _write_qa_run(tmp_path)
    figure = tmp_path / "equity.png"
    _write_png(figure)
    add_report_asset(run_dir, figure, asset_id="equity", title="策略净值")
    manifest = json.loads((run_dir / "report_assets/manifest.json").read_text(encoding="utf-8"))
    manifest["assets"][0]["source"] = {"script": "../plot.py"}
    (run_dir / "report_assets/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    markdown = (
        "# 研究报告\n\n"
        "有效数据最后交易日：2024-03-29\n\n"
        "配置结束日：2024-03-31\n\n"
        "![策略净值](report_assets/figures/equity.png)\n"
    )
    (run_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (run_dir / "research_report.html").write_text(render_markdown_html_report(markdown), encoding="utf-8")

    result = run_report_qa(run_dir)

    assert result.status == "fail"
    assert any(finding.id == "source_script_path_invalid" for finding in result.findings)


def _write_qa_run(tmp_path):
    spec = StrategySpec.template(strategy_id="qa_case", hypothesis="qa should validate final reports")
    spec.validation.train_period = ["2024-01-02", "2024-01-31"]
    spec.validation.test_period = ["2024-02-01", "2024-03-31"]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    spec_hash = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml").compute_hash()
    (run_dir / "spec_hash.txt").write_text(spec_hash + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({"spec_hash": spec_hash}), encoding="utf-8")
    (run_dir / "compiled_plan.json").write_text(json.dumps({"spec_hash": spec_hash}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": "qa-run", "trade_count": 2, "oos_trade_count": 1, "total_return": 0.2}),
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
    return run_dir


def _write_governed_qa_package(tmp_path, *, run_id: str):
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    source = _write_qa_run(fixture_dir)
    version_dir = tmp_path / "versions" / "v001"
    source_run = version_dir / "09_backtests" / run_id
    shutil.copytree(source, source_run)
    _set_metrics_run_id(source_run, run_id)
    report_dir = version_dir / "10_reports" / run_id
    report_dir.mkdir(parents=True)
    config_dir = tmp_path / ".open-xquant"
    config_dir.mkdir()
    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {"versions_dir": "versions"},
                "workflow": {"layout": "version_governed"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "current.json").write_text(json.dumps({"active_version": "v001"}), encoding="utf-8")
    (version_dir / "version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "phase_paths": {
                    "09_backtests": "versions/v001/09_backtests",
                    "10_reports": "versions/v001/10_reports",
                },
            }
        ),
        encoding="utf-8",
    )
    _write_writer_result(
        report_dir,
        version_id="v001",
        run_id=run_id,
        strategy_id="qa_case",
        source_run_dir=f"versions/v001/09_backtests/{run_id}",
    )
    markdown = "# Report\n\nEffective last trading day: 2024-03-29\n\nConfigured end date: 2024-03-31\n"
    (report_dir / "research_report.md").write_text(markdown, encoding="utf-8")
    (report_dir / "research_report.html").write_text(
        render_markdown_html_report(markdown, lang="en"),
        encoding="utf-8",
    )
    return source_run, report_dir


def _set_metrics_run_id(run_dir, run_id: str) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["run_id"] = run_id
    metrics["strategy_id"] = "qa_case"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _write_source_integrity(run_dir)


def _write_source_integrity(run_dir) -> None:
    (run_dir / "data_manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "reproducibility_audit.json").write_text(
        json.dumps({"status": "pass", "fatal_count": 0, "warning_count": 0}),
        encoding="utf-8",
    )
    (run_dir / "research_bias_audit.json").write_text(
        json.dumps({"status": "pass", "fatal_count": 0, "warning_count": 0}),
        encoding="utf-8",
    )
    artifact_hashes = {
        "data_manifest.json": _canonical_json_hash(run_dir / "data_manifest.json"),
        "equity_curve.csv": _short_file_hash(run_dir / "equity_curve.csv"),
        "trades.csv": _short_file_hash(run_dir / "trades.csv"),
        "metrics.json": _canonical_json_hash(run_dir / "metrics.json", exclude_keys={"run_id"}),
        "reproducibility_audit.json": _short_file_hash(run_dir / "reproducibility_audit.json"),
        "research_bias_audit.json": _short_file_hash(run_dir / "research_bias_audit.json"),
    }
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_current_run_digest(run_dir)


def _write_current_run_digest(run_dir) -> None:
    digest_path = run_dir.parent / "run_digests.jsonl"
    entry = {
        "run_id": run_dir.name,
        "artifact_hashes": _canonical_json_hash(run_dir / "artifact_hashes.json"),
    }
    digest_path.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json_hash(path, *, exclude_keys: set[str] | None = None) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and exclude_keys:
        payload = {key: value for key, value in payload.items() if key not in exclude_keys}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _short_file_hash(path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def _write_writer_result(
    report_dir,
    *,
    version_id: str,
    run_id: str,
    strategy_id,
    source_run_dir: str,
) -> None:
    (report_dir / "writer_result.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "version_id": version_id,
                "run_id": run_id,
                "strategy_id": strategy_id,
                "source_run_dir": source_run_dir,
            }
        ),
        encoding="utf-8",
    )


def _write_png(path) -> None:
    path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR4nGNgYGAAAAAEAAHIiY1AAAAAAElFTkSuQmCC"))


def _write_webp_vp8x(path, *, width: int, height: int) -> None:
    payload = b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    chunk = b"VP8X" + len(payload).to_bytes(4, "little") + payload
    data = b"WEBP" + chunk
    path.write_bytes(b"RIFF" + len(data).to_bytes(4, "little") + data)


def _write_manifest(run_dir, assets: list[dict]) -> None:
    path = run_dir / "report_assets/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "assets": assets}), encoding="utf-8")
