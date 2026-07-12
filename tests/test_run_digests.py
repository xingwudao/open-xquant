from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from oxq import run_digests
from oxq.audit import reproducibility as reproducibility_module
from oxq.cli import main as main_module
from oxq.report import qa as report_qa_module
from oxq.run_digests import (
    RunDigestError,
    publish_run_artifacts,
    replace_run_digest_entry,
    require_current_run_digest,
    run_digest_transaction,
    update_artifact_hashes_and_run_digest,
)


def _case_insensitive_alias(path: Path) -> Path:
    alias = path.with_name(path.name.swapcase())
    try:
        aliases_same_entry = alias.exists() and os.path.samefile(path, alias)
    except OSError:
        aliases_same_entry = False
    if not aliases_same_entry:
        pytest.skip("filesystem does not expose case-insensitive aliases")
    return alias


def _write_current_run(run_dir, artifacts: dict[str, bytes]) -> None:
    run_dir.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for name, content in artifacts.items():
        artifact_path = run_dir / name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)
        hashes[name] = "sha256:" + hashlib.sha256(content).hexdigest()[:16]
    (run_dir / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    update_artifact_hashes_and_run_digest(run_dir, lambda manifest: None)


def _write_metrics_manifest(run_dir) -> None:
    metrics = b"{}\n"
    (run_dir / "metrics.json").write_bytes(metrics)
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps({"metrics.json": run_digests._hash_json_payload({})}),
        encoding="utf-8",
    )


_SCHEMA_5_REQUIRED_ARTIFACTS = {
    "strategy_spec.yaml",
    "environment.json",
    "data_manifest.json",
    "execution_assumptions.json",
    "compiled_plan.json",
    "strategy.py",
    "equity_curve.csv",
    "trades.csv",
    "positions.csv",
    "orders.csv",
    "target_weights.csv",
    "metrics.json",
}


def _write_versioned_run(
    run_dir: Path,
    *,
    schema_version: int = 5,
    artifact_names: set[str] | None = None,
    data_manifest_schema_version: int | None = 1,
    write_digest: bool = True,
) -> dict[str, object]:
    run_dir.mkdir(parents=True)
    names = artifact_names or set(_SCHEMA_5_REQUIRED_ARTIFACTS)
    manifest: dict[str, object] = {"schema_version": schema_version}
    for name in names:
        if name == "data_manifest.json":
            payload = {} if data_manifest_schema_version is None else {"schema_version": data_manifest_schema_version}
            content = (json.dumps(payload) + "\n").encode()
        elif name in {"environment.json", "execution_assumptions.json", "compiled_plan.json", "metrics.json"}:
            content = b"{}\n"
        else:
            content = f"{name}\n".encode()
        (run_dir / name).write_bytes(content)
        manifest[name] = run_digests._hash_bound_artifact(name, content)
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    if write_digest:
        replace_run_digest_entry(run_dir, run_digests._hash_json_payload(manifest))
    return manifest


def _rewrite_manifest_and_digest(run_dir: Path, manifest: dict[str, object]) -> None:
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    digest_path = run_dir.parent / "run_digests.jsonl"
    entries = [json.loads(line) for line in digest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [entry for entry in entries if entry.get("run_id") == run_dir.name]
    assert len(matches) == 1
    matches[0]["artifact_hashes"] = run_digests._hash_json_payload(manifest)
    digest_path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")


def _write_recovery_journal(
    parent: Path,
    *,
    run_id: str,
    recovery: str,
    old: bytes,
    new: bytes,
) -> Path:
    journal_path = parent / "run_digests.jsonl.journal"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recovery": recovery,
                "run_id": run_id,
                "targets": [
                    {
                        "kind": "artifact",
                        "name": "outside-sentinel.txt",
                        "old": run_digests._encode_content(old),
                        "new": run_digests._encode_content(new),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return journal_path


def _write_component_manifest_fixture(path: Path, extension_id: str) -> str:
    from oxq.core.component_manifest import compute_component_bundle_hash

    path.parent.mkdir(parents=True, exist_ok=True)
    extension_root = path.parent / f"{extension_id}_root"
    extension_root.mkdir()
    payload = {
        "schema_version": 1,
        "extension_id": extension_id,
        "extension_root": extension_root.name,
        "components": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    bundle_hash = compute_component_bundle_hash(path)
    payload["bundle_hash"] = bundle_hash
    path.write_text(json.dumps(payload), encoding="utf-8")
    return bundle_hash


def _bind_component_file(run_dir: Path, manifest: dict[str, object], name: str, content: bytes) -> None:
    (run_dir / name).write_bytes(content)
    manifest[name] = run_digests._hash_bound_artifact(name, content)


def test_require_current_run_digest_rejects_recomputed_manifest_missing_schema_5_order_binding(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    manifest.pop("orders.csv")
    _rewrite_manifest_and_digest(run_dir, manifest)

    with pytest.raises(RunDigestError, match=r"artifact_hashes_v5.*missing required bindings.*orders\.csv"):
        require_current_run_digest(run_dir)


def test_require_current_run_digest_rejects_inventory_profile_downgrade(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    manifest["schema_version"] = 4
    manifest.pop("strategy.py")
    _rewrite_manifest_and_digest(run_dir, manifest)

    with pytest.raises(RunDigestError, match=r"inventory profile mismatch.*artifact_hashes_v5.*artifact_hashes_v4"):
        require_current_run_digest(run_dir)


def test_inventory_downgrade_update_rolls_back_when_digest_row_retains_binding(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)
    manifest_path = run_dir / "artifact_hashes.json"
    digest_path = run_dir.parent / "run_digests.jsonl"
    original_manifest = manifest_path.read_bytes()
    original_digest = digest_path.read_bytes()

    def drop_inventory_markers(manifest: dict[str, object]) -> None:
        manifest.pop("schema_version")
        manifest.pop("data_manifest.json")

    with pytest.raises(RunDigestError, match="legacy artifact inventory"):
        update_artifact_hashes_and_run_digest(run_dir, drop_inventory_markers)

    assert manifest_path.read_bytes() == original_manifest
    assert digest_path.read_bytes() == original_digest
    require_current_run_digest(run_dir)


def test_run_digest_entry_records_versioned_inventory_profile(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)

    entry = json.loads((run_dir.parent / "run_digests.jsonl").read_text(encoding="utf-8"))

    assert entry["artifact_inventory"] == {
        "schema_version": 1,
        "profile": "artifact_hashes_v5",
    }


def test_validate_run_artifact_inventory_rejects_unregistered_binding(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    extra = run_dir / "notes.txt"
    extra.write_text("not governed\n", encoding="utf-8")
    manifest[extra.name] = run_digests._hash_bytes(extra.read_bytes())
    _rewrite_manifest_and_digest(run_dir, manifest)

    with pytest.raises(RunDigestError, match=r"unregistered artifact bindings.*notes\.txt"):
        run_digests.validate_run_artifact_inventory(run_dir)


def test_validate_run_artifact_inventory_rejects_noncanonical_alias(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    manifest["./orders.csv"] = manifest.pop("orders.csv")
    _rewrite_manifest_and_digest(run_dir, manifest)

    with pytest.raises(RunDigestError, match=r"non-canonical artifact path.*\./orders\.csv"):
        run_digests.validate_run_artifact_inventory(run_dir)


@pytest.mark.parametrize(
    "artifact_name",
    [r"..\outside.txt", r"C:\portable.txt", r"nested\asset.txt"],
)
def test_validate_run_artifact_inventory_rejects_windows_path_spellings(
    tmp_path,
    artifact_name: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    manifest[artifact_name] = manifest.pop("orders.csv")
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunDigestError, match="non-canonical artifact path"):
        run_digests.validate_run_artifact_inventory(run_dir)


def test_validate_run_artifact_inventory_rejects_duplicate_json_binding_key(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir, write_digest=False)
    duplicate_value = json.dumps(manifest["orders.csv"])
    manifest_text = json.dumps(manifest)
    (run_dir / "artifact_hashes.json").write_text(
        f'{manifest_text[:-1]}, "orders.csv": {duplicate_value}}}',
        encoding="utf-8",
    )

    with pytest.raises(RunDigestError, match=r"duplicate JSON key.*orders\.csv"):
        run_digests.validate_run_artifact_inventory(run_dir)


def test_validate_run_artifact_inventory_allows_optional_monitor_binding(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)

    profile = run_digests.validate_run_artifact_inventory(run_dir)

    assert profile.name == "artifact_hashes_v5"
    assert profile.contract_schema_version == 1
    assert "robustness.json" in profile.optional_bindings

    robustness = run_dir / "robustness.json"
    robustness.write_text('{"status": "pass"}\n', encoding="utf-8")
    manifest[robustness.name] = run_digests._hash_bytes(robustness.read_bytes())
    _rewrite_manifest_and_digest(run_dir, manifest)

    assert run_digests.validate_run_artifact_inventory(run_dir) == profile


def test_validate_run_artifact_inventory_rejects_unbound_optional_governed_file(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)
    (run_dir / "robustness.json").write_text('{"status": "pass"}\n', encoding="utf-8")

    with pytest.raises(RunDigestError, match=r"unbound governed files.*robustness\.json"):
        run_digests.validate_run_artifact_inventory(run_dir)


def test_validate_run_artifact_inventory_rejects_stale_optional_hash(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    robustness = run_dir / "robustness.json"
    robustness.write_text('{"status": "pass"}\n', encoding="utf-8")
    manifest[robustness.name] = run_digests._hash_bytes(robustness.read_bytes())
    _rewrite_manifest_and_digest(run_dir, manifest)
    robustness.write_text('{"status": "fail"}\n', encoding="utf-8")

    with pytest.raises(RunDigestError, match="robustness.json hash mismatch"):
        run_digests.validate_run_artifact_inventory(run_dir)


@pytest.mark.parametrize(
    "component_form",
    ["legacy_single", "legacy_single_no_text", "archived_single", "archived_multi"],
)
def test_validate_run_artifact_inventory_accepts_complete_component_forms(tmp_path, component_form: str) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    summary: list[dict[str, object]] = []
    count = 2 if component_form == "archived_multi" else 1
    for index in range(count):
        if component_form.startswith("legacy_single"):
            manifest_path = run_dir / "component_manifest.json"
            archived_path = None
        else:
            relative = Path("component_extensions") / f"{index:02d}_component" / "component_manifest.json"
            manifest_path = run_dir / relative
            archived_path = relative.as_posix()
        bundle_hash = _write_component_manifest_fixture(manifest_path, f"component_{index}")
        item: dict[str, object] = {
            "manifest_path": f"/deleted/component_{index}/component_manifest.json",
            "bundle_hash": bundle_hash,
        }
        if archived_path is not None:
            item["archived_manifest_path"] = archived_path
        summary.append(item)

    summary_content = (json.dumps(summary, indent=2) + "\n").encode()
    _bind_component_file(run_dir, manifest, "component_manifests.json", summary_content)
    if component_form in {"legacy_single", "archived_single"}:
        text_content = (str(summary[0]["bundle_hash"]) + "\n").encode()
        _bind_component_file(run_dir, manifest, "component_bundle_hash.txt", text_content)
    if component_form.startswith("legacy_single"):
        content = (run_dir / "component_manifest.json").read_bytes()
        manifest["component_manifest.json"] = run_digests._hash_bound_artifact("component_manifest.json", content)
    _rewrite_manifest_and_digest(run_dir, manifest)

    run_digests.validate_run_artifact_inventory(run_dir)


@pytest.mark.parametrize("orphan", ["component_manifest.json", "component_bundle_hash.txt"])
def test_validate_run_artifact_inventory_rejects_incomplete_component_group(tmp_path, orphan: str) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    if orphan == "component_manifest.json":
        _write_component_manifest_fixture(run_dir / orphan, "orphan")
        content = (run_dir / orphan).read_bytes()
        manifest[orphan] = run_digests._hash_bound_artifact(orphan, content)
    elif orphan == "component_bundle_hash.txt":
        _bind_component_file(run_dir, manifest, orphan, ("sha256:" + "1" * 64 + "\n").encode())
    _rewrite_manifest_and_digest(run_dir, manifest)

    with pytest.raises(RunDigestError, match="component provenance"):
        run_digests.validate_run_artifact_inventory(run_dir)


def test_validate_run_artifact_inventory_accepts_bound_empty_component_summary(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    _bind_component_file(run_dir, manifest, "component_manifests.json", b"[]\n")
    _rewrite_manifest_and_digest(run_dir, manifest)

    run_digests.validate_run_artifact_inventory(run_dir)


@pytest.mark.parametrize("forgery", ["summary", "text"])
def test_validate_run_artifact_inventory_rejects_invented_component_hash(tmp_path, forgery: str) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    relative = Path("component_extensions/00_component/component_manifest.json")
    bundle_hash = _write_component_manifest_fixture(run_dir / relative, "component")
    summary_hash = "sha256:" + "1" * 64 if forgery == "summary" else bundle_hash
    summary = [
        {
            "manifest_path": "/deleted/component_manifest.json",
            "archived_manifest_path": relative.as_posix(),
            "bundle_hash": summary_hash,
        }
    ]
    _bind_component_file(run_dir, manifest, "component_manifests.json", (json.dumps(summary) + "\n").encode())
    text_hash = "sha256:" + "2" * 64 if forgery == "text" else summary_hash
    _bind_component_file(run_dir, manifest, "component_bundle_hash.txt", (text_hash + "\n").encode())
    _rewrite_manifest_and_digest(run_dir, manifest)

    with pytest.raises(RunDigestError, match="component.*hash mismatch"):
        run_digests.validate_run_artifact_inventory(run_dir)


def test_validate_run_artifact_inventory_rejects_component_summary_without_manifest_path(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    relative = Path("component_extensions/00_component/component_manifest.json")
    bundle_hash = _write_component_manifest_fixture(run_dir / relative, "component")
    summary = [{"archived_manifest_path": relative.as_posix(), "bundle_hash": bundle_hash}]
    _bind_component_file(run_dir, manifest, "component_manifests.json", (json.dumps(summary) + "\n").encode())
    _rewrite_manifest_and_digest(run_dir, manifest)

    with pytest.raises(RunDigestError, match=r"component_manifests\.json\[0\]\.manifest_path is required"):
        run_digests.validate_run_artifact_inventory(run_dir)


def test_validate_run_artifact_inventory_supports_explicit_schema_0_legacy_profile(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(
        run_dir,
        schema_version=0,
        artifact_names={"data_manifest.json", "equity_curve.csv", "trades.csv", "metrics.json"},
        data_manifest_schema_version=None,
    )
    profile = run_digests.validate_run_artifact_inventory(run_dir)

    assert profile.name == "artifact_hashes_v0_legacy"
    require_current_run_digest(run_dir)


@pytest.mark.parametrize("schema_version", [6, -1])
def test_validate_run_artifact_inventory_rejects_unsupported_schema(tmp_path, schema_version: int) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir, schema_version=schema_version, write_digest=False)

    with pytest.raises(RunDigestError, match="unsupported artifact_hashes.json schema_version"):
        run_digests.validate_run_artifact_inventory(run_dir)


def test_validate_run_artifact_inventory_rejects_legacy_profile_for_current_data_manifest(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(
        run_dir,
        schema_version=0,
        artifact_names={"data_manifest.json", "equity_curve.csv", "trades.csv", "metrics.json"},
        data_manifest_schema_version=1,
        write_digest=False,
    )

    with pytest.raises(RunDigestError, match="cannot use legacy artifact inventory"):
        run_digests.validate_run_artifact_inventory(run_dir)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("data_manifest.json", b'{\n  "schema_version": 1, "source": "test"\n}\n'),
        ("metrics.json", b'{"run_id":"volatile", "value": 1}\n'),
        ("environment.json", b'{"run_timestamp":"volatile", "python":"3"}\n'),
        ("execution_assumptions.json", b'{"timing": "close"}\n'),
        ("compiled_plan.json", b'{"plan": [1, 2]}\n'),
        ("runtime_audit.json", b'{"status": "pass"}\n'),
        ("robustness.json", b'{ "status": "pass" }\n'),
        ("reproducibility_audit.json", b'{ "status": "pass" }\n'),
        ("research_bias_audit.json", b'{ "status": "pass" }\n'),
    ],
)
def test_publish_registered_json_uses_artifact_contract_hash(tmp_path, name: str, content: bytes) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)

    publish_run_artifacts(run_dir, {name: content})

    artifact_hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert artifact_hashes[name] == run_digests._hash_bound_artifact(name, content)
    require_current_run_digest(run_dir)


def test_publish_spec_audit_derives_canonical_hash_with_compatible_argument(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)
    spec_audit = b'{\n  "status": "pass", "rows": []\n}\n'
    artifacts = {
        "spec_audit.json": spec_audit,
        "conversation_hash.txt": b"sha256:conversation\n",
        "component_catalog_hash.txt": b"sha256:catalog\n",
        "recipe_catalog_hash.txt": b"sha256:recipe\n",
    }

    publish_run_artifacts(run_dir, artifacts, canonical_json={"spec_audit.json"})

    artifact_hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert artifact_hashes["spec_audit.json"] == run_digests._hash_bound_artifact("spec_audit.json", spec_audit)
    require_current_run_digest(run_dir)


def test_publish_run_artifacts_rejects_canonical_json_semantic_override(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)

    with pytest.raises(run_digests.ArtifactHashesError, match="canonical_json cannot override"):
        publish_run_artifacts(
            run_dir,
            {"robustness.json": b'{ "status": "pass" }\n'},
            canonical_json={"robustness.json"},
        )

    assert not (run_dir / "robustness.json").exists()
    assert json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize(
    "artifact_name",
    [
        "",
        ".",
        "..",
        "./asset.txt",
        "../outside.txt",
        "/tmp/outside.txt",
        "nested/./asset.txt",
        "nested/../asset.txt",
        "nested//asset.txt",
        "nested/asset.txt/",
        r"..\outside.txt",
        r"C:\portable.txt",
        "C:portable.txt",
        "C:/portable.txt",
        r"\\server\share\asset.txt",
        "//server/share/asset.txt",
        r"nested\asset.txt",
        "CON",
        "con.txt",
        "nested/AUX.log",
        "NUL.tar.gz",
        "COM1.csv",
        "COM\N{SUPERSCRIPT ONE}.csv",
        "lpt9",
        "LPT\N{SUPERSCRIPT THREE}.log",
        "bad<name>.txt",
        "bad>name.txt",
        'bad"name.txt',
        "bad|name.txt",
        "bad?name.txt",
        "bad*name.txt",
        "control\x1f.txt",
        "trailing.",
        "trailing ",
    ],
)
def test_publish_run_artifacts_rejects_nonportable_artifact_paths(
    tmp_path,
    artifact_name: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})

    with pytest.raises(run_digests.ArtifactHashesError, match="unsafe artifact path"):
        publish_run_artifacts(run_dir, {artifact_name: b"new\n"})


@pytest.mark.parametrize(
    "artifact_name",
    [r"..\outside.txt", r"C:\portable.txt", r"nested\asset.txt", "CON.txt", "bad?.txt", "trailing."],
)
def test_publish_run_artifacts_rejects_nonportable_artifact_removals(
    tmp_path,
    artifact_name: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})

    with pytest.raises(run_digests.ArtifactHashesError, match="unsafe artifact path"):
        publish_run_artifacts(run_dir, {}, remove_artifacts={artifact_name})


@pytest.mark.parametrize("replacement_name", ["PRN.log", "bad|tree", "trailing "])
def test_publish_run_artifacts_rejects_nonportable_replacement_targets(
    tmp_path,
    replacement_name: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})
    replacement = tmp_path / "replacement"
    replacement.mkdir()

    with pytest.raises(run_digests.ArtifactHashesError, match="unsafe replacement path"):
        publish_run_artifacts(
            run_dir,
            {},
            replacement_paths={replacement_name: replacement},
        )


@pytest.mark.parametrize(
    "entries",
    [
        ["Report.txt", "report.txt"],
        ["caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt", "cafe\N{COMBINING ACUTE ACCENT}.txt"],
    ],
    ids=["casefold", "unicode-normalization"],
)
def test_publish_run_artifacts_rejects_portable_artifact_name_collisions(
    tmp_path,
    entries: list[str],
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})
    original_manifest = (run_dir / "artifact_hashes.json").read_bytes()
    original_digest = (run_dir.parent / "run_digests.jsonl").read_bytes()

    with pytest.raises(run_digests.ArtifactHashesError, match="portable path collision"):
        publish_run_artifacts(run_dir, {name: name.encode() for name in entries})

    assert (run_dir / "artifact_hashes.json").read_bytes() == original_manifest
    assert (run_dir.parent / "run_digests.jsonl").read_bytes() == original_digest
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


def test_publish_rejects_portable_collision_across_publication_and_removal(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"Report.txt": b"old\n"})
    original_manifest = (run_dir / "artifact_hashes.json").read_bytes()
    original_digest = (run_dir.parent / "run_digests.jsonl").read_bytes()

    with pytest.raises(run_digests.ArtifactHashesError, match="portable path collision"):
        publish_run_artifacts(
            run_dir,
            {"report.txt": b"new\n"},
            remove_artifacts={"Report.txt"},
        )

    assert (run_dir / "artifact_hashes.json").read_bytes() == original_manifest
    assert (run_dir.parent / "run_digests.jsonl").read_bytes() == original_digest
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


def test_publish_rejects_portable_collision_across_replacement_targets(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    with pytest.raises(run_digests.ArtifactHashesError, match="portable path collision"):
        publish_run_artifacts(
            run_dir,
            {},
            replacement_paths={"ComponentTree": first, "componenttree": second},
        )

    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


def test_publish_and_inventory_preserve_nested_posix_artifact_paths(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})

    publish_run_artifacts(run_dir, {"nested/asset.txt": b"nested\n"})

    assert (run_dir / "nested" / "asset.txt").read_bytes() == b"nested\n"
    manifest = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert manifest["nested/asset.txt"] == run_digests._hash_bytes(b"nested\n")
    require_current_run_digest(run_dir)

    publish_run_artifacts(run_dir, {}, remove_artifacts={"nested/asset.txt"})

    assert not (run_dir / "nested" / "asset.txt").exists()
    manifest = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "nested/asset.txt" not in manifest
    require_current_run_digest(run_dir)


def test_publish_validates_completed_inventory_before_clearing_journal(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)
    journal_path = run_dir.parent / "run_digests.jsonl.journal"
    original_validate = run_digests._validate_run_artifact_inventory
    validation_observations: list[bool] = []

    def observe_validation(run_path, artifact_hashes):
        validation_observations.append(journal_path.exists())
        return original_validate(run_path, artifact_hashes)

    monkeypatch.setattr(run_digests, "_validate_run_artifact_inventory", observe_validation)

    publish_run_artifacts(run_dir, {"robustness.json": b'{"status":"pass"}\n'})

    assert validation_observations == [True]
    assert not journal_path.exists()


def test_publish_rolls_back_completed_invalid_inventory(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    original_manifest = _write_versioned_run(run_dir)
    digest_path = run_dir.parent / "run_digests.jsonl"
    original_digest = digest_path.read_bytes()

    with pytest.raises(RunDigestError, match="incomplete component provenance"):
        publish_run_artifacts(
            run_dir,
            {"component_bundle_hash.txt": ("sha256:" + "1" * 64 + "\n").encode()},
        )

    assert not (run_dir / "component_bundle_hash.txt").exists()
    assert json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8")) == original_manifest
    assert digest_path.read_bytes() == original_digest
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


def test_publish_rolls_back_completed_invalid_digest_row(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    original_manifest = _write_versioned_run(run_dir)
    digest_path = run_dir.parent / "run_digests.jsonl"
    original_digest = digest_path.read_bytes()
    original_replace = run_digests._replace_run_digest_entry_locked

    def corrupt_digest_row(run_path: Path, artifact_hashes_hash: str) -> None:
        original_replace(run_path, artifact_hashes_hash)
        digest_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(run_digests, "_replace_run_digest_entry_locked", corrupt_digest_row)

    with pytest.raises(RunDigestError, match="run_digests.jsonl entry 1 has invalid run_id"):
        publish_run_artifacts(run_dir, {"robustness.json": b'{"status":"pass"}\n'})

    assert not (run_dir / "robustness.json").exists()
    assert json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8")) == original_manifest
    assert digest_path.read_bytes() == original_digest
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


def test_replace_run_digest_rejects_completed_invalid_inventory(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    digest_path = run_dir.parent / "run_digests.jsonl"
    original_digest = digest_path.read_bytes()
    orphan_content = ("sha256:" + "1" * 64 + "\n").encode()
    (run_dir / "component_bundle_hash.txt").write_bytes(orphan_content)
    manifest["component_bundle_hash.txt"] = run_digests._hash_bytes(orphan_content)
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RunDigestError, match="incomplete component provenance"):
        replace_run_digest_entry(run_dir, run_digests._hash_json_payload(manifest))

    assert digest_path.read_bytes() == original_digest
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


def test_commit_recovery_preserves_journal_when_completed_state_is_invalid(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    manifest = _write_versioned_run(run_dir)
    digest_path = run_dir.parent / "run_digests.jsonl"
    orphan_content = ("sha256:" + "1" * 64 + "\n").encode()
    invalid_manifest = {
        **manifest,
        "component_bundle_hash.txt": run_digests._hash_bytes(orphan_content),
    }
    invalid_digest = run_digests._hash_json_payload(invalid_manifest)
    digest_entry = json.loads(digest_path.read_text(encoding="utf-8"))
    digest_entry["artifact_hashes"] = invalid_digest
    targets = [
        run_digests._journal_target(
            "artifact",
            run_dir / "component_bundle_hash.txt",
            orphan_content,
            name="component_bundle_hash.txt",
        ),
        run_digests._journal_target(
            "manifest",
            run_dir / "artifact_hashes.json",
            (json.dumps(invalid_manifest, indent=2, sort_keys=True) + "\n").encode(),
        ),
        run_digests._journal_target(
            "digest",
            digest_path,
            (json.dumps(digest_entry, sort_keys=True) + "\n").encode(),
        ),
    ]
    journal_path = run_dir.parent / "run_digests.jsonl.journal"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recovery": "commit",
                "run_id": run_dir.name,
                "targets": targets,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunDigestError, match="incomplete component provenance"):
        run_digests._recover_publication_locked(run_dir.parent)

    assert journal_path.exists()
    assert (run_dir / "component_bundle_hash.txt").read_bytes() == orphan_content


@pytest.mark.parametrize("boundary", ["before_digest_replace", "after_digest_replace"])
def test_replace_run_digest_crash_recovery_retains_journal_for_incorrect_postcondition(
    tmp_path,
    boundary: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    wrong_digest = "sha256:" + "9" * 16
    script = """
import os
import sys
from pathlib import Path
from oxq import run_digests

run_dir = Path(sys.argv[1])
digest = sys.argv[2]
boundary = sys.argv[3]
original_replace = run_digests._replace_run_digest_entry_locked

def terminate(run_path, artifact_hashes_hash):
    if boundary == "before_digest_replace":
        os._exit(94)
    original_replace(run_path, artifact_hashes_hash)
    if boundary == "after_digest_replace":
        os._exit(94)

run_digests._replace_run_digest_entry_locked = terminate
run_digests.replace_run_digest_entry(run_dir, digest)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), wrong_digest, boundary],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        check=False,
    )
    journal_path = run_dir.parent / "run_digests.jsonl.journal"
    assert process.returncode == 94, (process.stdout, process.stderr)
    assert journal_path.is_file()
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal["postconditions"] == {"artifact_hashes": wrong_digest}

    with pytest.raises(RunDigestError, match="completed publication.*mismatch"):
        require_current_run_digest(run_dir)

    assert journal_path.is_file()


@pytest.mark.parametrize("boundary", ["before_digest_replace", "after_digest_replace"])
def test_replace_run_digest_recovers_valid_digest_only_crash_boundary(
    tmp_path,
    boundary: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    new_content = b"published\n"
    (run_dir / "published.txt").write_bytes(new_content)
    manifest_path = run_dir / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["published.txt"] = run_digests._hash_bytes(new_content)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    expected_digest = run_digests._hash_json_payload(manifest)
    script = """
import os
import sys
from pathlib import Path
from oxq import run_digests

run_dir = Path(sys.argv[1])
digest = sys.argv[2]
boundary = sys.argv[3]
original_replace = run_digests._replace_run_digest_entry_locked

def terminate(run_path, artifact_hashes_hash):
    if boundary == "before_digest_replace":
        os._exit(95)
    original_replace(run_path, artifact_hashes_hash)
    if boundary == "after_digest_replace":
        os._exit(95)

run_digests._replace_run_digest_entry_locked = terminate
run_digests.replace_run_digest_entry(run_dir, digest)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), expected_digest, boundary],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        check=False,
    )
    assert process.returncode == 95, (process.stdout, process.stderr)

    require_current_run_digest(run_dir)

    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


def test_publish_run_artifacts_serializes_concurrent_output_manifest_and_digest_updates(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def publish(name: str) -> None:
        try:
            barrier.wait(timeout=5)
            publish_run_artifacts(run_dir, {name: f"{name}\n".encode()})
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=publish, args=("first.txt",))
    second = threading.Thread(target=publish, args=("second.txt",))
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert failures == []
    manifest = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert {"metrics.txt", "first.txt", "second.txt"} <= manifest.keys()
    require_current_run_digest(run_dir)


def test_publish_run_artifacts_rolls_back_output_when_manifest_replace_fails(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    manifest_path = run_dir / "artifact_hashes.json"
    original_manifest = manifest_path.read_bytes()
    original_digest = (run_dir.parent / "run_digests.jsonl").read_bytes()
    original_replace = run_digests.os.replace

    def fail_manifest_replace(source, target) -> None:
        if target == manifest_path:
            raise OSError("manifest replace failed")
        original_replace(source, target)

    monkeypatch.setattr(run_digests.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="manifest replace failed"):
        publish_run_artifacts(run_dir, {"robustness.json": b"new\n"})

    assert not (run_dir / "robustness.json").exists()
    assert manifest_path.read_bytes() == original_manifest
    assert (run_dir.parent / "run_digests.jsonl").read_bytes() == original_digest
    require_current_run_digest(run_dir)


@pytest.mark.parametrize(
    ("boundary", "committed"),
    [
        ("journal.temp_fsync", False),
        ("journal.replace", True),
        ("journal.dir_fsync", True),
        ("artifact:robustness.json.temp_fsync", True),
        ("artifact:robustness.json.replace", True),
        ("artifact:robustness.json.dir_fsync", True),
        ("manifest.temp_fsync", True),
        ("manifest.replace", True),
        ("manifest.dir_fsync", True),
        ("digest.temp_fsync", True),
        ("digest.replace", True),
        ("digest.dir_fsync", True),
        ("journal.unlink", True),
        ("journal.unlink_dir_fsync", True),
    ],
)
def test_publish_run_artifacts_recovers_real_process_crash_at_every_boundary(
    tmp_path,
    boundary: str,
    committed: bool,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    script = """
import os
import sys
from pathlib import Path
from oxq import run_digests

run_dir = Path(sys.argv[1])
boundary = sys.argv[2]

def terminate(label):
    if label == boundary:
        os._exit(91)

run_digests._publication_boundary = terminate
run_digests.publish_run_artifacts(run_dir, {"robustness.json": b'{"status":"new"}\\n'})
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), boundary],
        cwd=Path.cwd(),
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert completed.returncode == 91
    require_current_run_digest(run_dir)
    assert (run_dir / "robustness.json").exists() is committed
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()
    assert list(run_dir.glob(".*.tmp")) == []
    assert list(run_dir.parent.glob(".run_digests.jsonl.*.tmp")) == []


def test_publish_run_artifacts_recovers_sigterm_during_commit(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    script = """
import os
import signal
import sys
from pathlib import Path
from oxq import run_digests

run_dir = Path(sys.argv[1])

def terminate(label):
    if label == "manifest.replace":
        os.kill(os.getpid(), signal.SIGTERM)

run_digests._publication_boundary = terminate
run_digests.publish_run_artifacts(run_dir, {"robustness.json": b'new\\n'})
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(run_dir)],
        cwd=Path.cwd(),
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert completed.returncode < 0
    require_current_run_digest(run_dir)
    assert (run_dir / "robustness.json").read_bytes() == b"new\n"


@pytest.mark.parametrize(
    ("boundary", "committed"),
    [
        ("journal.temp_fsync", False),
        ("journal.replace", True),
        ("journal.dir_fsync", True),
        ("path:component_extensions.temp_fsync", True),
        ("path:component_extensions.old_replace", True),
        ("path:component_extensions.old_dir_fsync", True),
        ("path:component_extensions.replace", True),
        ("path:component_extensions.dir_fsync", True),
        ("path:component_extensions.backup_unlink", True),
        ("path:component_extensions.backup_unlink_dir_fsync", True),
        ("artifact:robustness.json.temp_fsync", True),
        ("artifact:robustness.json.replace", True),
        ("artifact:robustness.json.dir_fsync", True),
        ("manifest.temp_fsync", True),
        ("manifest.replace", True),
        ("manifest.dir_fsync", True),
        ("digest.temp_fsync", True),
        ("digest.replace", True),
        ("digest.dir_fsync", True),
        ("journal.unlink", True),
        ("journal.unlink_dir_fsync", True),
    ],
)
def test_publish_run_artifacts_recovers_component_tree_crash_at_every_boundary(
    tmp_path,
    boundary: str,
    committed: bool,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)
    target = run_dir / "component_extensions"
    target.mkdir()
    (target / "stale.py").write_text("STALE = True\n", encoding="utf-8")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "current.py").write_text("CURRENT = True\n", encoding="utf-8")
    script = """
import os
import sys
from pathlib import Path
from oxq import run_digests

run_dir = Path(sys.argv[1])
replacement = Path(sys.argv[2])
boundary = sys.argv[3]

def terminate(label):
    if label == boundary:
        os._exit(91)

run_digests._publication_boundary = terminate
run_digests.publish_run_artifacts(
    run_dir,
    {"robustness.json": b'{"status":"new"}\\n'},
    replacement_paths={"component_extensions": replacement},
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), str(replacement), boundary],
        cwd=Path.cwd(),
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert completed.returncode == 91
    require_current_run_digest(run_dir)
    assert (target / "current.py").exists() is committed
    assert (target / "stale.py").exists() is (not committed)
    assert (run_dir / "robustness.json").exists() is committed
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()
    assert not (run_dir / ".component_extensions.oxq-path-new").exists()
    assert not (run_dir / ".component_extensions.oxq-path-old").exists()


@pytest.mark.parametrize(
    "replacement_name",
    [
        "",
        ".",
        "..",
        "./outside",
        "../outside",
        "/tmp/outside",
        "nested/./outside",
        "nested/../outside",
        "nested//outside",
        "nested/outside/",
        r"..\outside.txt",
        r"C:\portable.txt",
        "C:portable.txt",
        "C:/portable.txt",
        r"\\server\share\asset.txt",
        "//server/share/asset.txt",
        r"nested\asset.txt",
    ],
)
def test_publish_run_artifacts_rejects_unsafe_replacement_path(tmp_path, replacement_name: str) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)
    replacement = tmp_path / "replacement"
    replacement.mkdir()

    with pytest.raises(run_digests.ArtifactHashesError, match="unsafe replacement path"):
        publish_run_artifacts(
            run_dir,
            {"robustness.json": b"new\n"},
            replacement_paths={replacement_name: replacement},
        )


def test_publish_run_artifacts_preserves_nested_posix_replacement_path(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})
    replacement = tmp_path / "replacement"
    source_file = replacement / "package" / "module.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("VALUE = 1\n", encoding="utf-8")

    publish_run_artifacts(
        run_dir,
        {},
        replacement_paths={"nested/component_extensions": replacement},
    )

    target = run_dir / "nested" / "component_extensions" / "package" / "module.py"
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
    require_current_run_digest(run_dir)


@pytest.mark.parametrize(
    "relative_path",
    [r"..\outside.txt", r"C:\portable.txt", r"nested\asset.txt"],
)
def test_replacement_snapshot_rejects_windows_path_spellings(relative_path: str) -> None:
    snapshot = {
        "type": "directory",
        "mode": 0o755,
        "entries": [
            {
                "path": relative_path,
                "type": "file",
                "mode": 0o644,
                "content": run_digests._encode_content(b"content\n"),
            }
        ],
    }

    with pytest.raises(RunDigestError, match="unsafe or duplicate directory entry"):
        run_digests._validated_path_snapshot(snapshot)


@pytest.mark.parametrize(
    "relative_path",
    ["CON.txt", "nested/bad?.txt", "nested/trailing."],
)
def test_replacement_snapshot_rejects_nonportable_directory_entries(relative_path: str) -> None:
    snapshot = {
        "type": "directory",
        "mode": 0o755,
        "entries": [
            {
                "path": relative_path,
                "type": "file",
                "mode": 0o644,
                "content": run_digests._encode_content(b"content\n"),
            }
        ],
    }

    with pytest.raises(RunDigestError, match="unsafe or duplicate directory entry"):
        run_digests._validated_path_snapshot(snapshot)


def test_replacement_snapshot_rejects_normalized_casefold_entry_collision() -> None:
    snapshot = {
        "type": "directory",
        "mode": 0o755,
        "entries": [
            {"path": "Nested/Report.txt", "type": "file", "mode": 0o644, "content": None},
            {"path": "nested/report.TXT", "type": "file", "mode": 0o644, "content": None},
        ],
    }
    for entry in snapshot["entries"]:
        entry["content"] = run_digests._encode_content(b"content\n")

    with pytest.raises(RunDigestError, match="unsafe or duplicate directory entry"):
        run_digests._validated_path_snapshot(snapshot)


def test_inventory_rejects_normalized_casefold_binding_collision_before_file_reads(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    digest = "sha256:" + "1" * 16
    manifest = {"Report.txt": digest, "report.TXT": digest}
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "artifact_hashes": run_digests._hash_json_payload(manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RunDigestError, match="portable path collision"):
        require_current_run_digest(run_dir)


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("Report.txt", "report.txt"),
        ("caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt", "cafe\N{COMBINING ACUTE ACCENT}.txt"),
    ],
    ids=["case-sensitive", "normalization-sensitive"],
)
def test_inventory_rejects_colliding_bindings_when_filesystem_materializes_both_spellings(
    tmp_path,
    first_name: str,
    second_name: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    first_path = run_dir / first_name
    second_path = run_dir / second_name
    first_path.write_bytes(b"first\n")
    if second_path.exists():
        pytest.skip("filesystem aliases these portable spellings")
    second_path.write_bytes(b"second\n")
    if os.path.samefile(first_path, second_path):
        pytest.skip("filesystem aliases these portable spellings")
    manifest = {
        first_name: run_digests._hash_bytes(first_path.read_bytes()),
        second_name: run_digests._hash_bytes(second_path.read_bytes()),
    }
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "artifact_hashes": run_digests._hash_json_payload(manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RunDigestError, match="portable path collision"):
        require_current_run_digest(run_dir)


def test_publish_run_artifacts_rejects_symlink_in_replacement_tree(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE = True\n", encoding="utf-8")
    (replacement / "linked.py").symlink_to(outside)

    with pytest.raises(run_digests.ArtifactHashesError, match="symlink"):
        publish_run_artifacts(
            run_dir,
            {"robustness.json": b"new\n"},
            replacement_paths={"component_extensions": replacement},
        )

    assert not (run_dir / "component_extensions").exists()
    require_current_run_digest(run_dir)


@pytest.mark.parametrize("replacement_name", ["artifact_hashes.json", "artifact_hashes.json/child"])
def test_publish_rejects_replacement_overlap_with_managed_manifest_before_journal(
    tmp_path,
    replacement_name: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    manifest_path = run_dir / "artifact_hashes.json"
    digest_path = run_dir.parent / "run_digests.jsonl"
    original_manifest = manifest_path.read_bytes()
    original_digest = digest_path.read_bytes()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "content.txt").write_text("replacement\n", encoding="utf-8")

    with pytest.raises(run_digests.ArtifactHashesError, match="managed transaction targets overlap"):
        publish_run_artifacts(
            run_dir,
            {"new.txt": b"new\n"},
            replacement_paths={replacement_name: replacement},
        )

    assert manifest_path.is_file()
    assert manifest_path.read_bytes() == original_manifest
    assert digest_path.read_bytes() == original_digest
    assert not (run_dir / "new.txt").exists()
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


@pytest.mark.parametrize("staging_suffix", ["new", "old"])
def test_publish_rejects_replacement_collision_with_managed_path_staging_before_journal(
    tmp_path,
    staging_suffix: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    original_manifest = (run_dir / "artifact_hashes.json").read_bytes()
    original_digest = (run_dir.parent / "run_digests.jsonl").read_bytes()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    staging_replacement = tmp_path / "staging-replacement"
    staging_replacement.mkdir()
    staging_name = f".component_extensions.oxq-path-{staging_suffix}"

    with pytest.raises(run_digests.ArtifactHashesError, match="managed transaction targets overlap"):
        publish_run_artifacts(
            run_dir,
            {},
            replacement_paths={
                "component_extensions": replacement,
                staging_name: staging_replacement,
            },
        )

    assert (run_dir / "artifact_hashes.json").read_bytes() == original_manifest
    assert (run_dir.parent / "run_digests.jsonl").read_bytes() == original_digest
    assert not (run_dir / "component_extensions").exists()
    assert not (run_dir / staging_name).exists()
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


@pytest.mark.parametrize("target_kind", ["artifact", "replacement"])
def test_publish_rejects_manifest_atomic_temp_namespace_before_journal(
    tmp_path,
    target_kind: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    original_manifest = (run_dir / "artifact_hashes.json").read_bytes()
    original_digest = (run_dir.parent / "run_digests.jsonl").read_bytes()
    collision_name = ".artifact_hashes.json.collision.tmp"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    artifacts = {collision_name: b"collision\n"} if target_kind == "artifact" else {}
    replacements = {collision_name: replacement} if target_kind == "replacement" else {}

    with pytest.raises(run_digests.ArtifactHashesError, match="managed atomic temp namespace"):
        publish_run_artifacts(run_dir, artifacts, replacement_paths=replacements)

    assert (run_dir / "artifact_hashes.json").read_bytes() == original_manifest
    assert (run_dir.parent / "run_digests.jsonl").read_bytes() == original_digest
    assert not (run_dir / collision_name).exists()
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


def test_recovery_rejects_collision_with_managed_path_staging_before_mutation(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "content.txt").write_text("replacement\n", encoding="utf-8")
    path_target = run_digests._journal_path_target(run_dir, "component_extensions", replacement)
    staging_name = ".component_extensions.oxq-path-new"
    staging_target = run_digests._journal_target(
        "artifact",
        run_dir / staging_name,
        b"collision\n",
        name=staging_name,
    )
    journal_path = run_dir.parent / "run_digests.jsonl.journal"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recovery": "commit",
                "run_id": run_dir.name,
                "targets": [path_target, staging_target],
            }
        ),
        encoding="utf-8",
    )
    original_journal = journal_path.read_bytes()

    with pytest.raises(run_digests.RunDigestError, match="managed transaction targets overlap"):
        with run_digest_transaction(run_dir):
            pass

    assert journal_path.read_bytes() == original_journal
    assert not (run_dir / "component_extensions").exists()
    assert not (run_dir / staging_name).exists()


@pytest.mark.parametrize("alias_parent_kind", ["same", "different"])
@pytest.mark.parametrize("publisher", ["publish", "update", "replace"])
def test_public_publishers_use_canonical_run_identity_through_symlink_alias(
    tmp_path,
    alias_parent_kind: str,
    publisher: str,
) -> None:
    run_dir = tmp_path / "runs" / "real-run"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    alias_parent = run_dir.parent if alias_parent_kind == "same" else tmp_path / "aliases"
    alias_parent.mkdir(parents=True, exist_ok=True)
    alias = alias_parent / "alias-run"
    alias.symlink_to(run_dir, target_is_directory=True)

    if publisher == "publish":
        publish_run_artifacts(alias, {"published.txt": b"published\n"})
    elif publisher == "update":
        update_artifact_hashes_and_run_digest(alias, lambda manifest: None)
    else:
        manifest = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
        replace_run_digest_entry(alias, run_digests._hash_json_payload(manifest))

    require_current_run_digest(run_dir)
    entries = [
        json.loads(line)
        for line in (run_dir.parent / "run_digests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["run_id"] for entry in entries] == [run_dir.name]
    if alias_parent != run_dir.parent:
        assert not (alias_parent / "run_digests.jsonl").exists()
        assert not (alias_parent / "run_digests.jsonl.lock").exists()
        assert not (alias_parent / "run_digests.jsonl.journal").exists()


@pytest.mark.parametrize("alias_parent_kind", ["same", "different"])
def test_run_transaction_recovers_canonical_journal_through_symlink_alias(
    tmp_path,
    alias_parent_kind: str,
) -> None:
    run_dir = tmp_path / "runs" / "real-run"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    marker_path = run_dir / "recovery-marker.txt"
    marker_path.write_bytes(b"old\n")
    target = run_digests._journal_target(
        "artifact",
        marker_path,
        b"recovered\n",
        name=marker_path.name,
    )
    journal_path = run_dir.parent / "run_digests.jsonl.journal"
    journal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recovery": "commit",
                "run_id": run_dir.name,
                "targets": [target],
            }
        ),
        encoding="utf-8",
    )
    alias_parent = run_dir.parent if alias_parent_kind == "same" else tmp_path / "aliases"
    alias_parent.mkdir(parents=True, exist_ok=True)
    alias = alias_parent / "alias-run"
    alias.symlink_to(run_dir, target_is_directory=True)

    with run_digest_transaction(alias):
        pass

    assert marker_path.read_bytes() == b"recovered\n"
    assert not journal_path.exists()
    if alias_parent != run_dir.parent:
        assert not (alias_parent / "run_digests.jsonl.lock").exists()
        assert not (alias_parent / "run_digests.jsonl.journal").exists()


def test_public_publishers_use_actual_directory_entry_spelling_through_case_alias(tmp_path) -> None:
    run_dir = tmp_path / "Runs" / "ActualRun"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    alias_parent = _case_insensitive_alias(run_dir.parent)
    alias_run = alias_parent / run_dir.name.swapcase()
    assert os.path.samefile(run_dir, alias_run)

    publish_run_artifacts(alias_run, {"published.txt": b"published\n"})

    require_current_run_digest(run_dir)
    entries = run_digests._read_entries(run_dir.parent / "run_digests.jsonl")
    assert [entry["run_id"] for entry in entries] == ["ActualRun"]
    assert not (alias_parent / "run_digests.jsonl.journal").exists()


def test_multi_run_transaction_deduplicates_alternate_case_lock_identity(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "Runs" / "ActualRun"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    alias_parent = _case_insensitive_alias(run_dir.parent)
    alias_run = alias_parent / run_dir.name.swapcase()
    entered_locks: list[Path] = []

    class RecordingLock:
        def __init__(self, path: str | Path) -> None:
            self.path = Path(path)

        def __enter__(self):
            entered_locks.append(self.path)
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    monkeypatch.setattr(run_digests, "ProcessFileLock", RecordingLock)

    with run_digests.multi_run_digest_read_transaction([run_dir, alias_run]) as canonical:
        assert canonical == (run_dir, run_dir)

    assert len(entered_locks) == 1


def test_replace_run_digest_recovers_alternate_case_crash_with_actual_run_id(tmp_path) -> None:
    run_dir = tmp_path / "Runs" / "ActualRun"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    alias_parent = _case_insensitive_alias(run_dir.parent)
    alias_run = alias_parent / run_dir.name.swapcase()
    new_content = b"published\n"
    (run_dir / "published.txt").write_bytes(new_content)
    manifest_path = run_dir / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["published.txt"] = run_digests._hash_bytes(new_content)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    expected_digest = run_digests._hash_json_payload(manifest)
    script = """
import os
import sys
from pathlib import Path
from oxq import run_digests

run_dir = Path(sys.argv[1])
digest = sys.argv[2]
original_replace = run_digests._replace_run_digest_entry_locked

def terminate(run_path, artifact_hashes_hash):
    original_replace(run_path, artifact_hashes_hash)
    os._exit(96)

run_digests._replace_run_digest_entry_locked = terminate
run_digests.replace_run_digest_entry(run_dir, digest)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(alias_run), expected_digest],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        check=False,
    )
    assert process.returncode == 96, (process.stdout, process.stderr)

    require_current_run_digest(run_dir)

    entries = run_digests._read_entries(run_dir.parent / "run_digests.jsonl")
    assert [entry["run_id"] for entry in entries] == ["ActualRun"]
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


@pytest.mark.parametrize(
    "boundary",
    [
        "artifact:robustness.json.unlink",
        "artifact:robustness.json.unlink_dir_fsync",
    ],
)
def test_publish_run_artifacts_recovers_removed_artifact_crash_boundaries(tmp_path, boundary: str) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_versioned_run(run_dir)
    publish_run_artifacts(run_dir, {"robustness.json": b'{"status":"old"}\n'})
    script = """
import os
import sys
from pathlib import Path
from oxq import run_digests

run_dir = Path(sys.argv[1])
boundary = sys.argv[2]

def terminate(label):
    if label == boundary:
        os._exit(91)

run_digests._publication_boundary = terminate
run_digests.publish_run_artifacts(
    run_dir,
    {"research_bias_audit.json": b'{"status":"pass"}\\n'},
    remove_artifacts={"robustness.json"},
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), boundary],
        cwd=Path.cwd(),
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    assert completed.returncode == 91
    require_current_run_digest(run_dir)
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "robustness.json" not in hashes
    assert not (run_dir / "robustness.json").exists()
    assert "research_bias_audit.json" in hashes
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


@pytest.mark.parametrize("recovery", ["commit", "rollback"])
@pytest.mark.parametrize("run_id", [".", "..", "run-a/../..", r"run-a\..\.."])
def test_recovery_rejects_non_normal_run_id_before_outside_mutation(
    tmp_path,
    run_id: str,
    recovery: str,
) -> None:
    parent = tmp_path / "workspace" / "runs"
    parent.mkdir(parents=True)
    sentinel = tmp_path / "workspace" / "outside-sentinel.txt"
    sentinel.write_bytes(b"outside-original\n")
    journal_path = _write_recovery_journal(
        parent,
        run_id=run_id,
        recovery=recovery,
        old=b"rollback-overwrite\n",
        new=b"commit-overwrite\n",
    )
    original_journal = journal_path.read_bytes()

    with pytest.raises(RunDigestError, match="unsafe run_id"):
        run_digests._recover_publication_locked(parent)

    assert sentinel.read_bytes() == b"outside-original\n"
    assert journal_path.read_bytes() == original_journal


@pytest.mark.parametrize("run_id", ["CON", "nul.txt", "bad?.txt", "trailing.", "trailing "])
def test_recovery_rejects_nonportable_run_id_before_outside_mutation(
    tmp_path,
    run_id: str,
) -> None:
    parent = tmp_path / "workspace" / "runs"
    parent.mkdir(parents=True)
    journal_path = _write_recovery_journal(
        parent,
        run_id=run_id,
        recovery="commit",
        old=b"old\n",
        new=b"new\n",
    )
    original_journal = journal_path.read_bytes()

    with pytest.raises(RunDigestError, match="unsafe run_id"):
        run_digests._recover_publication_locked(parent)

    assert journal_path.read_bytes() == original_journal


def test_run_digest_registry_rejects_casefold_colliding_run_ids(tmp_path) -> None:
    digest_path = tmp_path / "run_digests.jsonl"
    digest_path.write_text(
        "".join(
            json.dumps({"run_id": run_id, "artifact_hashes": "sha256:" + digit * 16}) + "\n"
            for run_id, digit in (("ActualRun", "1"), ("actualrun", "2"))
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunDigestError, match="portable run_id collision"):
        run_digests._read_entries(digest_path)


def test_public_digest_writer_rejects_nonportable_run_directory_name(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "CON.txt"
    run_dir.mkdir(parents=True)
    (run_dir / "artifact_hashes.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RunDigestError, match="normally named directory"):
        update_artifact_hashes_and_run_digest(run_dir, lambda manifest: None)

    assert not (run_dir.parent / "run_digests.jsonl.lock").exists()
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()


@pytest.mark.parametrize("operation", ["journal_clear", "rollback_delete", "temp_cleanup"])
def test_windows_directory_fsync_callers_do_not_open_directories(monkeypatch, tmp_path, operation: str) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"target\n")

    def fail_open(*_args, **_kwargs):
        raise AssertionError("Windows directory fsync must not call os.open")

    monkeypatch.setattr(run_digests, "_is_windows", lambda: True, raising=False)
    monkeypatch.setattr(run_digests.os, "open", fail_open)

    if operation == "journal_clear":
        run_digests._clear_journal(target)
    elif operation == "rollback_delete":
        run_digests._restore_file_bytes(target, None)
    else:
        temp = tmp_path / ".target.txt.interrupted.tmp"
        temp.write_bytes(b"temp\n")
        run_digests._cleanup_atomic_temps(target)
        assert not temp.exists()

    assert not target.exists() or operation == "temp_cleanup"


def test_require_current_run_digest_rejects_changed_bound_artifact(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"original\n"})
    (run_dir / "metrics.txt").write_bytes(b"changed\n")

    with pytest.raises(RunDigestError, match="metrics.txt hash mismatch"):
        require_current_run_digest(run_dir)


def test_require_current_run_digest_rejects_symlinked_manifest(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"original\n"})
    manifest_path = run_dir / "artifact_hashes.json"
    manifest_copy = tmp_path / "manifest.json"
    manifest_copy.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(manifest_copy)

    with pytest.raises(RunDigestError, match="artifact_hashes.json is invalid"):
        require_current_run_digest(run_dir)


@pytest.mark.parametrize("artifact_name", ["../outside.txt", "/tmp/outside.txt"])
def test_require_current_run_digest_rejects_manifest_artifact_outside_run(tmp_path, artifact_name: str) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"metrics.txt": b"baseline\n"})
    manifest_path = run_dir / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[artifact_name] = "sha256:" + "1" * 16
    _rewrite_manifest_and_digest(run_dir, manifest)

    with pytest.raises(RunDigestError, match="unsafe artifact path"):
        require_current_run_digest(run_dir)


@pytest.mark.parametrize(
    "artifact_name",
    [r"..\outside.txt", r"C:\portable.txt", r"nested\asset.txt"],
)
def test_require_current_run_digest_rejects_windows_path_spellings(
    tmp_path,
    artifact_name: str,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    manifest = {artifact_name: "sha256:" + "1" * 16}
    (run_dir / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "artifact_hashes": run_digests._hash_json_payload(manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RunDigestError, match="unsafe artifact path"):
        require_current_run_digest(run_dir)


def test_require_current_run_digest_rejects_symlink_bound_artifact(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside\n")
    _write_current_run(run_dir, {"metrics.txt": b"outside\n"})
    (run_dir / "metrics.txt").unlink()
    (run_dir / "metrics.txt").symlink_to(outside)

    with pytest.raises(RunDigestError, match="regular, non-symlink"):
        require_current_run_digest(run_dir)


def test_require_current_run_digest_rejects_symlinked_artifact_parent(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_current_run(run_dir, {"nested/metrics.txt": b"outside\n"})
    (run_dir / "nested/metrics.txt").unlink()
    (run_dir / "nested").rmdir()
    (run_dir / "nested").symlink_to(outside, target_is_directory=True)
    (outside / "metrics.txt").write_bytes(b"outside\n")

    with pytest.raises(RunDigestError, match="regular, non-symlink"):
        require_current_run_digest(run_dir)


def test_stable_artifact_read_rejects_in_place_mutation_at_descriptor_eof(
    tmp_path,
    monkeypatch,
) -> None:
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"original\n")
    original_read = run_digests.os.read
    mutated = False

    def mutate_at_eof(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, size)
        if chunk == b"" and not mutated:
            mutated = True
            artifact_path.write_bytes(b"mutated!\n")
            status = artifact_path.stat()
            os.utime(
                artifact_path,
                ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000),
            )
        return chunk

    monkeypatch.setattr(run_digests.os, "read", mutate_at_eof)

    with pytest.raises(RunDigestError, match="changed while being read"):
        run_digests._read_regular_file_bytes(artifact_path, artifact_name=artifact_path.name)

    assert mutated


def test_replace_run_digest_entry_leaves_original_intact_when_atomic_replace_fails(
    monkeypatch,
    tmp_path,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    digest_path = run_dir.parent / "run_digests.jsonl"
    original = (
        json.dumps(
            {
                "run_id": run_dir.name,
                "artifact_hashes": "sha256:" + "1" * 16,
                "created_at": "2026-07-12T00:00:00+00:00",
            },
            sort_keys=True,
        )
        + "\n"
    )
    digest_path.write_text(original, encoding="utf-8")

    def fail_replace(_source, _target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("oxq.run_digests.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        replace_run_digest_entry(run_dir, "sha256:" + "2" * 16)

    assert digest_path.read_text(encoding="utf-8") == original
    assert list(run_dir.parent.glob(".run_digests.jsonl.*.tmp")) == []


def test_artifact_hashes_transaction_leaves_manifest_and_digest_intact_when_publish_fails(
    monkeypatch,
    tmp_path,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "artifact_hashes.json"
    _write_metrics_manifest(run_dir)
    update_artifact_hashes_and_run_digest(run_dir, lambda hashes: None)
    original_manifest = manifest_path.read_bytes()
    original_digest = (run_dir.parent / "run_digests.jsonl").read_bytes()
    original_replace = run_digests.os.replace

    def fail_manifest_replace(source, target) -> None:
        if target == manifest_path:
            raise OSError("manifest replace failed")
        original_replace(source, target)

    monkeypatch.setattr(run_digests.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="manifest replace failed"):
        update_artifact_hashes_and_run_digest(
            run_dir,
            lambda hashes: hashes.__setitem__("robustness.json", "sha256:" + "2" * 16),
        )

    assert manifest_path.read_bytes() == original_manifest
    assert (run_dir.parent / "run_digests.jsonl").read_bytes() == original_digest
    assert list(run_dir.glob(".artifact_hashes.json.*.tmp")) == []
    require_current_run_digest(run_dir)


def test_artifact_hashes_transaction_restores_manifest_when_digest_publish_fails(
    monkeypatch,
    tmp_path,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "artifact_hashes.json"
    digest_path = run_dir.parent / "run_digests.jsonl"
    _write_metrics_manifest(run_dir)
    update_artifact_hashes_and_run_digest(run_dir, lambda hashes: None)
    original_manifest = manifest_path.read_bytes()
    original_digest = digest_path.read_bytes()
    original_replace = run_digests.os.replace

    def fail_digest_replace(source, target) -> None:
        if target == digest_path:
            raise OSError("digest replace failed")
        original_replace(source, target)

    monkeypatch.setattr(run_digests.os, "replace", fail_digest_replace)

    with pytest.raises(OSError, match="digest replace failed"):
        update_artifact_hashes_and_run_digest(
            run_dir,
            lambda hashes: hashes.__setitem__("robustness.json", "sha256:" + "2" * 16),
        )

    assert manifest_path.read_bytes() == original_manifest
    assert digest_path.read_bytes() == original_digest
    assert list(run_dir.glob(".artifact_hashes.json.*.tmp")) == []
    assert list(run_dir.parent.glob(".run_digests.jsonl.*.tmp")) == []
    require_current_run_digest(run_dir)


@pytest.mark.parametrize("failing_directory_fsync", [1, 2], ids=["manifest", "digest"])
def test_artifact_hashes_transaction_recovers_pair_after_directory_fsync_failure(
    monkeypatch,
    tmp_path,
    failing_directory_fsync: int,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    _write_metrics_manifest(run_dir)
    update_artifact_hashes_and_run_digest(run_dir, lambda hashes: None)
    original_fsync = run_digests.os.fsync
    directory_fsyncs = 0

    def fail_selected_directory_fsync(fd: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(run_digests.os.fstat(fd).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == failing_directory_fsync:
                raise OSError(f"directory fsync {failing_directory_fsync} failed")
        original_fsync(fd)

    monkeypatch.setattr(run_digests.os, "fsync", fail_selected_directory_fsync)

    with pytest.raises(OSError, match=rf"directory fsync {failing_directory_fsync} failed"):
        update_artifact_hashes_and_run_digest(
            run_dir,
            lambda hashes: hashes.__setitem__("robustness.json", "sha256:" + "2" * 16),
        )

    require_current_run_digest(run_dir)


def test_artifact_hashes_update_and_digest_replacement_share_one_transaction(
    monkeypatch,
    tmp_path,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    artifact_hashes_path = run_dir / "artifact_hashes.json"
    _write_metrics_manifest(run_dir)
    update_artifact_hashes_and_run_digest(run_dir, lambda hashes: None)

    manifest_published = threading.Event()
    allow_digest_replace = threading.Event()
    reader_attempted = threading.Event()
    reader_completed = threading.Event()
    reader_failures: list[BaseException] = []
    writer_failures: list[BaseException] = []
    original_replace = run_digests._replace_run_digest_entry_locked
    original_transaction = run_digests.run_digest_transaction

    def paused_replace(run_path, artifact_hashes_hash) -> None:
        manifest_published.set()
        assert allow_digest_replace.wait(timeout=5)
        original_replace(run_path, artifact_hashes_hash)

    @contextmanager
    def observed_transaction(run_path) -> Iterator[None]:
        if threading.current_thread().name == "digest-reader":
            reader_attempted.set()
        with original_transaction(run_path):
            yield

    def write_update() -> None:
        try:
            update_artifact_hashes_and_run_digest(
                run_dir,
                lambda hashes: hashes.__setitem__("robustness.json", robustness_hash),
            )
        except BaseException as exc:
            writer_failures.append(exc)

    def read_pair() -> None:
        try:
            require_current_run_digest(run_dir)
        except BaseException as exc:
            reader_failures.append(exc)
        finally:
            reader_completed.set()

    robustness_content = b"robustness\n"
    (run_dir / "robustness.json").write_bytes(robustness_content)
    robustness_hash = "sha256:" + hashlib.sha256(robustness_content).hexdigest()[:16]
    monkeypatch.setattr(run_digests, "_replace_run_digest_entry_locked", paused_replace)
    monkeypatch.setattr(run_digests, "run_digest_transaction", observed_transaction)
    writer = threading.Thread(target=write_update, name="digest-writer")
    writer.start()
    assert manifest_published.wait(timeout=5)
    reader = threading.Thread(target=read_pair, name="digest-reader")
    reader.start()
    assert reader_attempted.wait(timeout=5)
    assert not reader_completed.is_set()
    allow_digest_replace.set()

    writer.join(timeout=5)
    reader.join(timeout=5)
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_failures == []
    assert reader_failures == []
    assert json.loads(artifact_hashes_path.read_text(encoding="utf-8"))["robustness.json"] == robustness_hash


@pytest.mark.parametrize(
    ("reader", "lock_target"),
    [
        pytest.param(report_qa_module._validate_governed_run_digest, run_digests, id="report"),
        pytest.param(main_module._require_run_digest_current, run_digests, id="compare"),
        pytest.param(
            reproducibility_module._check_run_digest,
            reproducibility_module,
            id="reproducibility",
        ),
    ],
)
def test_digest_readers_wait_for_compatible_transaction_lock(
    monkeypatch,
    tmp_path,
    reader: Callable,
    lock_target,
) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    _write_metrics_manifest(run_dir)
    update_artifact_hashes_and_run_digest(run_dir, lambda hashes: None)
    attempted = threading.Event()
    completed = threading.Event()
    failures: list[BaseException] = []
    original_transaction = run_digest_transaction

    @contextmanager
    def observed_transaction(run_path) -> Iterator[None]:
        attempted.set()
        with original_transaction(run_path):
            yield

    def read_pair() -> None:
        try:
            reader(run_dir)
        except BaseException as exc:
            failures.append(exc)
        finally:
            completed.set()

    with original_transaction(run_dir):
        monkeypatch.setattr(lock_target, "run_digest_transaction", observed_transaction)
        thread = threading.Thread(target=read_pair)
        thread.start()
        assert attempted.wait(timeout=5)
        assert not completed.is_set()

    assert completed.wait(timeout=5)
    thread.join(timeout=5)
    assert failures == []


def test_digest_validation_rejects_symlinked_run_digest_registry(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})
    digest_path = run_dir.parent / "run_digests.jsonl"
    outside_registry = tmp_path / "outside-run-digests.jsonl"
    outside_registry.write_bytes(digest_path.read_bytes())
    digest_path.unlink()
    digest_path.symlink_to(outside_registry)

    with pytest.raises(RunDigestError, match="regular, non-symlink"):
        require_current_run_digest(run_dir)

    assert outside_registry.read_bytes() != b""


def test_multi_run_read_transaction_acquires_unique_lock_classes_in_sorted_order(
    tmp_path,
    monkeypatch,
) -> None:
    runs: list[Path] = []
    for name in ("workspace-b", "workspace-a"):
        workspace = tmp_path / name
        run_dir = workspace / "versions" / "v001" / "09_backtests" / "run-a"
        _write_current_run(run_dir, {"baseline.txt": b"baseline\n"})
        config = workspace / ".open-xquant" / "workspace.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("workflow:\n  layout: version_governed\n", encoding="utf-8")
        runs.append(run_dir)
    alias = tmp_path / "run-alias"
    alias.symlink_to(runs[0], target_is_directory=True)
    events: list[tuple[str, Path]] = []

    class RecordingLock:
        def __init__(self, path: str | Path) -> None:
            self.path = Path(path).resolve(strict=False)

        def __enter__(self):
            events.append(("enter-run", self.path))
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            events.append(("exit-run", self.path))

    @contextmanager
    def recording_selection_lock(path: str | Path | None) -> Iterator[None]:
        assert path is not None
        resolved = Path(path).resolve(strict=False)
        events.append(("enter-final", resolved))
        try:
            yield
        finally:
            events.append(("exit-final", resolved))

    monkeypatch.setattr(run_digests, "ProcessFileLock", RecordingLock)
    monkeypatch.setattr(run_digests, "hold_final_selection_lock", recording_selection_lock)

    with run_digests.multi_run_digest_read_transaction([runs[0], runs[1], alias]) as canonical:
        assert canonical == (runs[0].resolve(), runs[1].resolve(), runs[0].resolve())

    run_locks = sorted(
        {
            (run.parent / "run_digests.jsonl.lock").resolve(strict=False)
            for run in runs
        },
        key=lambda path: os.path.normcase(str(path)),
    )
    final_locks = sorted(
        {
            (run.parents[3] / ".open-xquant/locks/final-selection.lock").resolve(strict=False)
            for run in runs
        },
        key=lambda path: os.path.normcase(str(path)),
    )
    assert events == [
        *(("enter-run", path) for path in run_locks),
        *(("enter-final", path) for path in final_locks),
        *(("exit-final", path) for path in reversed(final_locks)),
        *(("exit-run", path) for path in reversed(run_locks)),
    ]
