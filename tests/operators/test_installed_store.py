"""Installed operator release store safety tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from oxq.operators.installed_store import InstalledReleaseStore


TARGET = "cp312-cp312-macosx_14_0_arm64"


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _tree_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_digest(value).encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _write_release(staging: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    files = {
        "release-index.json": b'{"release":"1.0.0"}\n',
        "bundle.zip": b"PK\x03\x04certified bundle bytes",
        "publication/registry-entry.json": b'{"operators":[]}\n',
        "publication/bindings/equant.ttr.sma@1.0.0.binding.json": (
            b'{"operator_id":"equant.ttr.sma","operator_version":"1.0.0"}\n'
        ),
        "manifests/equant.ttr.sma@1.0.0.operator.json": (
            b'{"operator_id":"equant.ttr.sma","operator_version":"1.0.0"}\n'
        ),
        "baselines/technical-v1.json": (
            b'{"cases":[{"case_id":"case-1","operator_id":"equant.ttr.sma",'
            b'"operator_version":"1.0.0"}]}\n'
        ),
        "wheels/equant_ttr-1.0.0-cp312-cp312-macosx_14_0_arm64.whl": b"wheel bytes",
    }
    for name, value in files.items():
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    marker = {
        "schema_version": 1,
        "provider": "equant-py",
        "release": "1.0.0",
        "target": {"python_tag": "cp312", "abi_tag": "cp312", "platform_tag": "macosx_14_0_arm64"},
        "trust_state": "github-source-trusted",
        "certification_state": "research-certified",
        "release_index": _file("release-index.json", files["release-index.json"]),
        "bundle": _file("bundle.zip", files["bundle.zip"]),
        "files": [_file(name, value) for name, value in sorted(files.items())],
        "trees": [
            {
                "path": "publication",
                "digest": _tree_digest({name.removeprefix("publication/"): value for name, value in files.items() if name.startswith("publication/")}),
                "size_bytes": sum(len(value) for name, value in files.items() if name.startswith("publication/")),
            }
        ],
        "runtime_protocol_digest": _digest(b"runtime protocol"),
        "open_xquant_version_min": "1.0.0",
        "open_xquant_version_max": "1.0.0",
    }
    return marker, files


def _file(path: str, value: bytes) -> dict[str, object]:
    return {"path": path, "size_bytes": len(value), "digest": _digest(value)}


def create_partial_release(home: Path) -> None:
    partial = home / "equant-py" / "1.0.0" / TARGET
    partial.mkdir(parents=True)
    (partial / "bundle.zip").write_bytes(b"incomplete")


def test_release_is_invisible_until_valid_marker_is_committed(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    create_partial_release(store.home)
    assert store.list() == ()


def test_publish_validates_exact_file_set_and_is_idempotent(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, files = _write_release(staging)

    first = store.publish(staging, marker)
    second = store.publish(first.path, marker)

    assert first == second
    assert first.operators == (("equant.ttr.sma", "1.0.0"),)
    assert (first.path / "installed-release.json").read_bytes() == (
        json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert (first.path / "bundle.zip").read_bytes() == files["bundle.zip"]


def test_publish_rejects_corruption_and_existing_conflict(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    (staging / "bundle.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        store.publish(staging, marker)

    (staging / "bundle.zip").write_bytes(b"PK\x03\x04certified bundle bytes")
    store.publish(staging, marker)
    changed = dict(marker)
    changed["runtime_protocol_digest"] = _digest(b"other protocol")
    with pytest.raises(ValueError, match="conflict"):
        store.publish(staging, changed)


def test_publish_rejects_symlink_and_invalid_tree_digest(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    (staging / "wheels" / "linked.whl").symlink_to(staging / "bundle.zip")
    with pytest.raises(ValueError):
        store.publish(staging, marker)

    (staging / "wheels" / "linked.whl").unlink()
    marker["trees"] = [{"path": "publication", "digest": _digest(b"wrong"), "size_bytes": 1}]
    with pytest.raises(ValueError, match="tree"):
        store.publish(staging, marker)


def test_get_and_resolve_operator_return_validated_metadata(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    release = store.publish(staging, marker)

    assert store.get("equant-py", "1.0.0") == release
    operator = store.resolve_operator("equant.ttr.sma", "1.0.0", "equant-py", "1.0.0")
    assert operator.release == release
    assert operator.binding["operator_id"] == "equant.ttr.sma"
    assert operator.manifest["operator_version"] == "1.0.0"
    assert operator.certified_cases[0]["case_id"] == "case-1"


def test_snapshot_retains_verified_bytes_after_managed_path_is_replaced(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, files = _write_release(staging)
    release = store.publish(staging, marker)

    with store.snapshot(release) as snapshot:
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        moved = tmp_path / "moved-release"
        os.replace(release.path, moved)
        os.replace(replacement, release.path)
        assert snapshot.bundle == files["bundle.zip"]
        assert snapshot.release_index == files["release-index.json"]
        assert snapshot.publication_files["registry-entry.json"] == files["publication/registry-entry.json"]
        assert snapshot.wheel_snapshots
        assert next(iter(snapshot.wheel_snapshots.values())).read_bytes() == files[
            "wheels/equant_ttr-1.0.0-cp312-cp312-macosx_14_0_arm64.whl"
        ]


def test_home_defaults_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_XQUANT_OPERATOR_HOME", str(tmp_path / "custom-home"))
    assert InstalledReleaseStore().home == (tmp_path / "custom-home").resolve()


@pytest.mark.parametrize("field,value", [("release", "../escape"), ("release", "a/b"), ("release", ""), ("python_tag", ".."), ("abi_tag", "cp\\312"), ("platform_tag", "bad\x00tag")])
def test_publish_rejects_marker_path_components(tmp_path: Path, field: str, value: str) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    if field == "release":
        marker[field] = value
    else:
        marker["target"][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="marker"):
        store.publish(staging, marker)
    assert not any(store.home.rglob("bundle.zip")) if store.home.exists() else True


def test_publish_rejects_symlinked_provider_ancestor(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    outside = tmp_path / "outside"
    outside.mkdir()
    store.home.mkdir()
    (store.home / "equant-py").symlink_to(outside, target_is_directory=True)
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    with pytest.raises(ValueError, match="store"):
        store.publish(staging, marker)
    assert not any(outside.rglob("bundle.zip"))


def test_list_ignores_symlinked_provider_ancestor(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    outside_store = InstalledReleaseStore(tmp_path / "outside")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    outside_store.publish(staging, marker)
    store.home.mkdir()
    (store.home / "equant-py").symlink_to(outside_store.home / "equant-py", target_is_directory=True)
    assert store.list() == ()


def test_publish_revalidates_provider_after_staging_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import oxq.operators.installed_store as store_module

    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    outside = tmp_path / "outside"
    outside.mkdir()
    original = store_module.tempfile.mkdtemp

    def swap_provider(*args, **kwargs):
        result = original(*args, **kwargs)
        provider = store.home / "equant-py"
        moved = tmp_path / "moved-provider"
        os.replace(provider, moved)
        provider.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(store_module.tempfile, "mkdtemp", swap_provider)
    with pytest.raises(ValueError, match="store"):
        store.publish(staging, marker)
    assert not any(outside.rglob("bundle.zip"))


def test_snapshot_rejects_stale_release_path_replaced_by_outside_symlink(tmp_path: Path) -> None:
    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    release = store.publish(staging, marker)
    outside_store = InstalledReleaseStore(tmp_path / "outside")
    outside = outside_store.publish(release.path, marker)
    moved = tmp_path / "moved-release"
    os.replace(release.path, moved)
    release.path.symlink_to(outside.path, target_is_directory=True)
    with pytest.raises(ValueError, match="store"):
        with store.snapshot(release):
            pass


def test_publish_cleans_redirected_final_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import oxq.operators.installed_store as store_module

    store = InstalledReleaseStore(tmp_path / "home")
    staging = tmp_path / "staging"
    staging.mkdir()
    marker, _ = _write_release(staging)
    outside = tmp_path / "outside"
    outside.mkdir()
    original = store_module.replace_directory

    def redirect(source: Path, target: Path) -> None:
        provider = store.home / "equant-py"
        moved = tmp_path / "moved-provider"
        os.replace(provider, moved)
        provider.symlink_to(outside, target_is_directory=True)
        (outside / "1.0.0").mkdir()
        original(moved / "1.0.0" / source.name, target)

    monkeypatch.setattr(store_module, "replace_directory", redirect)
    with pytest.raises(ValueError, match="store"):
        store.publish(staging, marker)
    assert not any(outside.rglob("bundle.zip"))
