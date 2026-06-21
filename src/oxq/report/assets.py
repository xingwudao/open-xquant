"""Report asset manifest management."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

EMBEDDED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
MANIFEST_SCHEMA_VERSION = 1
ASSET_ID_ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
URL_RESERVED_PATH_CHARS = {"#", "?", "%", " ", "(", ")", "[", "]", "&"}
ASSET_KIND_SUBDIR = {"figure": "figures", "attachment": "attachments"}


@dataclass(frozen=True)
class AssetSource:
    script: str | None = None
    input_artifacts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> AssetSource:
        if not isinstance(raw, dict):
            return cls()
        artifacts = raw.get("input_artifacts", [])
        return cls(
            script=raw.get("script") if isinstance(raw.get("script"), str) else None,
            input_artifacts=[str(item) for item in artifacts] if isinstance(artifacts, list) else [],
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.script:
            data["script"] = self.script
        if self.input_artifacts:
            data["input_artifacts"] = self.input_artifacts
        return data


@dataclass(frozen=True)
class ReportAsset:
    id: str
    kind: str
    path: str
    title: str
    caption: str = ""
    section: str = "results"
    order: int = 100
    mime_type: str = "application/octet-stream"
    sha256: str = ""
    source: AssetSource = field(default_factory=AssetSource)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReportAsset:
        return cls(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            path=str(raw["path"]),
            title=str(raw["title"]),
            caption=str(raw.get("caption", "")),
            section=str(raw.get("section", "results")),
            order=int(raw.get("order", 100)),
            mime_type=str(raw.get("mime_type", "application/octet-stream")),
            sha256=str(raw.get("sha256", "")),
            source=AssetSource.from_dict(raw.get("source")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "path": self.path,
            "title": self.title,
            "caption": self.caption,
            "section": self.section,
            "order": self.order,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
        }
        source = self.source.to_dict()
        if source:
            data["source"] = source
        return data


@dataclass(frozen=True)
class ReportAssetBatchEntry:
    asset_id: str
    file_path: Path
    title: str
    caption: str = ""
    section: str = "results"
    order: int = 100
    source_script: Path | None = None
    source_artifacts: list[str] = field(default_factory=list)


def report_assets_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "report_assets"


def manifest_path(run_dir: str | Path) -> Path:
    return report_assets_dir(run_dir) / "manifest.json"


def safe_asset_id(asset_id: str) -> str:
    candidate = asset_id.strip()
    if (
        not candidate
        or candidate in {".", ".."}
        or "/" in candidate
        or "\\" in candidate
        or any(char in candidate for char in URL_RESERVED_PATH_CHARS)
        or any(char not in ASSET_ID_ALLOWED_CHARS for char in candidate)
    ):
        raise ValueError(f"invalid asset id: {asset_id}")
    path = Path(candidate)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid asset id: {asset_id}")
    return candidate


def list_report_assets(run_dir: str | Path) -> list[ReportAsset]:
    path = manifest_path(run_dir)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    assets = raw.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError(f"invalid report asset manifest: {path}")
    parsed = [_validate_manifest_asset(Path(run_dir), ReportAsset.from_dict(item)) for item in assets if isinstance(item, dict)]
    return sorted(parsed, key=_asset_sort_key)


def _list_report_assets_excluding(run_dir: Path, replacing_asset_ids: set[str]) -> list[ReportAsset]:
    path = manifest_path(run_dir)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    assets = raw.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError(f"invalid report asset manifest: {path}")
    parsed = []
    for item in assets:
        if not isinstance(item, dict):
            continue
        asset = ReportAsset.from_dict(item)
        if asset.id in replacing_asset_ids:
            continue
        parsed.append(_validate_manifest_asset(run_dir, asset))
    return sorted(parsed, key=_asset_sort_key)


def add_report_asset(
    run_dir: str | Path,
    file_path: str | Path,
    *,
    asset_id: str,
    title: str,
    caption: str = "",
    section: str = "results",
    order: int = 100,
    source_script: str | Path | None = None,
    source_artifacts: list[str] | None = None,
) -> ReportAsset:
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"run directory not found: {run_path}")

    source_path = Path(file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"asset file not found: {source_path}")

    asset_id = safe_asset_id(asset_id)
    existing = _list_report_assets_excluding(run_path, {asset_id})
    asset = _build_report_asset(
        run_path,
        source_path,
        asset_id=asset_id,
        title=title,
        caption=caption,
        section=section,
        order=order,
        source_script=source_script,
        source_artifacts=source_artifacts or [],
    )
    existing.append(asset)
    _write_manifest(run_path, existing)
    return asset


def add_report_assets(run_dir: str | Path, entries: Iterable[Mapping[str, Any]]) -> list[ReportAsset]:
    """Register multiple report assets in one manifest update."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"run directory not found: {run_path}")

    parsed_entries = [_parse_batch_entry(raw) for raw in entries]
    if not parsed_entries:
        raise ValueError("report asset batch is empty")
    id_counts: dict[str, int] = {}
    for entry in parsed_entries:
        id_counts[entry.asset_id] = id_counts.get(entry.asset_id, 0) + 1
    duplicate_ids = sorted(asset_id for asset_id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate report asset id in batch: {', '.join(duplicate_ids)}")

    replacing_ids = {entry.asset_id for entry in parsed_entries}
    existing = _list_report_assets_excluding(run_path, replacing_ids)
    assets = [
        _build_report_asset(
            run_path,
            entry.file_path,
            asset_id=entry.asset_id,
            title=entry.title,
            caption=entry.caption,
            section=entry.section,
            order=entry.order,
            source_script=entry.source_script,
            source_artifacts=entry.source_artifacts,
        )
        for entry in parsed_entries
    ]
    _write_manifest(run_path, existing + assets)
    return assets


def _parse_batch_entry(raw: Mapping[str, Any]) -> ReportAssetBatchEntry:
    if not isinstance(raw, Mapping):
        raise ValueError("report asset batch item must be an object")
    missing = [key for key in ("id", "file_path", "title") if key not in raw]
    if missing:
        raise ValueError(f"report asset batch item missing required field(s): {', '.join(missing)}")

    asset_id = safe_asset_id(str(raw["id"]))
    file_path = Path(str(raw["file_path"]))
    if not file_path.exists():
        raise FileNotFoundError(f"asset file not found: {file_path}")

    source_script = None
    if raw.get("source_script") is not None:
        source_script = Path(str(raw["source_script"]))
        if not source_script.exists():
            raise FileNotFoundError(f"source script not found: {source_script}")

    source_artifacts = raw.get("source_artifacts", [])
    if not isinstance(source_artifacts, list):
        raise ValueError("source_artifacts must be a list")

    try:
        order = int(raw.get("order", 100))
    except (TypeError, ValueError) as exc:
        raise ValueError("order must be an integer") from exc

    return ReportAssetBatchEntry(
        asset_id=asset_id,
        file_path=file_path,
        title=str(raw["title"]),
        caption=str(raw.get("caption", "")),
        section=str(raw.get("section", "results")),
        order=order,
        source_script=source_script,
        source_artifacts=[str(item) for item in source_artifacts],
    )


def _build_report_asset(
    run_path: Path,
    source_path: Path,
    *,
    asset_id: str,
    title: str,
    caption: str,
    section: str,
    order: int,
    source_script: str | Path | None,
    source_artifacts: list[str],
) -> ReportAsset:
    suffix = source_path.suffix.lower()
    kind = "figure" if suffix in EMBEDDED_IMAGE_EXTENSIONS else "attachment"
    subdir = "figures" if kind == "figure" else "attachments"
    destination = report_assets_dir(run_path) / subdir / f"{asset_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)

    copied_script = _copy_source_script(run_path, source_script, asset_id)
    asset = ReportAsset(
        id=asset_id,
        kind=kind,
        path=_relative_to_report_assets(run_path, destination),
        title=title,
        caption=caption,
        section=section,
        order=order,
        mime_type=mimetypes.guess_type(destination.name)[0] or "application/octet-stream",
        sha256=_sha256(destination),
        source=AssetSource(
            script=copied_script,
            input_artifacts=[str(item) for item in source_artifacts],
        ),
    )
    return asset


def _copy_source_script(run_dir: Path, source_script: str | Path | None, asset_id: str) -> str | None:
    if source_script is None:
        return None
    script_path = Path(source_script)
    if not script_path.exists():
        raise FileNotFoundError(f"source script not found: {script_path}")

    assets_dir = report_assets_dir(run_dir)
    assets_dir_resolved = assets_dir.resolve()
    script_path_resolved = script_path.resolve()
    try:
        return script_path_resolved.relative_to(assets_dir_resolved).as_posix()
    except ValueError:
        destination = assets_dir / "scripts" / script_path.name
        destination = _available_source_script_destination(destination, script_path_resolved, asset_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != script_path_resolved:
            shutil.copy2(script_path, destination)
        return destination.relative_to(assets_dir).as_posix()


def _available_source_script_destination(default_destination: Path, source_resolved: Path, asset_id: str) -> Path:
    if not default_destination.exists() or default_destination.resolve() == source_resolved:
        return default_destination

    candidate = default_destination.with_name(f"{asset_id}_{default_destination.name}")
    if not candidate.exists() or candidate.resolve() == source_resolved:
        return candidate

    counter = 2
    while True:
        candidate = default_destination.with_name(f"{asset_id}_{counter}_{default_destination.name}")
        if not candidate.exists() or candidate.resolve() == source_resolved:
            return candidate
        counter += 1


def _relative_to_report_assets(run_dir: Path, path: Path) -> str:
    return path.relative_to(report_assets_dir(run_dir)).as_posix()


def _validate_manifest_asset(run_dir: Path, asset: ReportAsset) -> ReportAsset:
    relative_path = _validate_manifest_asset_path(asset)
    asset_path = report_assets_dir(run_dir) / relative_path
    if not asset_path.exists():
        raise ValueError(f"missing report asset file: {asset_path}")
    if asset.sha256:
        actual_sha256 = _sha256(asset_path)
        if actual_sha256 != asset.sha256:
            raise ValueError(f"hash mismatch for report asset {asset.id}: expected {asset.sha256}, got {actual_sha256}")
    return ReportAsset(
        id=asset.id,
        kind=asset.kind,
        path=relative_path,
        title=asset.title,
        caption=asset.caption,
        section=asset.section,
        order=asset.order,
        mime_type=asset.mime_type,
        sha256=asset.sha256,
        source=asset.source,
    )


def _validate_manifest_asset_path(asset: ReportAsset) -> str:
    raw_path = asset.path
    if "\\" in raw_path or any(char in raw_path for char in URL_RESERVED_PATH_CHARS):
        raise ValueError(f"invalid report asset path for {asset.id}: {raw_path}")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid report asset path for {asset.id}: {raw_path}")
    expected_subdir = ASSET_KIND_SUBDIR.get(asset.kind)
    if expected_subdir is None or not path.parts or path.parts[0] != expected_subdir:
        raise ValueError(f"invalid report asset path for {asset.id}: {raw_path}")
    return path.as_posix()


def _write_manifest(run_dir: Path, assets: list[ReportAsset]) -> None:
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "assets": [asset.to_dict() for asset in sorted(assets, key=_asset_sort_key)],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _asset_sort_key(asset: ReportAsset) -> tuple[str, int, str]:
    return (asset.section, asset.order, asset.id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
