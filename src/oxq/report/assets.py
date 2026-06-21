"""Report asset manifest management."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EMBEDDED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
MANIFEST_SCHEMA_VERSION = 1


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


def report_assets_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "report_assets"


def manifest_path(run_dir: str | Path) -> Path:
    return report_assets_dir(run_dir) / "manifest.json"


def safe_asset_id(asset_id: str) -> str:
    candidate = asset_id.strip()
    if not candidate or candidate in {".", ".."} or "/" in candidate or "\\" in candidate:
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
    parsed = [ReportAsset.from_dict(item) for item in assets if isinstance(item, dict)]
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
    suffix = source_path.suffix.lower()
    kind = "figure" if suffix in EMBEDDED_IMAGE_EXTENSIONS else "attachment"
    subdir = "figures" if kind == "figure" else "attachments"
    destination = report_assets_dir(run_path) / subdir / f"{asset_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)

    copied_script = _copy_source_script(run_path, source_script)
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
            input_artifacts=[str(item) for item in source_artifacts or []],
        ),
    )

    existing = [item for item in list_report_assets(run_path) if item.id != asset.id]
    existing.append(asset)
    _write_manifest(run_path, existing)
    return asset


def _copy_source_script(run_dir: Path, source_script: str | Path | None) -> str | None:
    if source_script is None:
        return None
    script_path = Path(source_script)
    if not script_path.exists():
        raise FileNotFoundError(f"source script not found: {script_path}")

    assets_dir = report_assets_dir(run_dir)
    try:
        return script_path.relative_to(assets_dir).as_posix()
    except ValueError:
        destination = assets_dir / "scripts" / script_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script_path, destination)
        return destination.relative_to(assets_dir).as_posix()


def _relative_to_report_assets(run_dir: Path, path: Path) -> str:
    return path.relative_to(report_assets_dir(run_dir)).as_posix()


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
