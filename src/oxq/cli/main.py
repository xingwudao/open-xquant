"""oxq CLI — Agentic Quant Research Kernel command-line interface."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import tempfile
from pathlib import Path

import click
import yaml

from oxq.cli.agent import agent as agent_group
from oxq.cli.doctor import doctor
from oxq.cli.research import (
    _is_version_governed_workspace,
    _path_has_symlink_component,
    _resolve_active_version_dir,
    _resolve_version_phase_path,
)
from oxq.cli.research import (
    research as research_group,
)
from oxq.process_lock import ProcessFileLock, ProcessLockError, stable_filesystem_identity
from oxq.run_digests import (
    RunDigestError,
    multi_run_digest_read_transaction,
    publish_run_artifacts,
    run_digest_transaction,
)
from oxq.spec.schema import StrategySpec, make_strategy_id
from oxq.spec.validator import validate as validate_spec

_COMPILE_PREVIEW_MARKER_NAME = ".oxq-compile-preview.json"
_COMPILE_PREVIEW_MARKER = {
    "artifact": "strategy-compile-preview",
    "managed_by": "open-xquant",
    "schema_version": 1,
}
_COMPILE_PREVIEW_TRANSACTION_SCHEMA_VERSION = 2
_COMPILE_PREVIEW_TRANSACTION_TYPE = "compile-preview-replacement"

_WORKSPACE_VERSION_RE = re.compile(r"^v[0-9][A-Za-z0-9_-]*$")


@click.group()
def main():
    """oxq — Agentic Quant Research Kernel CLI."""


@main.group(name="operator")
def operator_group() -> None:
    """Certify external operator providers."""


@operator_group.command(name="certify-provider")
@click.option(
    "--provider-repo",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help="Existing local Git repository containing the provider submission.",
)
@click.option(
    "--provider-commit",
    required=True,
    help="Exact lowercase 40-character provider submission commit.",
)
@click.option(
    "--artifact-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    default=None,
    help="Local wheel directory; defaults to PROVIDER_REPO/dist.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Certification root; defaults to .open-xquant/certifications in the current directory.",
)
@click.option(
    "--trust-provider-code",
    is_flag=True,
    help="Acknowledge that provider wheels execute as trusted local code.",
)
@click.option(
    "--baseline-timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Seconds allowed for each isolated numerical baseline execution.",
)
@click.option(
    "--target",
    default=None,
    help="Optional canonical Python-ABI-platform target for v2 certification.",
)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def certify_provider_command(
    provider_repo: Path,
    provider_commit: str,
    artifact_dir: Path | None,
    output_dir: Path | None,
    trust_provider_code: bool,
    baseline_timeout: float,
    target: str | None,
    as_json: bool,
) -> None:
    """Certify one exact local provider submission for research use."""
    from oxq.operators.errors import OperatorCertificationError

    known_identity: tuple[str, str] | None = None
    try:
        if not trust_provider_code:
            raise OperatorCertificationError(
                "provider_code_trust_required",
                "--trust-provider-code is required to execute provider wheels",
                stage="trust",
            )

        from oxq.operators.models import CertificationTarget

        try:
            certification_target = (
                CertificationTarget.parse(target) if target is not None else None
            )
        except ValueError as exc:
            raise OperatorCertificationError(
                "certification_target_invalid",
                str(exc),
                stage="target",
            ) from None
        from oxq.operators.certification import certify_provider
        from oxq.operators.registry import publish_certification
        from oxq.operators.submission import load_provider_submission

        resolved_artifact_dir = (
            artifact_dir if artifact_dir is not None else provider_repo / "dist"
        )
        resolved_output_dir = (
            output_dir
            if output_dir is not None
            else Path.cwd() / ".open-xquant" / "certifications"
        )
        with load_provider_submission(
            provider_repo,
            provider_commit,
            resolved_artifact_dir,
        ) as submission:
            known_identity = (submission.provider, submission.release)
            certified = certify_provider(
                submission,
                baseline_timeout_seconds=baseline_timeout,
            )
            published = publish_certification(
                certified,
                resolved_output_dir,
                target=certification_target,
            )
    except OperatorCertificationError as error:
        if as_json:
            payload = error.as_dict()
            if known_identity is not None:
                payload["provider"], payload["release"] = known_identity
            click.echo(json.dumps(payload, sort_keys=True))
        else:
            identity = (
                ""
                if known_identity is None
                else f" for {known_identity[0]} {known_identity[1]}"
            )
            click.echo(
                f"Certification failed{identity}: "
                f"[{error.stage}/{error.code}] {error.message}"
            )
        raise click.exceptions.Exit(1) from None

    payload = {
        "status": "research-certified",
        "provider": certified.provider,
        "release": certified.release,
        "submission_commit": certified.submission_commit,
        "source_commit": certified.source_commit,
        "operator_count": len(certified.operators),
        "output": str(published.release_dir),
    }
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    click.echo("Status: research-certified")
    click.echo(f"Provider: {certified.provider}")
    click.echo(f"Release: {certified.release}")
    click.echo(f"Operators: {len(certified.operators)}")
    click.echo(f"Output: {published.release_dir}")


@operator_group.command(name="install")
@click.argument("requirement")
def operator_install_command(requirement: str) -> None:
    """Show provider package installation guidance."""
    click.echo("Install provider package with:")
    click.echo(f"pip install {requirement}")
    click.echo("Then run:")
    click.echo(f"oxq operator verify {requirement}")
    raise click.exceptions.Exit(1)


@operator_group.command(name="verify")
@click.argument("requirement")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def operator_verify_command(requirement: str, as_json: bool) -> None:
    """Verify an installed certified environment provider package."""
    from oxq.operators.environment_provider import verify_installed_provider
    from oxq.operators.errors import OperatorCertificationError

    try:
        installed = verify_installed_provider(requirement)
    except OperatorCertificationError as error:
        _operator_environment_error(error, as_json)

    payload = _operator_environment_payload(installed)
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    click.echo(f"{payload['provider']}=={payload['version']} verified")
    click.echo(f"Status: {payload['status']}")
    click.echo(f"Operators: {payload['operator_count']}")


@operator_group.command(name="list")
@click.option("--provider", "provider_name", required=True, help="Canonical provider identifier.")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def operator_list_command(provider_name: str, as_json: bool) -> None:
    """List verified installed certified environment provider packages."""
    from oxq.operators.environment_index import _load_index_payload
    from oxq.operators.environment_provider import verify_installed_provider
    from oxq.operators.errors import OperatorCertificationError

    try:
        payload = _load_index_payload()
        providers = payload.get("providers")
        if not isinstance(providers, dict):
            raise OperatorCertificationError(
                "environment_provider_index_invalid",
                "official environment provider index is invalid",
                stage="environment_provider",
            )
        versions = providers.get(provider_name)
        if not isinstance(versions, dict):
            raise OperatorCertificationError(
                "environment_provider_invalid",
                "environment provider is not officially supported",
                stage="environment_provider",
            )

        verified = []
        last_error: OperatorCertificationError | None = None
        for version in sorted(versions):
            if not isinstance(version, str):
                continue
            try:
                verified.append(
                    _operator_environment_payload(
                        verify_installed_provider(f"{provider_name}=={version}")
                    )
                )
            except OperatorCertificationError as error:
                last_error = error
        if not verified:
            raise last_error or OperatorCertificationError(
                "environment_provider_not_installed",
                f"environment provider distribution is not installed: {provider_name}",
                stage="environment_provider",
            )
    except OperatorCertificationError as error:
        _operator_environment_error(error, as_json)

    if as_json:
        if len(verified) == 1:
            click.echo(json.dumps(verified[0], sort_keys=True))
        else:
            click.echo(json.dumps({"providers": verified}, sort_keys=True))
        return
    for index, item in enumerate(verified):
        if index:
            click.echo()
        click.echo(f"Provider: {item['provider']}")
        click.echo(f"Version: {item['version']}")
        click.echo(f"Status: {item['status']}")
        click.echo(f"Operators: {item['operator_count']}")


@operator_group.command(name="export-certification")
@click.option("--provider", required=True, help="Canonical provider identifier.")
@click.option("--release", required=True, help="Exact provider release SemVer.")
@click.option("--registry-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--manifest-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--baseline-file",
    "baseline_files",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--target", required=True, help="Canonical Python-ABI-platform target.")
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def export_certification_command(
    provider: str,
    release: str,
    registry_dir: Path,
    manifest_dir: Path,
    baseline_files: tuple[Path, ...],
    target: str,
    output: Path,
    as_json: bool,
) -> None:
    """Export one validated local certification as a portable ZIP."""
    from oxq.operators.errors import OperatorCertificationError

    try:
        from oxq.operators.bundle import export_certification_bundle
        from oxq.operators.models import CertificationTarget

        resolved_registry = registry_dir.expanduser().resolve()
        resolved_output = output.expanduser().resolve()
        try:
            resolved_output.relative_to(resolved_registry)
        except ValueError:
            pass
        else:
            raise OperatorCertificationError(
                "bundle_output_invalid",
                "bundle output must be outside the source registry",
                stage="output",
            )
        try:
            certification_target = CertificationTarget.parse(target)
        except ValueError as exc:
            raise OperatorCertificationError("certification_target_invalid", str(exc), stage="target") from None
        bundle = export_certification_bundle(
            provider=provider,
            release=release,
            registry_dir=resolved_registry,
            manifest_dir=manifest_dir,
            baseline_files=baseline_files,
            target=certification_target,
            output_path=resolved_output,
        )
    except OperatorCertificationError as error:
        _bundle_cli_error(error, as_json)
    except (OSError, ValueError):
        _bundle_cli_error(OperatorCertificationError("bundle_export_failed", "certification bundle export failed", stage="export"), as_json)

    payload = {
        "bundle": str(bundle.bundle_path), "operator_count": bundle.operator_count,
        "provider": bundle.provider, "release": bundle.release,
        "status": "research-certified", "target": target,
    }
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    click.echo("Status: research-certified")
    click.echo(f"Provider: {bundle.provider}")
    click.echo(f"Release: {bundle.release}")
    click.echo(f"Operators: {bundle.operator_count}")
    click.echo(f"Output: {bundle.bundle_path}")


@operator_group.command(name="import-certification")
@click.option(
    "--bundle",
    "bundle_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--trust-unsigned-bundle",
    is_flag=True,
    help="Acknowledge that the unsigned bundle is locally trusted.",
)
@click.option(
    "--bundle-store",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional audit store for the original bundle ZIP.",
)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def import_certification_command(
    bundle_path: Path,
    output_dir: Path,
    trust_unsigned_bundle: bool,
    bundle_store: Path | None,
    as_json: bool,
) -> None:
    """Import one trusted portable certification into a local registry."""
    from oxq.operators.bundle import import_certification_bundle
    from oxq.operators.errors import OperatorCertificationError

    try:
        imported = import_certification_bundle(
            bundle_path, output_dir, trust_unsigned_bundle=trust_unsigned_bundle, bundle_store=bundle_store
        )
    except OperatorCertificationError as error:
        _bundle_cli_error(error, as_json)

    payload = {
        "output": str(imported.release_dir), "provider": imported.record["provider"],
        "release": imported.record["release"], "status": "research-certified",
    }
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    click.echo("Status: research-certified")
    click.echo(f"Provider: {imported.record['provider']}")
    click.echo(f"Release: {imported.record['release']}")
    click.echo(f"Output: {imported.release_dir}")


def _operator_environment_payload(installed) -> dict[str, object]:
    provider = installed.provider
    operators = [
        {
            "operator_id": operator.operator_id,
            "operator_version": operator.operator_version,
        }
        for operator in provider.operators
    ]
    return {
        "operator_count": len(operators),
        "operators": operators,
        "provider": provider.provider,
        "status": provider.certification_state,
        "version": provider.version,
    }


def _operator_environment_error(error, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(error.as_dict(), sort_keys=True))
    else:
        click.echo(
            f"Operator environment failed: "
            f"[{error.stage}/{error.code}] {error.message}"
        )
    raise click.exceptions.Exit(1)


def _bundle_cli_error(error, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(error.as_dict(), sort_keys=True))
    else:
        click.echo(f"Certification bundle failed: [{error.stage}/{error.code}] {error.message}")
    raise click.exceptions.Exit(1)


@main.group()
def spec():
    """Manage strategy specs."""


@spec.command()
@click.argument("description")
@click.option(
    "--out",
    "-o",
    default=None,
    help=(
        "Output file path. Defaults to versions/<active_version>/04_spec_build/strategy_spec.yaml "
        "inside version-governed research workspaces."
    ),
)
@click.option(
    "--market-preset",
    type=click.Choice(["us_equity", "cn_a_share"]),
    default="us_equity",
    help="Explicit template preset; generated values are candidates until user-confirmed in Agent workflows.",
)
def init(description: str, out: str | None, market_preset: str):
    """Initialize a new strategy spec from a natural language description.

    DESCRIPTION is a brief strategy idea in natural language.
    """
    strategy_id = make_strategy_id(description)
    template = StrategySpec.template(strategy_id=strategy_id, hypothesis=description, market_preset=market_preset)

    payload = template.to_dict()
    if market_preset == "cn_a_share":
        payload.setdefault("market", {})["asset_class"] = "equity"
        payload.setdefault("universe", {})["type"] = "static"
        payload.setdefault("universe", {})["point_in_time"] = False
        payload.setdefault("universe", {})["survivorship_bias_policy"] = "warn"
        payload.setdefault("data", {})["provider"] = "local"
        payload.setdefault("data", {})["price_adjustment"] = "adjusted"
        payload.setdefault("data", {})["required_columns"] = ["open", "high", "low", "close", "volume"]
        payload.setdefault("signal", {})["signal_time"] = "close_t"
        payload.setdefault("execution", {})["trade_time"] = "next_open"
        payload.setdefault("execution", {})["fill_price_mode"] = "next_open"
        payload.setdefault("execution", {})["cash_annual_return"] = 0.0
        payload.setdefault("execution", {})["initial_cash"] = 100000.0
        payload.setdefault("metrics", {})["profile"] = "open_xquant_default"
        payload.setdefault("metrics", {})["risk_free_rate"] = 0.0
        payload.setdefault("metrics", {})["return_type"] = "simple"
        payload.setdefault("metrics", {})["annualization_days"] = 252
        payload.setdefault("metrics", {})["calmar_denominator"] = "max_drawdown"
        payload.setdefault("metrics", {})["evaluation_window"] = "full"

    output_path = Path(out) if out else _default_spec_init_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False), encoding="utf-8")

    click.echo(f"Spec template written to {output_path}")
    click.echo(f"Strategy ID: {strategy_id}")
    click.echo("Template preset values are candidate values; Agent workflows must still collect user confirmation.")
    click.echo("Next: edit the file, then run `oxq spec validate`")


def _default_spec_init_output_path() -> Path:
    active_state = _active_workspace_state()
    if active_state is not None:
        version, versions_dir = active_state
        version_dir = _resolve_active_version_dir(Path.cwd(), versions_dir, version)
        manifest_path = version_dir / "version_manifest.json"
        if not manifest_path.exists():
            raw_spec_dir = (version_dir / "04_spec_build").relative_to(Path.cwd()).as_posix()
            spec_dir = _resolve_version_phase_path(
                Path.cwd(),
                version_dir,
                "04_spec_build",
                raw_spec_dir,
            )
            return spec_dir / "strategy_spec.yaml"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise click.ClickException(
                "active version_manifest.json must contain a valid JSON object"
            ) from exc
        if not isinstance(manifest, dict) or manifest.get("version_id") != version:
            raise click.ClickException(
                "active version_manifest.json must match current.json active_version"
            )
        phase_paths = manifest.get("phase_paths")
        raw_spec_dir = phase_paths.get("04_spec_build") if isinstance(phase_paths, dict) else None
        if not isinstance(raw_spec_dir, str) or not raw_spec_dir:
            raise click.ClickException(
                "active version_manifest.json requires phase_paths.04_spec_build"
            )
        spec_dir = _resolve_version_phase_path(
            Path.cwd(),
            version_dir,
            "04_spec_build",
            raw_spec_dir,
        )
        return spec_dir / "strategy_spec.yaml"
    return Path("strategy_spec.yaml")


def _active_workspace_state() -> tuple[str, Path] | None:
    workspace_file = Path(".open-xquant") / "workspace.yaml"
    if not workspace_file.exists():
        return None
    workspace = _read_workspace_config(workspace_file)
    if not _is_version_governed_workspace(workspace):
        return None
    versions_dir = _workspace_versions_dir(workspace)
    paths = workspace.get("paths")
    current_manifest = "current.json"
    if isinstance(paths, dict) and isinstance(paths.get("current_manifest"), str):
        configured_manifest = Path(paths["current_manifest"])
        invalid_manifest_path = (
            configured_manifest.is_absolute()
            or len(configured_manifest.parts) != 1
            or configured_manifest.name != "current.json"
        )
        if invalid_manifest_path:
            raise click.ClickException(
                "version-governed workspace requires root current.json active_version; "
                "run `oxq research init` to repair manifests"
            )
        current_manifest = paths["current_manifest"]
    manifest_path = Path(current_manifest)
    if manifest_path.is_absolute() or ".." in manifest_path.parts:
        raise click.ClickException(
            "version-governed workspace requires root current.json active_version; "
            "run `oxq research init` to repair manifests"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise click.ClickException(
            "version-governed workspace requires root current.json active_version; "
            "run `oxq research init` to repair manifests"
        )
    if not isinstance(payload, dict):
        raise click.ClickException(
            "version-governed workspace requires root current.json active_version; "
            "run `oxq research init` to repair manifests"
        )
    version = payload.get("active_version")
    if isinstance(version, str) and _WORKSPACE_VERSION_RE.fullmatch(version):
        return version, versions_dir
    raise click.ClickException(
        "version-governed workspace requires a safe current.json active_version; "
        "run `oxq research init` to repair manifests"
    )


@spec.command()
@click.argument("spec_file", type=click.Path(exists=True))
@click.option(
    "--component-manifest",
    "component_manifest",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Workspace component manifest to load before validation.",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def validate(spec_file: str, component_manifest: tuple[str, ...], as_json: bool):
    """Validate a strategy spec file.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    try:
        _load_component_manifests(component_manifest)
        parsed = StrategySpec.from_yaml(spec_file)
    except Exception as e:
        result = {
            "status": "fail",
            "errors": [{"severity": "fatal", "check": "parse_error", "message": str(e)}],
            "warnings": [],
            "spec_hash": "",
        }
        if as_json:
            import json

            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"FAIL: {e}")
        raise SystemExit(1)

    result = validate_spec(parsed)

    if as_json:
        import json

        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(f"Status: {result.status.upper()}")
        click.echo(f"Spec Hash: {result.spec_hash}")
        if result.errors:
            click.echo(f"\nErrors ({len(result.errors)}):")
            for e in result.errors:
                click.echo(f"  [{e['severity']}] {e['check']}: {e['message']}")
        if result.warnings:
            click.echo(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                click.echo(f"  [{w['severity']}] {w['check']}: {w['message']}")
        if result.status == "pass":
            click.echo("\nSpec is valid.")

    if result.status == "fail":
        raise SystemExit(1)


@spec.command(name="hash")
@click.argument("spec_file", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def spec_hash(spec_file: str, as_json: bool):
    """Compute the canonical strategy spec hash."""
    parsed = StrategySpec.from_yaml(spec_file)
    digest = parsed.compute_hash()
    if as_json:
        click.echo(json.dumps({"spec_hash": digest}, indent=2))
    else:
        click.echo(digest)


@spec.command(name="fields")
@click.argument("spec_file", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def spec_fields(spec_file: str, as_json: bool):
    """Export deterministic flattened fields from a strategy spec."""
    parsed = StrategySpec.from_yaml(spec_file)
    fields = [{"path": path, "value": value} for path, value in _flatten_fields(parsed.to_effective_dict())]
    if as_json:
        click.echo(json.dumps({"spec_hash": parsed.compute_hash(), "fields": fields}, indent=2, ensure_ascii=False, default=str))
        return
    for item in fields:
        click.echo(f"{item['path']}={json.dumps(item['value'], ensure_ascii=False, sort_keys=True, default=str)}")


def _flatten_fields(value: object, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        rows: list[tuple[str, object]] = []
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_fields(value[key], path))
        return rows
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return [(prefix, value)]
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten_fields(item, f"{prefix}[{index}]"))
        return rows
    return [(prefix, value)]


def _load_component_manifests(
    manifest_paths: tuple[str, ...],
    *,
    source_manifest_paths: tuple[str, ...] | None = None,
) -> list[dict]:
    """Load workspace component manifests and annotate them for catalog export."""
    if not manifest_paths:
        return []
    if source_manifest_paths is not None and len(source_manifest_paths) != len(manifest_paths):
        raise ValueError("source component manifest paths must match staged manifest paths")
    from oxq.core.component_manifest import load_component_manifest, snapshot_component_registries

    restore_registries = snapshot_component_registries()
    ctx = click.get_current_context(silent=True)
    if ctx is not None:
        ctx.call_on_close(restore_registries)

    manifests: list[dict] = []
    for index, raw_path in enumerate(manifest_paths):
        loaded = load_component_manifest(raw_path, verify_hash=True)
        loaded["_manifest_path"] = str(Path(raw_path).resolve())
        if source_manifest_paths is not None:
            loaded["_source_manifest_path"] = str(Path(source_manifest_paths[index]).resolve())
        manifests.append(loaded)
    return manifests


def _read_component_manifest_payloads(manifest_paths: tuple[str, ...]) -> list[dict]:
    """Read and hash workspace component manifests without importing component code."""
    if not manifest_paths:
        return []
    from oxq.core.component_manifest import component_manifest_summary

    manifests: list[dict] = []
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path).resolve()
        summary = component_manifest_summary(manifest_path)
        if summary["status"] != "pass":
            raise click.ClickException(
                "component manifest bundle hash mismatch: "
                f"stored={summary['bundle_hash']}, actual={summary['computed_bundle_hash']}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise click.ClickException(f"component manifest must be a JSON object: {manifest_path}")
        payload["_manifest_path"] = str(manifest_path)
        manifests.append(payload)
    return manifests


def _stage_component_manifest_snapshots(
    manifests: list[dict],
) -> tuple[tempfile.TemporaryDirectory[str], tuple[str, ...], list[dict]]:
    """Copy verified bundles into private snapshots used for all later imports."""
    staging = tempfile.TemporaryDirectory(prefix="oxq-component-stage-")
    staging_root = Path(staging.name)
    try:
        staged_paths: list[str] = []
        for index, manifest in enumerate(manifests):
            archived = _archive_component_extension(staging_root, manifest, index)
            if archived is None:
                raise click.ClickException("component bundle could not be staged from its verified manifest")
            archived_manifest, _ = archived
            staged_paths.append(str(staging_root / archived_manifest))
        staged_payloads = _read_component_manifest_payloads(tuple(staged_paths))
        expected_hashes = [manifest.get("bundle_hash") for manifest in manifests]
        staged_hashes = [manifest.get("bundle_hash") for manifest in staged_payloads]
        if staged_hashes != expected_hashes:
            raise click.ClickException(
                "component bundle changed while its authorized snapshot was being staged"
            )
        return staging, tuple(staged_paths), staged_payloads
    except BaseException:
        staging.cleanup()
        raise


def _require_component_sources_match_staged(
    source_manifest_paths: tuple[str, ...],
    staged_manifests: list[dict],
) -> None:
    """Detect a source replacement after authorization without importing it."""
    from oxq.core.component_manifest import component_manifest_summary

    if len(source_manifest_paths) != len(staged_manifests):
        raise click.ClickException("authorized component manifests do not match staged snapshots")
    for raw_path, staged in zip(source_manifest_paths, staged_manifests, strict=True):
        expected = staged.get("bundle_hash")
        try:
            summary = component_manifest_summary(raw_path)
        except (OSError, ValueError) as exc:
            raise click.ClickException(
                f"component bundle changed after its authorized snapshot was staged: {raw_path}: {exc}"
            ) from exc
        if summary.get("status") != "pass" or summary.get("bundle_hash") != expected:
            raise click.ClickException(
                "component bundle changed after its authorized snapshot was staged: "
                f"path={raw_path}, authorized={expected}, current={summary.get('computed_bundle_hash')}"
            )


def _write_run_component_manifest_artifacts(
    run_dir: Path,
    manifests: list[dict],
    *,
    update_artifact_hashes: bool = True,
) -> None:
    _preflight_component_extension_archives(run_dir, manifests)
    with tempfile.TemporaryDirectory(prefix="oxq-component-publish-") as staging_dir:
        staging_root = Path(staging_dir) / "run"
        staging_root.mkdir()
        archived_paths: dict[int, tuple[str, str]] = {}
        for index, manifest in enumerate(manifests):
            archived = _archive_component_extension(staging_root, manifest, index)
            if archived is not None:
                archived_paths[index] = archived

        artifacts = _component_manifest_artifact_contents(
            staging_root,
            manifests,
            archived_paths,
        )
        if not update_artifact_hashes:
            shutil.copytree(staging_root, run_dir, dirs_exist_ok=True)
            for name, content in artifacts.items():
                (run_dir / name).write_bytes(content)
            return

        replacement_names = _existing_component_replacement_names(run_dir)
        replacement_names.update(path.name for path in staging_root.iterdir())
        replacement_paths = {
            name: (staging_root / name if (staging_root / name).exists() else None)
            for name in sorted(replacement_names)
        }
        component_artifact_names = {
            "component_manifests.json",
            "component_manifest.json",
            "component_bundle_hash.txt",
        }
        remove_artifacts = {
            name
            for name in component_artifact_names - artifacts.keys()
            if (run_dir / name).exists() or (run_dir / name).is_symlink()
        }
        try:
            publish_run_artifacts(
                run_dir,
                artifacts,
                replacement_paths=replacement_paths,
                remove_artifacts=remove_artifacts,
            )
        except RunDigestError as exc:
            raise click.ClickException(str(exc)) from exc


class _PinnedCompilePreviewParent:
    def __init__(self, path: Path, descriptor: int, identity: str) -> None:
        self.path = path
        self.descriptor = descriptor
        self.identity = identity

    @classmethod
    def open(cls, path: Path, *, expected_identity: str) -> _PinnedCompilePreviewParent:
        if os.name != "posix" or not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
            raise click.ClickException(
                "compile preview publication requires stable relative directory operations on this platform"
            )
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise click.ClickException(f"cannot pin compile preview output parent: {path}") from exc
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise click.ClickException(f"compile preview output parent is unsafe: {path}")
            identity = _compile_preview_status_identity(status)
            if identity != expected_identity:
                raise click.ClickException("compile preview output parent changed during publication")
            return cls(path, descriptor, identity)
        except BaseException:
            os.close(descriptor)
            raise

    def __enter__(self) -> _PinnedCompilePreviewParent:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        os.close(self.descriptor)

    def require_path_identity(self) -> None:
        _require_compile_preview_parent_identity(self.path, self.identity)

    def status(self, name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def identity_for(self, name: str) -> str | None:
        status = self.status(name)
        return None if status is None else _compile_preview_status_identity(status)

    def directory_identity(self, name: str) -> str:
        status = self.status(name)
        if status is None or not stat.S_ISDIR(status.st_mode):
            raise click.ClickException(
                f"compile preview directory identity is unsafe: {self.path / name}"
            )
        return _compile_preview_status_identity(status)

    def open_directory(self, name: str) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        return os.open(name, flags, dir_fd=self.descriptor)

    def replace(self, source: str, destination: str) -> None:
        os.replace(
            source,
            destination,
            src_dir_fd=self.descriptor,
            dst_dir_fd=self.descriptor,
        )

    def unlink(self, name: str) -> None:
        os.unlink(name, dir_fd=self.descriptor)

    def fsync(self) -> None:
        os.fsync(self.descriptor)


def _compile_preview_status_identity(status: os.stat_result) -> str:
    return f"posix:{int(status.st_dev)}:{int(status.st_ino)}"


def _publish_compile_preview(
    out_dir: Path,
    artifacts: dict[str, bytes],
    component_manifests: list[dict],
) -> None:
    if os.name != "posix":
        raise click.ClickException(
            "compile preview publication requires stable relative directory operations on this platform"
        )
    target = Path(os.path.abspath(out_dir))
    _require_no_symlink_components(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = _compile_preview_parent_identity(target.parent)
    with _PinnedCompilePreviewParent.open(
        target.parent,
        expected_identity=parent_identity,
    ) as parent:
        staging_name, staging_identity = _create_compile_preview_staging(parent, target.name)
        staging_root = target.parent / staging_name
        try:
            parent.require_path_identity()
            for name, content in artifacts.items():
                (staging_root / name).write_bytes(content)
            if component_manifests:
                _write_run_component_manifest_artifacts(
                    staging_root,
                    component_manifests,
                    update_artifact_hashes=False,
                )
            (staging_root / _COMPILE_PREVIEW_MARKER_NAME).write_text(
                json.dumps(_COMPILE_PREVIEW_MARKER, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            parent.require_path_identity()
            _replace_compile_preview(
                target,
                staging_name,
                staging_identity=staging_identity,
                parent=parent,
            )
        finally:
            if parent.identity_for(staging_name) == staging_identity:
                _remove_compile_preview_directory(
                    parent,
                    staging_name,
                    staging_identity,
                    label="staging directory",
                )


def _create_compile_preview_staging(
    parent: _PinnedCompilePreviewParent,
    target_name: str,
) -> tuple[str, str]:
    while True:
        staging_name = f".{target_name}.stage-{secrets.token_hex(8)}"
        try:
            os.mkdir(staging_name, mode=0o700, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        parent.fsync()
        return staging_name, parent.directory_identity(staging_name)


def _replace_compile_preview(
    target: Path,
    staging_name: str,
    *,
    staging_identity: str,
    parent: _PinnedCompilePreviewParent,
) -> None:
    parent.require_path_identity()
    lock_path = target.with_name(f".{target.name}.lock")
    lock_status = parent.status(lock_path.name)
    if lock_status is not None and not stat.S_ISREG(lock_status.st_mode):
        raise click.ClickException(f"compile preview lock path is unsafe: {lock_path}")
    with ProcessFileLock(lock_path):
        parent.require_path_identity()
        _recover_compile_preview_replacement(target, parent=parent)
        _preflight_compile_preview_target(target, parent=parent)
        previous_kind = _compile_preview_path_kind_at(parent, target.name)
        had_target = previous_kind is not None
        if had_target and previous_kind not in {"empty", "managed"}:
            raise click.ClickException(f"compile preview output is not safely replaceable: {target}")
        backup_name = _unused_compile_preview_backup_name(parent, target.name)
        transaction = {
            "schema_version": _COMPILE_PREVIEW_TRANSACTION_SCHEMA_VERSION,
            "transaction_type": _COMPILE_PREVIEW_TRANSACTION_TYPE,
            "transaction_id": secrets.token_hex(16),
            "phase": "prepared",
            "target": target.name,
            "staging": staging_name,
            "staging_identity": staging_identity,
            "backup": backup_name,
            "had_target": had_target,
            "previous_kind": previous_kind,
            "parent_identity": parent.identity,
        }
        _fsync_compile_preview_tree(parent, staging_name, staging_identity)
        _write_compile_preview_transaction(target, transaction, parent=parent)
        try:
            if had_target:
                _replace_compile_preview_sibling(parent, target.name, backup_name)
                transaction["phase"] = "backup_created"
                _write_compile_preview_transaction(target, transaction, parent=parent)
            _replace_compile_preview_sibling(parent, staging_name, target.name)
            if parent.directory_identity(target.name) != staging_identity:
                raise click.ClickException("compile preview staging generation changed during installation")
            transaction["phase"] = "installed"
            _write_compile_preview_transaction(target, transaction, parent=parent)
        except BaseException:
            try:
                _rollback_compile_preview_replacement(
                    target,
                    transaction,
                    parent=parent,
                )
            except BaseException:
                pass
            raise
        _finish_compile_preview_replacement(
            target,
            transaction,
            parent=parent,
        )


def _replace_compile_preview_sibling(
    parent: _PinnedCompilePreviewParent,
    source: str,
    destination: str,
) -> None:
    parent.require_path_identity()
    parent.replace(source, destination)
    parent.fsync()
    parent.require_path_identity()


def _compile_preview_transaction_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.oxq-preview-transaction.json")


def _unused_compile_preview_backup_name(
    parent: _PinnedCompilePreviewParent,
    target_name: str,
) -> str:
    while True:
        candidate = f".{target_name}.oxq-preview-old-{secrets.token_hex(8)}"
        if parent.status(candidate) is None:
            return candidate


def _compile_preview_quarantine_name(target_name: str, transaction_id: str) -> str:
    return f".{target_name}.oxq-preview-cleanup-{transaction_id}"


def _write_compile_preview_transaction(
    target: Path,
    payload: dict[str, object],
    *,
    parent: _PinnedCompilePreviewParent,
) -> None:
    if payload.get("parent_identity") != parent.identity:
        raise click.ClickException("compile preview transaction parent identity changed")
    transaction_name = _compile_preview_transaction_path(target).name
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_name = f".{transaction_name}.write-{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent.descriptor)
    temporary_identity = _compile_preview_status_identity(os.fstat(descriptor))
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        parent.replace(temporary_name, transaction_name)
        parent.fsync()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent.identity_for(temporary_name) == temporary_identity:
            parent.unlink(temporary_name)
            parent.fsync()


def _read_compile_preview_transaction(
    target: Path,
    *,
    parent: _PinnedCompilePreviewParent,
) -> dict[str, object] | None:
    transaction_name = _compile_preview_transaction_path(target).name
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(transaction_name, flags, dir_fd=parent.descriptor)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise click.ClickException(
            "Invalid pending compile preview transaction: transaction record must be a regular file"
        ) from exc
    try:
        transaction_status = os.fstat(descriptor)
        if not stat.S_ISREG(transaction_status.st_mode):
            raise ValueError("transaction record must be a regular file")
        if transaction_status.st_nlink != 1:
            raise ValueError("transaction record must not have multiple hard links")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            content = stream.read()
        payload = json.loads(content.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("transaction record must be an object")
        if payload.get("schema_version") != _COMPILE_PREVIEW_TRANSACTION_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if payload.get("transaction_type") != _COMPILE_PREVIEW_TRANSACTION_TYPE:
            raise ValueError("unexpected transaction_type")
        transaction_id = payload.get("transaction_id")
        if (
            not isinstance(transaction_id, str)
            or len(transaction_id) != 32
            or any(character not in "0123456789abcdef" for character in transaction_id)
        ):
            raise ValueError("invalid transaction_id")
        phase = payload.get("phase")
        if phase not in {"prepared", "backup_created", "installed", "backup_cleanup"}:
            raise ValueError("unexpected transaction phase")
        target_name = _validated_compile_preview_target_name(
            payload.get("target"),
            parent,
            _compile_preview_status_identity(transaction_status),
        )
        staging = _validated_compile_preview_sibling(
            payload.get("staging"),
            prefix=f".{target_name}.stage-",
            label="staging",
        )
        backup = _validated_compile_preview_sibling(
            payload.get("backup"),
            prefix=f".{target_name}.oxq-preview-old-",
            label="backup",
        )
        staging_identity = payload.get("staging_identity")
        if staging_identity is not None and (
            not isinstance(staging_identity, str) or not staging_identity
        ):
            raise ValueError("staging_identity must be a non-empty string")
        quarantine_raw = payload.get("backup_quarantine")
        quarantine = None
        if quarantine_raw is not None:
            quarantine = _validated_compile_preview_sibling(
                quarantine_raw,
                prefix=f".{target_name}.oxq-preview-cleanup-",
                label="backup quarantine",
            )
        if len({name for name in (staging, backup, quarantine) if name is not None}) != (
            3 if quarantine is not None else 2
        ):
            raise ValueError("transaction-owned paths overlap")
        had_target = payload.get("had_target")
        previous_kind = payload.get("previous_kind")
        if not isinstance(had_target, bool):
            raise ValueError("had_target must be a boolean")
        if had_target and previous_kind not in {"empty", "managed"}:
            raise ValueError("previous_kind does not identify a replaceable preview")
        if not had_target and previous_kind is not None:
            raise ValueError("previous_kind must be null when no target existed")
        if payload.get("parent_identity") != parent.identity:
            raise ValueError("transaction parent identity does not match the output parent")
        cleanup_identity = payload.get("backup_cleanup_identity")
        cleanup_kind = payload.get("backup_cleanup_kind")
        if phase == "backup_cleanup":
            if not had_target:
                raise ValueError("backup cleanup requires a previously installed preview")
            if not isinstance(cleanup_identity, str) or not cleanup_identity:
                raise ValueError("backup cleanup identity must be a non-empty string")
            if cleanup_kind != previous_kind:
                raise ValueError("backup cleanup ownership kind does not match the prior preview")
        elif cleanup_identity is not None or cleanup_kind is not None or quarantine is not None:
            raise ValueError("backup cleanup evidence requires the backup_cleanup phase")
        return payload
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise click.ClickException(f"Invalid pending compile preview transaction: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validated_compile_preview_target_name(
    raw_name: object,
    parent: _PinnedCompilePreviewParent,
    requested_transaction_identity: str,
) -> str:
    target_name = _validated_compile_preview_sibling_name(raw_name, label="target")
    recorded_name = f".{target_name}.oxq-preview-transaction.json"
    recorded_status = parent.status(recorded_name)
    if recorded_status is None or not stat.S_ISREG(recorded_status.st_mode):
        raise ValueError("transaction target does not match the requested preview")
    if _compile_preview_status_identity(recorded_status) != requested_transaction_identity:
        raise ValueError("transaction target does not match the requested preview")
    return target_name


def _validated_compile_preview_sibling(
    raw_name: object,
    *,
    prefix: str,
    label: str,
) -> str:
    name = _validated_compile_preview_sibling_name(raw_name, label=label)
    suffix = name.removeprefix(prefix)
    if (
        not name.startswith(prefix)
        or not suffix
        or any(not (character.isascii() and (character.isalnum() or character in "_-")) for character in suffix)
    ):
        raise ValueError(f"transaction {label} path is not transaction-owned")
    return name


def _validated_compile_preview_sibling_name(raw_name: object, *, label: str) -> str:
    if not isinstance(raw_name, str):
        raise ValueError(f"transaction {label} path must be a string")
    path = Path(raw_name)
    if not raw_name or path.is_absolute() or path.parent != Path() or path.name != raw_name:
        raise ValueError(f"transaction {label} path must be a relative sibling name")
    return raw_name


def _recover_compile_preview_replacement(
    target: Path,
    *,
    parent: _PinnedCompilePreviewParent,
) -> bool:
    transaction = _read_compile_preview_transaction(target, parent=parent)
    if transaction is None:
        return False
    staging_name = str(transaction["staging"])
    backup_name = str(transaction["backup"])
    had_target = bool(transaction["had_target"])
    previous_kind = transaction.get("previous_kind")
    phase = str(transaction["phase"])
    staging_identity = transaction.get("staging_identity")

    target_kind = _compile_preview_path_kind_at(parent, target.name)
    staging_kind = _compile_preview_path_kind_at(parent, staging_name)
    backup_kind = _compile_preview_path_kind_at(parent, backup_name)
    if target_kind == "unrecognized":
        raise click.ClickException(
            f"Pending compile preview recovery refused an unrecognized output target: {target}"
        )
    if staging_kind not in {None, "managed"}:
        raise click.ClickException(
            "Pending compile preview recovery refused an unrecognized staging directory: "
            f"{target.parent / staging_name}"
        )
    if phase == "backup_cleanup":
        if target_kind != "managed" or staging_kind is not None:
            raise click.ClickException(
                "Pending compile preview backup cleanup does not contain the installed managed preview"
            )
        _finish_compile_preview_replacement(
            target,
            transaction,
            parent=parent,
        )
        return True
    if backup_kind is not None and backup_kind != previous_kind:
        raise click.ClickException(
            "Pending compile preview recovery refused an unrecognized backup directory: "
            f"{target.parent / backup_name}"
        )

    target_identity = (
        parent.directory_identity(target.name)
        if target_kind in {"empty", "managed"}
        else None
    )
    installed_before_phase_update = (
        target_kind == "managed"
        and staging_kind is None
        and isinstance(staging_identity, str)
        and target_identity == staging_identity
        and (
            (not had_target and phase == "prepared" and backup_kind is None)
            or (had_target and phase == "backup_created" and backup_kind == previous_kind)
        )
    )
    if installed_before_phase_update:
        transaction["phase"] = "installed"
        _write_compile_preview_transaction(target, transaction, parent=parent)
        _finish_compile_preview_replacement(target, transaction, parent=parent)
        return True

    if target_kind is None:
        if staging_kind == "managed" and (not had_target or backup_kind == previous_kind):
            if isinstance(staging_identity, str) and parent.directory_identity(staging_name) != staging_identity:
                raise click.ClickException("Pending compile preview staging generation identity changed")
            _replace_compile_preview_sibling(parent, staging_name, target.name)
            if isinstance(staging_identity, str) and parent.directory_identity(target.name) != staging_identity:
                raise click.ClickException("Pending compile preview installed generation identity changed")
            transaction["phase"] = "installed"
            _write_compile_preview_transaction(target, transaction, parent=parent)
        elif backup_kind == previous_kind and had_target:
            _replace_compile_preview_sibling(parent, backup_name, target.name)
            _remove_compile_preview_transaction(target, transaction, parent=parent)
            return True
        else:
            raise click.ClickException("Pending compile preview transaction cannot be recovered safely")
    elif staging_kind == "managed":
        if had_target and backup_kind is None and target_kind == previous_kind and phase == "prepared":
            _replace_compile_preview_sibling(parent, target.name, backup_name)
            transaction["phase"] = "backup_created"
            _write_compile_preview_transaction(target, transaction, parent=parent)
            if isinstance(staging_identity, str) and parent.directory_identity(staging_name) != staging_identity:
                raise click.ClickException("Pending compile preview staging generation identity changed")
            _replace_compile_preview_sibling(parent, staging_name, target.name)
            transaction["phase"] = "installed"
            _write_compile_preview_transaction(target, transaction, parent=parent)
        else:
            raise click.ClickException("Pending compile preview transaction has conflicting target and staging directories")
    elif target_kind != "managed" or phase != "installed":
        raise click.ClickException("Pending compile preview transaction does not contain a managed installed preview")
    elif isinstance(staging_identity, str) and target_identity != staging_identity:
        raise click.ClickException("Pending compile preview installed generation identity changed")

    _finish_compile_preview_replacement(target, transaction, parent=parent)
    return True


def _rollback_compile_preview_replacement(
    target: Path,
    transaction: dict[str, object],
    *,
    parent: _PinnedCompilePreviewParent,
) -> None:
    staging_name = str(transaction["staging"])
    backup_name = str(transaction["backup"])
    had_target = bool(transaction["had_target"])
    previous_kind = transaction.get("previous_kind")
    staging_identity = transaction.get("staging_identity")
    backup_kind = _compile_preview_path_kind_at(parent, backup_name)
    target_kind = _compile_preview_path_kind_at(parent, target.name)

    if backup_kind is not None:
        if backup_kind != previous_kind:
            raise click.ClickException(
                "Refusing to roll back an unrecognized compile preview backup: "
                f"{target.parent / backup_name}"
            )
        if target_kind is not None:
            _remove_owned_compile_preview_path(
                parent,
                target.name,
                "managed",
                expected_identity=staging_identity if isinstance(staging_identity, str) else None,
            )
        parent.replace(backup_name, target.name)
        parent.fsync()
    elif had_target:
        if target_kind != previous_kind:
            raise click.ClickException("Compile preview target changed while rollback was pending")
    elif target_kind is not None:
        _remove_owned_compile_preview_path(
            parent,
            target.name,
            "managed",
            expected_identity=staging_identity if isinstance(staging_identity, str) else None,
        )

    if _compile_preview_path_kind_at(parent, staging_name) is not None:
        _remove_owned_compile_preview_path(
            parent,
            staging_name,
            "managed",
            expected_identity=staging_identity if isinstance(staging_identity, str) else None,
        )
    _remove_compile_preview_transaction(target, transaction, parent=parent)


def _finish_compile_preview_replacement(
    target: Path,
    transaction: dict[str, object],
    *,
    parent: _PinnedCompilePreviewParent,
) -> None:
    parent.require_path_identity()
    if _compile_preview_path_kind_at(parent, target.name) != "managed":
        raise click.ClickException(f"Refusing to finalize an unrecognized compile preview: {target}")
    staging_identity = transaction.get("staging_identity")
    if (
        isinstance(staging_identity, str)
        and parent.directory_identity(target.name) != staging_identity
    ):
        raise click.ClickException("Refusing to finalize a changed compile preview generation")
    backup_name = str(transaction["backup"])
    phase = transaction.get("phase")
    if phase not in {"installed", "backup_cleanup"}:
        raise click.ClickException("Compile preview replacement is not ready for backup cleanup")

    if phase == "installed":
        backup_kind = _compile_preview_path_kind_at(parent, backup_name)
        if backup_kind is None:
            _remove_compile_preview_transaction(target, transaction, parent=parent)
            parent.require_path_identity()
            return
        previous_kind = transaction.get("previous_kind")
        if backup_kind != previous_kind:
            raise click.ClickException(
                "Refusing to delete an unrecognized compile preview backup: "
                f"{target.parent / backup_name}"
            )
        transaction_id = str(transaction["transaction_id"])
        quarantine_name = _compile_preview_quarantine_name(
            str(transaction["target"]),
            transaction_id,
        )
        transaction["phase"] = "backup_cleanup"
        transaction["backup_cleanup_identity"] = _compile_preview_directory_identity_in_parent(
            parent,
            backup_name,
        )
        transaction["backup_cleanup_kind"] = previous_kind
        transaction["backup_quarantine"] = quarantine_name
        _write_compile_preview_transaction(target, transaction, parent=parent)
    else:
        quarantine_name = transaction.get("backup_quarantine")
        if quarantine_name is None:
            quarantine_name = _compile_preview_quarantine_name(
                str(transaction["target"]),
                str(transaction["transaction_id"]),
            )
            transaction["backup_quarantine"] = quarantine_name
            _write_compile_preview_transaction(target, transaction, parent=parent)
        quarantine_name = str(quarantine_name)

    expected_identity = str(transaction["backup_cleanup_identity"])
    backup_identity = parent.identity_for(backup_name)
    quarantine_identity = parent.identity_for(quarantine_name)
    if quarantine_identity is not None:
        verified_quarantine_identity = _compile_preview_directory_identity_in_parent(
            parent,
            quarantine_name,
        )
        if verified_quarantine_identity != expected_identity:
            raise click.ClickException(
                "Refusing to delete a compile preview quarantine whose identity changed: "
                f"{target.parent / quarantine_name}"
            )
        if backup_identity is not None:
            raise click.ClickException(
                "Refusing to delete a compile preview backup replacement whose identity is not cleanup-owned: "
                f"{target.parent / backup_name}"
            )
    elif backup_identity is None:
        _remove_compile_preview_transaction(target, transaction, parent=parent)
        parent.require_path_identity()
        return
    else:
        if _compile_preview_directory_identity_in_parent(parent, backup_name) != expected_identity:
            raise click.ClickException(
                "Refusing to quarantine a compile preview backup whose identity changed: "
                f"{target.parent / backup_name}"
            )
        _replace_compile_preview_sibling(parent, backup_name, quarantine_name)
        if _compile_preview_directory_identity_in_parent(parent, quarantine_name) != expected_identity:
            raise click.ClickException(
                "Refusing to delete a compile preview quarantine whose identity changed: "
                f"{target.parent / quarantine_name}"
            )
        if parent.status(backup_name) is not None:
            raise click.ClickException(
                "Refusing to delete a compile preview backup replacement whose identity is not cleanup-owned: "
                f"{target.parent / backup_name}"
            )

    _remove_compile_preview_cleanup_path(
        parent,
        quarantine_name,
        expected_identity,
    )
    if parent.status(backup_name) is not None:
        raise click.ClickException(
            "Compile preview backup was replaced while cleanup completed; transaction retained"
        )
    _remove_compile_preview_transaction(target, transaction, parent=parent)
    parent.require_path_identity()


def _remove_compile_preview_transaction(
    target: Path,
    expected: dict[str, object],
    *,
    parent: _PinnedCompilePreviewParent,
) -> None:
    current = _read_compile_preview_transaction(target, parent=parent)
    if current is None:
        return
    if current.get("transaction_id") != expected.get("transaction_id"):
        raise click.ClickException("Compile preview transaction changed before cleanup")
    parent.unlink(_compile_preview_transaction_path(target).name)
    parent.fsync()


def _compile_preview_path_kind_at(
    parent: _PinnedCompilePreviewParent,
    name: str,
) -> str | None:
    status = parent.status(name)
    if status is None:
        return None
    if not stat.S_ISDIR(status.st_mode):
        return "unrecognized"
    try:
        descriptor = parent.open_directory(name)
    except OSError:
        return "unrecognized"
    try:
        if _compile_preview_status_identity(os.fstat(descriptor)) != _compile_preview_status_identity(status):
            return "unrecognized"
        if _is_managed_compile_preview_descriptor(descriptor):
            return "managed"
        with os.scandir(descriptor) as entries:
            if next(entries, None) is None:
                return "empty"
        return "unrecognized"
    finally:
        os.close(descriptor)


def _is_managed_compile_preview_descriptor(descriptor: int) -> bool:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        marker_descriptor = os.open(
            _COMPILE_PREVIEW_MARKER_NAME,
            flags,
            dir_fd=descriptor,
        )
    except OSError:
        return False
    try:
        marker_status = os.fstat(marker_descriptor)
        if not stat.S_ISREG(marker_status.st_mode) or marker_status.st_nlink != 1:
            return False
        stream = os.fdopen(marker_descriptor, "rb")
        marker_descriptor = -1
        with stream:
            content = stream.read()
        return json.loads(content.decode("utf-8")) == _COMPILE_PREVIEW_MARKER
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    finally:
        if marker_descriptor >= 0:
            os.close(marker_descriptor)


def _remove_owned_compile_preview_path(
    parent: _PinnedCompilePreviewParent,
    name: str,
    expected_kind: str,
    *,
    expected_identity: str | None = None,
) -> None:
    path = parent.path / name
    if _compile_preview_path_kind_at(parent, name) != expected_kind:
        raise click.ClickException(f"Refusing to delete an unrecognized compile preview path: {path}")
    identity = parent.directory_identity(name)
    if expected_identity is not None and identity != expected_identity:
        raise click.ClickException(f"Refusing to delete a compile preview path whose identity changed: {path}")
    _remove_compile_preview_directory(
        parent,
        name,
        identity,
        label="owned path",
    )


def _remove_compile_preview_cleanup_path(
    parent: _PinnedCompilePreviewParent,
    name: str,
    expected_identity: str,
) -> None:
    _remove_compile_preview_directory(
        parent,
        name,
        expected_identity,
        label="backup quarantine",
    )


def _remove_compile_preview_directory(
    parent: _PinnedCompilePreviewParent,
    name: str,
    expected_identity: str,
    *,
    label: str,
) -> None:
    path = parent.path / name
    try:
        descriptor = parent.open_directory(name)
    except OSError as exc:
        raise click.ClickException(
            f"Refusing to delete a compile preview {label} whose identity changed: {path}"
        ) from exc
    try:
        descriptor_identity = _compile_preview_status_identity(os.fstat(descriptor))
        if descriptor_identity != expected_identity or parent.identity_for(name) != descriptor_identity:
            raise click.ClickException(
                f"Refusing to delete a compile preview {label} whose identity changed: {path}"
            )
        _remove_compile_preview_directory_entries(descriptor, path)
        if parent.identity_for(name) != descriptor_identity:
            raise click.ClickException(
                f"Refusing to remove a replaced compile preview {label}: {path}"
            )
        os.rmdir(name, dir_fd=parent.descriptor)
        parent.fsync()
    finally:
        os.close(descriptor)


def _remove_compile_preview_directory_entries(descriptor: int, path: Path) -> None:
    with os.scandir(descriptor) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        child_path = path / name
        try:
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        before_identity = _compile_preview_status_identity(before)
        if stat.S_ISDIR(before.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            try:
                child_descriptor = os.open(name, flags, dir_fd=descriptor)
            except OSError as exc:
                raise click.ClickException(
                    f"Refusing to delete a changed compile preview directory: {child_path}"
                ) from exc
            try:
                if _compile_preview_status_identity(os.fstat(child_descriptor)) != before_identity:
                    raise click.ClickException(
                        f"Refusing to delete a changed compile preview directory: {child_path}"
                    )
                _remove_compile_preview_directory_entries(child_descriptor, child_path)
                after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _compile_preview_status_identity(after) != before_identity:
                    raise click.ClickException(
                        f"Refusing to remove a replaced compile preview directory: {child_path}"
                    )
                os.rmdir(name, dir_fd=descriptor)
            finally:
                os.close(child_descriptor)
            continue

        if stat.S_ISREG(before.st_mode):
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            try:
                child_descriptor = os.open(name, flags, dir_fd=descriptor)
            except OSError as exc:
                raise click.ClickException(
                    f"Refusing to delete a changed compile preview file: {child_path}"
                ) from exc
            try:
                if _compile_preview_status_identity(os.fstat(child_descriptor)) != before_identity:
                    raise click.ClickException(
                        f"Refusing to delete a changed compile preview file: {child_path}"
                    )
            finally:
                os.close(child_descriptor)
        after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if _compile_preview_status_identity(after) != before_identity:
            raise click.ClickException(
                f"Refusing to unlink a replaced compile preview entry: {child_path}"
            )
        os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)


def _compile_preview_directory_identity(path: Path) -> str:
    if _compile_preview_is_link(path) or not path.is_dir():
        raise click.ClickException(f"compile preview directory identity is unsafe: {path}")
    try:
        return stable_filesystem_identity(path)
    except ProcessLockError as exc:
        raise click.ClickException(f"cannot verify compile preview directory identity: {path}") from exc


def _compile_preview_directory_identity_in_parent(
    parent: _PinnedCompilePreviewParent,
    name: str,
) -> str:
    path_identity = _compile_preview_directory_identity(parent.path / name)
    relative_identity = parent.directory_identity(name)
    if path_identity != relative_identity:
        raise click.ClickException(
            "Refusing to use a compile preview directory whose identity changed after validation: "
            f"{parent.path / name}"
        )
    return relative_identity


def _compile_preview_is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _fsync_compile_preview_tree(
    parent: _PinnedCompilePreviewParent,
    root_name: str,
    expected_identity: str,
) -> None:
    root_path = parent.path / root_name
    try:
        descriptor = parent.open_directory(root_name)
    except OSError as exc:
        raise click.ClickException(f"compile preview staging is unsafe: {root_path}") from exc
    try:
        if _compile_preview_status_identity(os.fstat(descriptor)) != expected_identity:
            raise click.ClickException(f"compile preview staging identity changed: {root_path}")
        _fsync_compile_preview_tree_descriptor(descriptor, root_path)
    finally:
        os.close(descriptor)
    parent.fsync()


def _fsync_compile_preview_tree_descriptor(descriptor: int, path: Path) -> None:
    with os.scandir(descriptor) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        child_path = path / name
        child_status = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISLNK(child_status.st_mode):
            raise click.ClickException(f"compile preview staging must not contain symlinks: {child_path}")
        if stat.S_ISREG(child_status.st_mode):
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            child_descriptor = os.open(name, flags, dir_fd=descriptor)
            try:
                if _compile_preview_status_identity(os.fstat(child_descriptor)) != _compile_preview_status_identity(
                    child_status
                ):
                    raise click.ClickException(f"compile preview staging file changed: {child_path}")
                os.fsync(child_descriptor)
            finally:
                os.close(child_descriptor)
        elif stat.S_ISDIR(child_status.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            child_descriptor = os.open(name, flags, dir_fd=descriptor)
            try:
                if _compile_preview_status_identity(os.fstat(child_descriptor)) != _compile_preview_status_identity(
                    child_status
                ):
                    raise click.ClickException(f"compile preview staging directory changed: {child_path}")
                _fsync_compile_preview_tree_descriptor(child_descriptor, child_path)
            finally:
                os.close(child_descriptor)
        else:
            raise click.ClickException(f"compile preview staging contains an unsafe entry: {child_path}")
    os.fsync(descriptor)


def _preflight_compile_preview_target(
    target: Path,
    *,
    parent: _PinnedCompilePreviewParent,
) -> str:
    parent.require_path_identity()
    _require_no_symlink_components(target)
    cwd = Path.cwd().resolve(strict=True)
    resolved_target = target.resolve(strict=False)
    if resolved_target == cwd or resolved_target in cwd.parents:
        raise click.ClickException(
            "compile preview output must not be the current working directory or one of its ancestors"
        )

    target_status = parent.status(target.name)
    if target_status is None:
        return parent.identity
    if not stat.S_ISDIR(target_status.st_mode):
        raise click.ClickException(f"compile preview output must be a directory: {target}")
    target_kind = _compile_preview_path_kind_at(parent, target.name)
    if target_kind == "unrecognized":
        raise click.ClickException(
            f"compile preview output is nonempty and not an open-xquant-managed compile preview: {target}"
        )
    return parent.identity


def _compile_preview_parent_identity(parent: Path) -> str:
    if _compile_preview_is_link(parent) or not parent.is_dir():
        raise click.ClickException(f"compile preview output parent is unsafe: {parent}")
    try:
        return stable_filesystem_identity(parent)
    except ProcessLockError as exc:
        raise click.ClickException(f"compile preview output parent identity is unavailable: {parent}") from exc


def _require_compile_preview_parent_identity(parent: Path, expected: str) -> None:
    if _compile_preview_parent_identity(parent) != expected:
        raise click.ClickException("compile preview output parent changed during publication")


def _require_no_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if _compile_preview_is_link(current):
            raise click.ClickException(
                f"compile preview output path must not contain symlink components: {absolute}"
            )
        if not current.exists():
            break


def _component_manifest_artifact_contents(
    staging_root: Path,
    manifests: list[dict],
    archived_paths: dict[int, tuple[str, str]],
) -> dict[str, bytes]:
    summary = [
        {
            "manifest_path": manifest.get("_source_manifest_path", manifest.get("_manifest_path", "")),
            **(
                {
                    "archived_manifest_path": archived_paths[index][0],
                    "archived_extension_root": archived_paths[index][1],
                }
                if index in archived_paths
                else {}
            ),
            "extension_id": manifest.get("extension_id", ""),
            "bundle_hash": manifest.get("bundle_hash", ""),
            "components": [
                {
                    "name": component.get("name", ""),
                    "kind": component.get("kind", ""),
                    "module": component.get("module", ""),
                    "class": component.get("class", ""),
                }
                for component in manifest.get("components", [])
                if isinstance(component, dict)
            ],
        }
        for index, manifest in enumerate(manifests)
    ]
    artifacts = {
        "component_manifests.json": (
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode(),
    }
    if len(manifests) == 1:
        if 0 in archived_paths:
            _copy_legacy_single_component_root(staging_root, manifests[0], archived_paths[0][1])
        if _single_component_manifest_is_run_local(staging_root, manifests[0], 0 in archived_paths):
            manifest_copy = dict(manifests[0])
            manifest_copy.pop("_manifest_path", None)
            manifest_copy.pop("_source_manifest_path", None)
            artifacts["component_manifest.json"] = (
                json.dumps(manifest_copy, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            ).encode()
        artifacts["component_bundle_hash.txt"] = (str(manifests[0].get("bundle_hash", "")) + "\n").encode()
    return artifacts


def _existing_component_replacement_names(run_dir: Path) -> set[str]:
    names = {"component_extensions"} if (run_dir / "component_extensions").exists() else set()
    manifest_path = run_dir / "component_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return names
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return names
    if not isinstance(manifest, dict):
        return names
    raw_root = manifest.get("extension_root") or manifest.get("extension_id")
    if isinstance(raw_root, str):
        _add_component_replacement_top_level(names, raw_root)
    for component in manifest.get("components") or []:
        if not isinstance(component, dict):
            continue
        for raw_test in component.get("tests") or []:
            if isinstance(raw_test, str):
                _add_component_replacement_top_level(names, raw_test)
    return names


def _add_component_replacement_top_level(names: set[str], raw_path: str) -> None:
    path = Path(raw_path)
    if raw_path and not path.is_absolute() and path != Path(".") and ".." not in path.parts:
        names.add(path.parts[0])


def _single_component_manifest_is_run_local(run_dir: Path, manifest: dict, archived: bool) -> bool:
    raw_root = manifest.get("extension_root") or manifest.get("extension_id")
    if not isinstance(raw_root, str) or not raw_root:
        return not archived
    root = Path(raw_root)
    if root.is_absolute() or ".." in root.parts:
        return False
    if raw_root == ".":
        return not archived
    return (run_dir / root).is_dir()


def _copy_legacy_single_component_root(run_dir: Path, manifest: dict, archived_extension_root: str) -> None:
    raw_root = manifest.get("extension_root") or manifest.get("extension_id")
    if not isinstance(raw_root, str) or not raw_root:
        return
    root = Path(raw_root)
    if root.is_absolute() or ".." in root.parts or raw_root == ".":
        return
    source_root = (run_dir / archived_extension_root).resolve()
    target_root = (run_dir / root).resolve()
    if not source_root.is_dir() or not source_root.is_relative_to(run_dir.resolve()):
        return
    if not target_root.is_relative_to(run_dir.resolve()):
        return
    shutil.copytree(
        source_root,
        target_root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "*.pyc", "*.pyo"),
    )
    archive_base = source_root.parent
    for component in manifest.get("components") or []:
        if not isinstance(component, dict) or not isinstance(component.get("tests"), list):
            continue
        for raw_path in component["tests"]:
            if not isinstance(raw_path, str):
                continue
            relative_path = Path(raw_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                continue
            archived_test = archive_base / relative_path
            target_test = run_dir / relative_path
            if archived_test.is_file() and archived_test.resolve() != target_test.resolve():
                target_test.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(archived_test, target_test)


def _preflight_component_extension_archives(run_dir: Path, manifests: list[dict]) -> None:
    for index, manifest in enumerate(manifests):
        _component_extension_archive_paths(run_dir, manifest, index)


def _component_extension_archive_paths(run_dir: Path, manifest: dict, index: int) -> tuple[Path, Path, Path, str] | None:
    manifest_path_raw = manifest.get("_manifest_path")
    if not isinstance(manifest_path_raw, str) or not manifest_path_raw:
        return None
    raw_root = manifest.get("extension_root") or manifest.get("extension_id")
    if not isinstance(raw_root, str) or not raw_root:
        return None
    manifest_path = Path(manifest_path_raw).resolve()
    source_root_raw = manifest_path.parent / raw_root
    source_root = source_root_raw.resolve()
    if not source_root.is_dir() or not source_root.is_relative_to(manifest_path.parent):
        return None
    _reject_component_extension_symlinks(source_root_raw)
    archive_name = f"{index:02d}_{_component_archive_slug(manifest, manifest_path)}"
    archive_base = (run_dir / "component_extensions" / archive_name).resolve()
    archived_root = (archive_base / raw_root).resolve()
    if not archive_base.is_relative_to(run_dir.resolve()) or not archived_root.is_relative_to(run_dir.resolve()):
        return None
    if archived_root.is_relative_to(source_root) or source_root.is_relative_to(archived_root):
        raise click.ClickException(
            "component extension archive would be nested inside the source extension; "
            "choose an --out directory outside the component extension root"
        )
    _component_extension_external_test_files(manifest, manifest_path, source_root)
    return manifest_path, source_root, archive_base, raw_root


def _reject_component_extension_symlinks(source_root: Path) -> None:
    if source_root.is_symlink():
        raise click.ClickException("component extension archive refuses symlinks inside the extension root")
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise click.ClickException("component extension archive refuses symlinks inside the extension root")


def _archive_component_extension(run_dir: Path, manifest: dict, index: int) -> tuple[str, str] | None:
    archive_paths = _component_extension_archive_paths(run_dir, manifest, index)
    if archive_paths is None:
        return None
    manifest_path, source_root, archive_base, raw_root = archive_paths
    archived_root = (archive_base / raw_root).resolve()
    shutil.copytree(
        source_root,
        archived_root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "*.pyc", "*.pyo"),
    )
    for source_file, relative_path in _component_extension_external_test_files(manifest, manifest_path, source_root):
        target = archive_base / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
    manifest_copy = dict(manifest)
    manifest_copy.pop("_manifest_path", None)
    manifest_copy.pop("_source_manifest_path", None)
    archived_manifest = archive_base / manifest_path.name
    archived_manifest.write_text(
        json.dumps(manifest_copy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return (
        archived_manifest.relative_to(run_dir.resolve()).as_posix(),
        archived_root.relative_to(run_dir.resolve()).as_posix(),
    )


def _component_extension_external_test_files(manifest: dict, manifest_path: Path, source_root: Path) -> list[tuple[Path, Path]]:
    workspace_root = manifest_path.parent.resolve()
    source_root = source_root.resolve()
    files: list[tuple[Path, Path]] = []
    for component in manifest.get("components") or []:
        if not isinstance(component, dict):
            continue
        tests = component.get("tests")
        if not isinstance(tests, list):
            continue
        for raw in tests:
            if not isinstance(raw, str):
                continue
            raw_path = Path(raw)
            if raw_path.is_absolute() or ".." in raw_path.parts:
                raise click.ClickException(f"component extension test path is unsafe: {raw}")
            raw_source_file = workspace_root / raw_path
            if _path_contains_symlink(raw_source_file, workspace_root):
                raise click.ClickException("component extension archive refuses symlinked external test files")
            source_file = raw_source_file.resolve()
            if not source_file.is_relative_to(workspace_root):
                raise click.ClickException(f"component extension test path escapes the workspace: {raw}")
            if not source_file.exists() or source_file.is_relative_to(source_root):
                continue
            if source_file.is_symlink() or not source_file.is_file():
                raise click.ClickException("component extension archive refuses non-file or symlinked external test files")
            files.append((source_file, raw_path))
    return files


def _path_contains_symlink(path: Path, root: Path) -> bool:
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _component_archive_slug(manifest: dict, manifest_path: Path) -> str:
    raw = str(manifest.get("extension_id") or manifest_path.stem)
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw).strip("_")
    return slug or "component_extension"


@main.group()
def backtest():
    """Run backtests from strategy specs."""


def _backtest_artifact_paths(run_dir: Path) -> dict[str, str]:
    artifacts = {
        "strategy_spec_yaml": str(run_dir / "strategy_spec.yaml"),
        "environment_json": str(run_dir / "environment.json"),
        "data_manifest_json": str(run_dir / "data_manifest.json"),
        "execution_assumptions_json": str(run_dir / "execution_assumptions.json"),
        "compiled_plan_json": str(run_dir / "compiled_plan.json"),
        "strategy_py": str(run_dir / "strategy.py"),
        "equity_curve_csv": str(run_dir / "equity_curve.csv"),
        "trades_csv": str(run_dir / "trades.csv"),
        "positions_csv": str(run_dir / "positions.csv"),
        "orders_csv": str(run_dir / "orders.csv"),
        "target_weights_csv": str(run_dir / "target_weights.csv"),
        "metrics_json": str(run_dir / "metrics.json"),
        "artifact_hashes_json": str(run_dir / "artifact_hashes.json"),
        "run_log_jsonl": str(run_dir / "run_log.jsonl"),
    }
    benchmark_curve = run_dir / "benchmark_curve.csv"
    if benchmark_curve.exists():
        artifacts["benchmark_curve_csv"] = str(benchmark_curve)
    component_manifest = run_dir / "component_manifest.json"
    if component_manifest.exists():
        artifacts["component_manifest_json"] = str(component_manifest)
    component_manifests = run_dir / "component_manifests.json"
    if component_manifests.exists():
        artifacts["component_manifests_json"] = str(component_manifests)
    component_bundle_hash = run_dir / "component_bundle_hash.txt"
    if component_bundle_hash.exists():
        artifacts["component_bundle_hash_txt"] = str(component_bundle_hash)
    spec_audit = run_dir / "spec_audit.json"
    if spec_audit.exists():
        artifacts["spec_audit_json"] = str(spec_audit)
    runtime_audit = run_dir / "runtime_audit.json"
    if runtime_audit.exists():
        artifacts["runtime_audit_json"] = str(runtime_audit)
    return artifacts


def _backtest_summary_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _backtest_json_failure(check: str, message: str, warnings: list[dict] | None = None) -> dict:
    return {
        "status": "fail",
        "run_id": "",
        "run_dir": "",
        "artifacts": {},
        "metrics": {},
        "warnings": warnings or [],
        "errors": [{"severity": "fatal", "check": check, "message": message}],
    }


def _load_run_json(run_dir: Path, name: str) -> dict:
    path = run_dir / name
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"run comparison artifact is not valid JSON: {path}: {exc.msg}") from exc
    return value if isinstance(value, dict) else {}


def _load_run_text(run_dir: Path, name: str) -> str:
    path = run_dir / name
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _require_comparable_run_artifacts(run_dir: Path) -> None:
    required = {
        "strategy_spec.yaml",
        "spec_hash.txt",
        "compiled_plan.json",
        "data_manifest.json",
        "execution_assumptions.json",
        "metrics.json",
        "artifact_hashes.json",
    }
    missing = sorted(name for name in required if not (run_dir / name).exists())
    if missing:
        raise click.ClickException(f"run directory is missing required comparison artifacts: {missing}")
    for name in ("compiled_plan.json", "data_manifest.json", "execution_assumptions.json", "metrics.json", "artifact_hashes.json"):
        payload = _load_run_json(run_dir, name)
        if not payload:
            raise click.ClickException(f"run comparison artifact must be a JSON object: {run_dir / name}")
    _require_comparison_metrics_identity(run_dir)
    _require_run_artifact_hashes_current(run_dir)


def _require_comparison_metrics_identity(run_dir: Path) -> None:
    metrics = _load_run_json(run_dir, "metrics.json")
    run_id = metrics.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise click.ClickException("metrics.json run_id is required for comparison")
    expected_run_id = run_dir.resolve().name
    if run_id != expected_run_id:
        raise click.ClickException(
            f"metrics.json run_id does not match run directory: metrics={run_id}, directory={expected_run_id}"
        )
    strategy_id = metrics.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id:
        raise click.ClickException("metrics.json strategy_id is required for comparison")
    try:
        expected_strategy_id = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml").strategy_id
    except Exception as exc:
        raise click.ClickException(f"strategy_spec.yaml cannot be parsed for comparison: {exc}") from exc
    if strategy_id != expected_strategy_id:
        raise click.ClickException(
            "metrics.json strategy_id does not match strategy_spec.yaml: "
            f"metrics={strategy_id}, strategy_spec={expected_strategy_id}"
        )


def _hash_run_artifact_for_comparison(run_dir: Path, name: str) -> str:
    from oxq.spec.compiler import _hash_file, _hash_json_file

    path = run_dir / name
    if name == "strategy_spec.yaml":
        return _hash_file(path)
    if name == "metrics.json":
        return _hash_json_file(path, exclude_keys={"run_id"})
    if name in {
        "compiled_plan.json",
        "data_manifest.json",
        "execution_assumptions.json",
        "spec_audit.json",
        "runtime_audit.json",
        "component_manifest.json",
        "component_manifests.json",
    }:
        try:
            return _hash_json_file(path)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"{name} is not valid JSON: {path}: {exc.msg}") from exc
    return _hash_file(path)


def _require_run_artifact_hashes_current(run_dir: Path) -> None:
    artifact_hashes = _load_run_json(run_dir, "artifact_hashes.json")
    required_hashes = {
        "strategy_spec.yaml",
        "compiled_plan.json",
        "data_manifest.json",
        "execution_assumptions.json",
        "metrics.json",
    }
    provenance_hashes = {
        "spec_audit.json",
        "runtime_audit.json",
        "conversation_hash.txt",
        "component_catalog_hash.txt",
        "recipe_catalog_hash.txt",
        "component_manifest.json",
        "component_manifests.json",
        "component_bundle_hash.txt",
    }
    for name in provenance_hashes:
        if name in artifact_hashes or (run_dir / name).exists():
            required_hashes.add(name)
    for name in required_hashes:
        stored = artifact_hashes.get(name)
        if not isinstance(stored, str) or not stored:
            raise click.ClickException(f"artifact_hashes.json missing required hash for comparison artifact: {name}")
        if not (run_dir / name).exists():
            raise click.ClickException(f"artifact_hashes.json references missing comparison artifact: {name}")
        actual = _hash_run_artifact_for_comparison(run_dir, name)
        if stored != actual:
            raise click.ClickException(f"artifact hash mismatch for {name}: stored={stored}, actual={actual}")
    try:
        actual_spec_hash = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml").compute_hash()
    except Exception as exc:
        raise click.ClickException(f"strategy_spec.yaml cannot be parsed for comparison: {exc}") from exc
    stored_spec_hash = _load_run_text(run_dir, "spec_hash.txt")
    if stored_spec_hash != actual_spec_hash:
        raise click.ClickException(
            f"spec_hash.txt mismatch for strategy_spec.yaml: stored={stored_spec_hash}, actual={actual_spec_hash}"
        )
    _require_run_digest_current(run_dir)


def _replace_run_digest_entry(run_dir: Path, artifact_hashes_hash: str) -> None:
    from oxq.run_digests import RunDigestError, replace_run_digest_entry

    try:
        replace_run_digest_entry(run_dir, artifact_hashes_hash)
    except RunDigestError as exc:
        raise click.ClickException(str(exc)) from exc


def _require_run_digest_current(run_dir: Path) -> None:
    from oxq.run_digests import RunDigestError, require_current_run_digest

    try:
        require_current_run_digest(run_dir)
    except RunDigestError as exc:
        message = str(exc)
        if "component manifest" in message:
            message = f"component bundle verification failed: {message}"
        raise click.ClickException(message) from exc


def _run_comparability_signature(run_dir: Path) -> dict[str, object]:
    resolved = run_dir.resolve()
    with run_digest_transaction(resolved):
        return _run_comparability_signature_locked(resolved)


def _run_comparability_signature_locked(run_dir: Path) -> dict[str, object]:
    _require_comparable_run_artifacts(run_dir)
    compiled_plan = _load_run_json(run_dir, "compiled_plan.json")
    data_manifest = _load_run_json(run_dir, "data_manifest.json")
    execution_assumptions = _load_run_json(run_dir, "execution_assumptions.json")
    artifact_hashes = _load_run_json(run_dir, "artifact_hashes.json")
    return {
        "spec_hash": _load_run_text(run_dir, "spec_hash.txt"),
        "component_catalog_hash": _load_run_text(run_dir, "component_catalog_hash.txt"),
        "recipe_catalog_hash": _load_run_text(run_dir, "recipe_catalog_hash.txt"),
        "spec_audit_hash": (
            _hash_run_artifact_for_comparison(run_dir, "spec_audit.json")
            if (run_dir / "spec_audit.json").exists()
            else artifact_hashes.get("spec_audit.json", "")
        ),
        "runtime_audit_hash": (
            _hash_run_artifact_for_comparison(run_dir, "runtime_audit.json")
            if (run_dir / "runtime_audit.json").exists()
            else artifact_hashes.get("runtime_audit.json", "")
        ),
        "compiled_plan_hash": _hash_run_artifact_for_comparison(run_dir, "compiled_plan.json"),
        "component_bundle_hashes": sorted(_run_component_bundle_hashes(run_dir)),
        "data": {
            "provider": data_manifest.get("provider", ""),
            "symbols": data_manifest.get("symbols", []),
            "calendar": data_manifest.get("calendar", ""),
            "price_adjustment": data_manifest.get("price_adjustment", ""),
            "start": data_manifest.get("start", ""),
            "end": data_manifest.get("end", ""),
            "min_start_date": data_manifest.get("min_start_date", ""),
            "analysis_start": data_manifest.get("analysis_start", ""),
            "warmup_policy": data_manifest.get("warmup_policy", ""),
            "effective_data_dir": data_manifest.get("effective_data_dir", ""),
            "data_fingerprints": data_manifest.get("data_fingerprints", {}),
        },
        "execution": compiled_plan.get("execution", {}),
        "cost": compiled_plan.get("cost", {}),
        "validation": compiled_plan.get("validation", {}),
        "metrics": compiled_plan.get("metrics", {}),
        "execution_assumptions": execution_assumptions,
    }


def _compare_run_signatures(left: dict[str, object], right: dict[str, object]) -> list[dict[str, object]]:
    checks = [
        ("spec_hash", "spec_hash"),
        ("component_catalog_hash", "component_catalog_hash"),
        ("recipe_catalog_hash", "recipe_catalog_hash"),
        ("spec_audit_hash", "spec_audit_hash"),
        ("runtime_audit_hash", "runtime_audit_hash"),
        ("compiled_plan_hash", "compiled_plan_hash"),
        ("component_bundle_hashes", "component_bundle_hashes"),
        ("data", "data"),
        ("execution", "execution"),
        ("cost", "cost"),
        ("validation", "validation"),
        ("metrics", "metrics"),
        ("execution_assumptions", "execution_assumptions"),
    ]
    differences: list[dict[str, object]] = []
    for key, label in checks:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value != right_value:
            differences.append(
                {
                    "field": label,
                    "left": left_value,
                    "right": right_value,
                    "severity": "blocking",
                }
            )
    return differences


def _comparison_run_digest_error_message(message: str) -> str:
    hash_mismatch = re.fullmatch(r"(?P<name>[^:]+) hash mismatch: stored=(?P<stored>[^,]+), actual=(?P<actual>.+)", message)
    if hash_mismatch is not None:
        return (
            f"artifact hash mismatch for {hash_mismatch.group('name')}: "
            f"stored={hash_mismatch.group('stored')}, actual={hash_mismatch.group('actual')}"
        )

    invalid_json = re.fullmatch(r"published JSON artifact is invalid: (?P<name>[^:]+): (?P<detail>.+)", message)
    if invalid_json is not None:
        return f"{invalid_json.group('name')} is not valid JSON: {invalid_json.group('detail')}"

    unbound_governed = re.fullmatch(r"artifact_hashes\.json has unbound governed files: \[(?P<names>.+)\]", message)
    if unbound_governed is not None:
        comparison_artifacts = {
            "strategy_spec.yaml",
            "compiled_plan.json",
            "data_manifest.json",
            "execution_assumptions.json",
            "metrics.json",
            "spec_audit.json",
            "runtime_audit.json",
            "conversation_hash.txt",
            "component_catalog_hash.txt",
            "recipe_catalog_hash.txt",
            "component_manifest.json",
            "component_manifests.json",
            "component_bundle_hash.txt",
        }
        names = re.findall(r"'([^']+)'", unbound_governed.group("names"))
        artifact_name = next((name for name in names if name in comparison_artifacts), None)
        if artifact_name is not None:
            return f"artifact_hashes.json missing required hash for comparison artifact: {artifact_name}"

    return message


@backtest.command(name="compare-runs")
@click.argument("left_run_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("right_run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def compare_runs(left_run_dir: str, right_run_dir: str, as_json: bool):
    """Check whether two run directories are comparable before judging returns."""
    left_path = Path(left_run_dir)
    right_path = Path(right_run_dir)
    try:
        with multi_run_digest_read_transaction([left_path, right_path]) as (left_resolved, right_resolved):
            left = _run_comparability_signature_locked(left_resolved)
            right = _run_comparability_signature_locked(right_resolved)
            differences = _compare_run_signatures(left, right)
    except (click.ClickException, RunDigestError) as exc:
        message = exc.message if isinstance(exc, click.ClickException) else str(exc)
        if isinstance(exc, RunDigestError):
            message = _comparison_run_digest_error_message(message)
            if "component manifest" in message:
                message = f"component bundle verification failed: {message}"
            for run_path in (left_path.resolve(), right_path.resolve()):
                missing = sorted(
                    name
                    for name in {
                        "strategy_spec.yaml",
                        "spec_hash.txt",
                        "compiled_plan.json",
                        "data_manifest.json",
                        "execution_assumptions.json",
                        "metrics.json",
                        "artifact_hashes.json",
                    }
                    if not (run_path / name).exists()
                )
                if missing:
                    message = f"run directory is missing required comparison artifacts: {missing}"
                    break
        check = "run_artifacts_missing" if "missing required comparison artifacts" in message else "run_artifacts_invalid"
        payload = {
            "status": "fail",
            "comparable": False,
            "left_run_dir": str(left_path),
            "right_run_dir": str(right_path),
            "differences": [],
            "errors": [{"severity": "fatal", "check": check, "message": message}],
        }
        if as_json:
            click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        else:
            click.echo(f"Status: {payload['status'].upper()}")
            click.echo("Comparable: false")
            click.echo(f"  [fatal] {message}")
        raise SystemExit(1)
    comparable = not any(item.get("severity") == "blocking" for item in differences)
    payload = {
        "status": "pass" if comparable else "fail",
        "comparable": comparable,
        "left_run_dir": str(left_path),
        "right_run_dir": str(right_path),
        "differences": differences,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        click.echo(f"Status: {payload['status'].upper()}")
        click.echo(f"Comparable: {str(comparable).lower()}")
        for item in differences:
            click.echo(f"  [{item['severity']}] {item['field']}")
    if not comparable:
        raise SystemExit(1)


@backtest.command()
@click.argument("spec_file", type=click.Path())
@click.option(
    "--out",
    "-o",
    default=None,
    help="Output directory for run artifacts. Defaults to workspace default_output_dir or runs/auto.",
)
@click.option(
    "--data-dir",
    default=None,
    help=(
        "Directory for market data files. The resolved effective data_dir is "
        "recorded in compiled_plan.json and affects runtime audit hashes."
    ),
)
@click.option(
    "--spec-audit",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Pre-run spec_audit.json gate for formal audited backtests.",
)
@click.option(
    "--runtime-audit",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Pre-run runtime_audit.json gate for formal audited backtests.",
)
@click.option(
    "--component-catalog",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="component_catalog.json used by the audited spec gate.",
)
@click.option(
    "--authorization",
    "authorization_path",
    default=None,
    type=click.Path(exists=False, dir_okay=False),
    help="backtest_authorization.json approving the formal audited backtest.",
)
@click.option(
    "--component-manifest",
    "component_manifest",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Workspace component manifest to load before validation, compile, and run.",
)
@click.option(
    "--allow-unaudited",
    is_flag=True,
    help="Allow an exploratory run without spec_audit.json and runtime_audit.json.",
)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def run(
    spec_file: str,
    out: str | None,
    data_dir: str | None,
    spec_audit: str | None,
    runtime_audit: str | None,
    component_catalog: str | None,
    authorization_path: str | None,
    component_manifest: tuple[str, ...],
    allow_unaudited: bool,
    as_json: bool,
):
    """Run a backtest from a strategy spec file.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    spec_path = Path(spec_file)
    if not spec_path.exists():
        message = f"strategy spec file not found: {spec_file}"
        if as_json:
            click.echo(json.dumps(_backtest_json_failure("spec_file_missing", message), indent=2))
            raise SystemExit(1)
        raise click.ClickException(message)

    pre_run_audit_path = (
        Path(spec_audit)
        if spec_audit is not None
        else (None if allow_unaudited else _default_spec_audit_path(spec_path))
    )
    pre_run_runtime_audit_path = (
        Path(runtime_audit)
        if runtime_audit is not None
        else (None if allow_unaudited else _default_runtime_audit_path(spec_path))
    )
    pre_run_component_catalog_path = (
        Path(component_catalog)
        if component_catalog is not None
        else (None if allow_unaudited and pre_run_audit_path is None else _default_component_catalog_path(spec_path))
    )
    pre_run_authorization_path = (
        Path(authorization_path)
        if authorization_path is not None
        else _default_backtest_authorization_path(pre_run_runtime_audit_path)
    )
    if pre_run_runtime_audit_path is not None and pre_run_audit_path is None:
        message = "spec_audit.json is required when a runtime audit gates a formal backtest"
        if as_json:
            click.echo(json.dumps(_backtest_json_failure("spec_audit_missing", message), indent=2))
            raise SystemExit(1)
        raise click.ClickException(message)
    if pre_run_audit_path is not None and pre_run_runtime_audit_path is None:
        message = "runtime_audit.json is required when a spec audit gates a formal backtest"
        if as_json:
            click.echo(json.dumps(_backtest_json_failure("runtime_audit_missing", message), indent=2))
            raise SystemExit(1)
        raise click.ClickException(message)
    formal_gated_run = pre_run_audit_path is not None or pre_run_runtime_audit_path is not None
    if not allow_unaudited or formal_gated_run:
        missing_gates = []
        if pre_run_audit_path is None:
            missing_gates.append("spec_audit.json")
        if pre_run_runtime_audit_path is None:
            missing_gates.append("runtime_audit.json")
        if pre_run_component_catalog_path is None:
            missing_gates.append("component_catalog.json")
        if missing_gates:
            message = (
                "formal backtest requires audited gate artifacts: "
                f"{', '.join(missing_gates)}. Use --allow-unaudited only for exploratory runs."
            )
            check = "audit_artifacts_missing"
            if pre_run_audit_path is None:
                check = "spec_audit_missing"
            elif pre_run_runtime_audit_path is None:
                check = "runtime_audit_missing"
            elif pre_run_component_catalog_path is None:
                check = "component_catalog_missing"
            if as_json:
                click.echo(json.dumps(_backtest_json_failure(check, message), indent=2))
                raise SystemExit(1)
            raise click.ClickException(message)
    if formal_gated_run:
        try:
            _require_complete_governed_workspace()
        except click.ClickException as e:
            if as_json:
                click.echo(json.dumps(_backtest_json_failure("formal_gate_path_failed", e.message), indent=2))
                raise SystemExit(1)
            raise

    component_import_paths = component_manifest
    try:
        component_manifest_payloads = _read_component_manifest_payloads(component_manifest)
        if formal_gated_run and component_manifest_payloads:
            staging, component_import_paths, component_manifest_payloads = _stage_component_manifest_snapshots(
                component_manifest_payloads
            )
            click.get_current_context().call_on_close(staging.cleanup)
        spec = StrategySpec.from_yaml(spec_file)
        gate_spec = _normalize_spec_for_run(spec)
    except Exception as e:
        if as_json:
            click.echo(json.dumps(_backtest_json_failure("parse_error", str(e)), indent=2))
            raise SystemExit(1)
        raise

    resolved_out: str | None = None
    if formal_gated_run:
        try:
            resolved_out = _resolve_backtest_output_dir(out)
        except click.ClickException as e:
            if as_json:
                click.echo(json.dumps(_backtest_json_failure("output_dir_failed", e.message), indent=2))
                raise SystemExit(1)
            raise
        try:
            _require_governed_formal_gate_paths(
                spec_path=spec_path,
                spec_audit_path=pre_run_audit_path,
                runtime_audit_path=pre_run_runtime_audit_path,
                component_catalog_path=pre_run_component_catalog_path,
                authorization_path=pre_run_authorization_path,
                component_manifest_paths=component_manifest,
                run_out=Path(resolved_out),
            )
        except click.ClickException as e:
            if as_json:
                click.echo(json.dumps(_backtest_json_failure("formal_gate_path_failed", e.message), indent=2))
                raise SystemExit(1)
            raise
    if pre_run_audit_path is not None:
        try:
            _require_pre_backtest_spec_audit(gate_spec, pre_run_audit_path)
        except click.ClickException as e:
            if as_json:
                click.echo(
                    json.dumps(_backtest_json_failure("spec_audit_failed", e.message), indent=2)
                )
                raise SystemExit(1)
            raise
    component_bundle_hashes = _component_bundle_hashes(component_manifest_payloads)
    if pre_run_runtime_audit_path is not None:
        try:
            _require_pre_backtest_runtime_audit(
                gate_spec,
                pre_run_runtime_audit_path,
                spec_audit_path=pre_run_audit_path,
                effective_data_dir=_resolve_effective_data_dir(spec, data_dir),
                component_bundle_hashes=component_bundle_hashes,
                component_manifests=component_manifest_payloads,
            )
        except click.ClickException as e:
            if as_json:
                click.echo(json.dumps(_backtest_json_failure("runtime_audit_failed", e.message), indent=2))
                raise SystemExit(1)
            raise
    if pre_run_component_catalog_path is not None and pre_run_audit_path is not None:
        try:
            _require_component_catalog_before_import(
                pre_run_component_catalog_path,
                spec_audit_path=pre_run_audit_path,
                component_bundle_hashes=component_bundle_hashes,
            )
        except click.ClickException as e:
            if as_json:
                click.echo(json.dumps(_backtest_json_failure("component_catalog_failed", e.message), indent=2))
                raise SystemExit(1)
            raise
    if resolved_out is None:
        try:
            resolved_out = _resolve_backtest_output_dir(out)
        except click.ClickException as e:
            if as_json:
                click.echo(json.dumps(_backtest_json_failure("output_dir_failed", e.message), indent=2))
                raise SystemExit(1)
            raise
    if component_manifest_payloads:
        out_path = Path(resolved_out)
        if out_path.name == "auto":
            preflight_run_dir = out_path.parent / "__component_archive_preflight__"
        else:
            preflight_run_dir = out_path / "__component_archive_preflight__"
        try:
            _preflight_component_extension_archives(preflight_run_dir, component_manifest_payloads)
        except click.ClickException as e:
            if as_json:
                click.echo(json.dumps(_backtest_json_failure("component_archive_failed", e.message), indent=2))
                raise SystemExit(1)
            raise

    def require_authorization(*, warnings: list[dict] | None = None) -> None:
        try:
            _require_backtest_authorization(
                pre_run_authorization_path,
                spec=gate_spec,
                spec_path=spec_path,
                spec_audit_path=pre_run_audit_path,
                runtime_audit_path=pre_run_runtime_audit_path,
                component_catalog_path=pre_run_component_catalog_path,
                component_manifest_paths=component_manifest,
                data_dir=data_dir,
                run_out=Path(resolved_out),
            )
        except click.ClickException as e:
            check = (
                "backtest_authorization_missing"
                if pre_run_authorization_path is None or not pre_run_authorization_path.exists()
                else "backtest_authorization_failed"
            )
            if as_json:
                click.echo(json.dumps(_backtest_json_failure(check, e.message, warnings=warnings), indent=2))
                raise SystemExit(1)
            raise

    authorization_validated = False
    loaded_component_manifests: list[dict] = []
    if component_manifest:
        if formal_gated_run:
            require_authorization()
            authorization_validated = True
        try:
            if formal_gated_run:
                _require_component_sources_match_staged(component_manifest, component_manifest_payloads)
            loaded_component_manifests = _load_component_manifests(
                component_import_paths,
                source_manifest_paths=component_manifest if formal_gated_run else None,
            )
        except Exception as e:
            if as_json:
                click.echo(json.dumps(_backtest_json_failure("component_manifest_failed", str(e)), indent=2))
                raise SystemExit(1)
            raise

    validation = validate_spec(spec)
    if validation.status == "fail":
        if as_json:
            click.echo(
                json.dumps(
                    {
                        "status": "fail",
                        "run_id": "",
                        "run_dir": "",
                        "artifacts": {},
                        "metrics": {},
                        "warnings": validation.warnings,
                        "errors": validation.errors,
                    },
                    indent=2,
                )
            )
            raise SystemExit(1)
        click.echo("Spec validation failed. Fix errors before running backtest:")
        for e in validation.errors:
            click.echo(f"  [{e['severity']}] {e['check']}: {e['message']}")
        raise SystemExit(1)
    if validation.warnings and not as_json:
        click.echo("Warnings (continuing):")
        for w in validation.warnings:
            click.echo(f"  [{w['severity']}] {w['check']}: {w['message']}")

    if formal_gated_run and not authorization_validated:
        require_authorization(warnings=validation.warnings)

    if not as_json:
        click.echo(f"Running backtest for '{spec.strategy_id}'...")
        effective_data_dir = _resolve_effective_data_dir(spec, data_dir)
        click.echo(f"  Effective data dir: {effective_data_dir}")
        click.echo("  Note: effective data_dir is included in compiled_plan.json and its hash.")
    from oxq.spec.compiler import compile_run

    try:
        result, run_dir = compile_run(spec, data_dir=data_dir, out_dir=resolved_out)
    except Exception as e:
        if as_json:
            click.echo(
                json.dumps(
                    _backtest_json_failure("runtime_error", str(e), warnings=validation.warnings),
                    indent=2,
                )
            )
            raise SystemExit(1)
        raise
    run_dir = Path(run_dir)
    if loaded_component_manifests:
        _write_run_component_manifest_artifacts(run_dir, loaded_component_manifests)
    if pre_run_audit_path is not None and pre_run_runtime_audit_path is not None:
        try:
            _attach_provenance_artifacts(
                run_dir,
                spec_audit_path=pre_run_audit_path,
                runtime_audit_path=pre_run_runtime_audit_path,
                component_catalog_path=pre_run_component_catalog_path,
            )
        except click.ClickException as e:
            if as_json:
                click.echo(
                    json.dumps(
                        _backtest_json_failure("runtime_audit_failed", e.message, warnings=validation.warnings),
                        indent=2,
                    )
                )
                raise SystemExit(1)
            raise

    if as_json:
        click.echo(
            json.dumps(
                {
                    "status": "pass",
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "artifacts": _backtest_artifact_paths(run_dir),
                    "metrics": _backtest_summary_metrics(run_dir),
                    "warnings": validation.warnings,
                    "errors": validation.errors,
                },
                indent=2,
            )
        )
        return

    click.echo(f"\nRun complete. Artifacts written to {run_dir}/")
    click.echo(f"  Total Return: {result.total_return():.2%}")
    click.echo(f"  Sharpe Ratio: {result.sharpe_ratio():.2f}")
    click.echo(f"  Max Drawdown: {result.max_drawdown():.2%}")
    click.echo(f"  Trade Count:  {len(result.trades)}")


@backtest.command(name="attach-provenance")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--spec-audit", required=True, type=click.Path(exists=True, dir_okay=False), help="spec_audit.json path.")
@click.option(
    "--runtime-audit",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="runtime_audit.json path.",
)
@click.option(
    "--component-catalog",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="component_catalog.json path.",
)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def attach_provenance(run_dir: str, spec_audit: str, runtime_audit: str, component_catalog: str, as_json: bool):
    """Attach pre-run provenance artifacts while preserving run digests."""
    from oxq.audit import audit_reproducibility
    from oxq.core.component_catalog import _catalog_hash, _stable_hash
    from oxq.spec.audit_schema import validate_spec_audit_file
    from oxq.spec.compiler import _hash_file
    from oxq.spec.runtime_audit_schema import validate_runtime_audit_file

    run_path = Path(run_dir)
    artifact_hashes_path = run_path / "artifact_hashes.json"
    if not artifact_hashes_path.exists():
        raise click.ClickException(f"missing artifact_hashes.json in run directory: {run_dir}")
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    if not isinstance(artifact_hashes, dict):
        raise click.ClickException("artifact_hashes.json must be an object")
    pre_attach_audit = audit_reproducibility(run_path)
    if pre_attach_audit.get("status") == "fail":
        failing = [
            check.get("id", "unknown")
            for check in pre_attach_audit.get("checks", [])
            if check.get("severity") == "fatal" and check.get("status") == "fail"
        ]
        raise click.ClickException(f"run reproducibility must pass before attaching provenance: {failing}")

    run_spec_path = run_path / "strategy_spec.yaml"
    if not run_spec_path.exists():
        raise click.ClickException(f"missing strategy_spec.yaml in run directory: {run_dir}")
    audit_validation = validate_spec_audit_file(
        spec_audit,
        spec_path=run_spec_path,
        require_confirmed_coverage=True,
        verify_confirmation_table=True,
    )
    if audit_validation["status"] == "fail":
        raise click.ClickException(f"invalid spec audit: {audit_validation['errors']}")

    audit_payload = json.loads(Path(spec_audit).read_text(encoding="utf-8"))
    catalog_payload = json.loads(Path(component_catalog).read_text(encoding="utf-8"))
    audit_status = _require_json_str(audit_payload, "status")
    if audit_status != "pass":
        raise click.ClickException(f"spec audit status must be pass before attaching provenance: {audit_status}")
    audit_conclusion = _require_json_str(audit_payload, "audit_conclusion")
    if audit_conclusion != "all_pass":
        raise click.ClickException(
            f"spec audit audit_conclusion must be all_pass before attaching provenance: {audit_conclusion}"
        )
    confirmation_status = _require_json_str(audit_payload, "user_confirmation_status")
    if confirmation_status != "confirmed":
        raise click.ClickException(
            "spec audit user_confirmation_status must be confirmed before attaching provenance: "
            f"{confirmation_status}"
        )
    _require_spec_confirmation_table_hash(audit_payload, Path(spec_audit), _hash_file)
    blocking_findings = audit_payload.get("blocking_findings")
    if isinstance(blocking_findings, list) and blocking_findings:
        raise click.ClickException("spec audit has blocking findings")
    if blocking_findings is not None and not isinstance(blocking_findings, list):
        raise click.ClickException("blocking_findings must be a list")
    _reject_blocking_spec_audit_rows(audit_payload)

    run_spec_hash_path = run_path / "spec_hash.txt"
    if not run_spec_hash_path.exists():
        raise click.ClickException(f"missing spec_hash.txt in run directory: {run_dir}")
    run_spec_hash = run_spec_hash_path.read_text(encoding="utf-8").strip()
    audit_spec_hash = _require_json_str(audit_payload, "spec_hash")
    if audit_spec_hash != run_spec_hash:
        raise click.ClickException(f"spec audit hash mismatch: audit={audit_spec_hash}, run={run_spec_hash}")
    if runtime_audit is None:
        message = "runtime_audit.json is required before attaching provenance"
        if as_json:
            click.echo(json.dumps(_backtest_json_failure("runtime_audit_missing", message), indent=2))
            raise SystemExit(1)
        raise click.ClickException(message)
    runtime_payload: dict[str, object] | None = None
    run_compiled_plan_path = run_path / "compiled_plan.json"
    runtime_validation = validate_runtime_audit_file(
        runtime_audit,
        spec=StrategySpec.from_yaml(run_spec_path),
        compiled_plan=json.loads(run_compiled_plan_path.read_text(encoding="utf-8")),
        require_material_coverage=True,
    )
    if runtime_validation["status"] == "fail":
        raise click.ClickException(f"invalid runtime audit: {runtime_validation['errors']}")
    runtime_payload = json.loads(Path(runtime_audit).read_text(encoding="utf-8"))
    runtime_status = _require_json_str(runtime_payload, "status")
    if runtime_status != "pass":
        raise click.ClickException(f"runtime audit status must be pass before attaching provenance: {runtime_status}")
    if runtime_payload.get("runtime_semantics_pass") is not True:
        raise click.ClickException("runtime audit runtime_semantics_pass must be true before attaching provenance")
    _reject_blocking_runtime_audit_rows(runtime_payload)
    runtime_spec_hash = _require_json_str(runtime_payload, "spec_hash")
    if runtime_spec_hash != run_spec_hash:
        raise click.ClickException(f"runtime audit hash mismatch: audit={runtime_spec_hash}, run={run_spec_hash}")
    _require_runtime_audit_hashes(
        runtime_payload,
        spec_hash=run_spec_hash,
        spec_audit_path=Path(spec_audit),
        compiled_plan_path=run_compiled_plan_path,
        component_bundle_hashes=_run_component_bundle_hashes(run_path),
    )
    _require_runtime_source_hash(runtime_payload, Path(runtime_audit))

    conversation_hash = _require_json_str(audit_payload, "conversation_hash")
    catalog_hash = _require_json_str(catalog_payload, "catalog_hash")
    computed_catalog_hash = _catalog_hash(catalog_payload)
    if computed_catalog_hash != catalog_hash:
        raise click.ClickException(f"component catalog hash mismatch: stored={catalog_hash}, actual={computed_catalog_hash}")
    audit_catalog_hash = _require_json_str(audit_payload, "catalog_hash")
    if audit_catalog_hash != catalog_hash:
        raise click.ClickException(f"catalog hash mismatch: audit={audit_catalog_hash}, catalog={catalog_hash}")
    recipe_catalog_hash = _require_json_str(catalog_payload, "recipe_catalog_hash")
    computed_recipe_catalog_hash = _stable_hash(catalog_payload.get("recipes", []))
    if computed_recipe_catalog_hash != recipe_catalog_hash:
        raise click.ClickException(
            f"recipe catalog hash mismatch: stored={recipe_catalog_hash}, actual={computed_recipe_catalog_hash}"
        )
    _require_run_component_bundles_in_catalog(run_path, catalog_payload)

    attached = [
        "spec_audit.json",
        "conversation_hash.txt",
        "component_catalog_hash.txt",
        "recipe_catalog_hash.txt",
    ]
    if runtime_audit is not None:
        attached.insert(1, "runtime_audit.json")
    artifact_hashes_digest = _publish_provenance_artifacts(
        run_path,
        spec_audit_content=Path(spec_audit).read_bytes(),
        runtime_audit_content=Path(runtime_audit).read_bytes() if runtime_audit is not None else None,
        conversation_hash=conversation_hash,
        catalog_hash=catalog_hash,
        recipe_catalog_hash=recipe_catalog_hash,
    )

    result = {
        "status": "pass",
        "run_dir": str(run_path),
        "artifact_hashes_digest": artifact_hashes_digest,
        "attached": attached,
    }
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("Status: PASS")
        click.echo(f"Run dir: {run_path}")
        click.echo(f"Artifact hashes digest: {artifact_hashes_digest}")


def _require_json_str(payload: object, key: str) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), str) or not payload[key]:
        raise click.ClickException(f"{key} must be present")
    return payload[key]


def _reject_blocking_spec_audit_rows(audit_payload: object) -> None:
    if not isinstance(audit_payload, dict):
        raise click.ClickException("spec audit must be an object")
    blocking_lists = {
        "missing_user_requirements": "spec audit has missing user requirements",
        "agent_added_fields": "spec audit has agent-added fields",
        "contradictions": "spec audit has contradictions",
    }
    for key, message in blocking_lists.items():
        value = audit_payload.get(key)
        if isinstance(value, list) and value:
            raise click.ClickException(message)

    blocking_field_statuses = {"unconfirmed", "contradiction", "agent_added"}
    blocking_component_statuses = {"missing", "non_canonical"}
    for index, item in enumerate(audit_payload.get("field_audits", [])):
        if not isinstance(item, dict):
            continue
        if item.get("blocking") is True or item.get("status") in blocking_field_statuses:
            raise click.ClickException(f"spec audit has blocking field audit row: field_audits[{index}]")
    for index, item in enumerate(audit_payload.get("component_audits", [])):
        if not isinstance(item, dict):
            continue
        if item.get("blocking") is True or item.get("status") in blocking_component_statuses:
            raise click.ClickException(f"spec audit has blocking component audit row: component_audits[{index}]")


def _reject_blocking_runtime_audit_rows(audit_payload: object) -> None:
    if not isinstance(audit_payload, dict):
        raise click.ClickException("runtime audit must be an object")
    blocking_findings = audit_payload.get("blocking_findings")
    if isinstance(blocking_findings, list) and blocking_findings:
        raise click.ClickException("runtime audit has blocking findings")
    if blocking_findings is not None and not isinstance(blocking_findings, list):
        raise click.ClickException("blocking_findings must be a list")
    field_statuses = {"missing", "mismatch"}
    for index, item in enumerate(audit_payload.get("material_field_audits", [])):
        if not isinstance(item, dict):
            continue
        if item.get("blocking") is True or item.get("status") in field_statuses:
            raise click.ClickException(f"runtime audit has blocking material field row: material_field_audits[{index}]")


def _require_pre_backtest_spec_audit(spec: StrategySpec, spec_audit_path: Path) -> None:
    """Deterministically gate a formal backtest on a pre-run spec audit."""
    from oxq.spec.audit_schema import validate_spec_audit_file
    from oxq.spec.compiler import _hash_file

    workspace = _read_workspace_config(Path.cwd() / ".open-xquant" / "workspace.yaml")
    audit_validation = validate_spec_audit_file(
        spec_audit_path,
        spec=spec,
        require_confirmed_coverage=True,
        verify_confirmation_table=True,
        require_formal_provenance=_is_version_governed_workspace(workspace),
    )
    if audit_validation["status"] == "fail":
        raise click.ClickException(f"invalid spec audit: {audit_validation['errors']}")

    audit_payload = json.loads(spec_audit_path.read_text(encoding="utf-8"))
    audit_status = _require_json_str(audit_payload, "status")
    if audit_status != "pass":
        raise click.ClickException(f"spec audit status must be pass before backtest: {audit_status}")
    audit_conclusion = _require_json_str(audit_payload, "audit_conclusion")
    if audit_conclusion != "all_pass":
        raise click.ClickException(f"spec audit audit_conclusion must be all_pass before backtest: {audit_conclusion}")
    confirmation_status = _require_json_str(audit_payload, "user_confirmation_status")
    if confirmation_status != "confirmed":
        raise click.ClickException(
            "spec audit user_confirmation_status must be confirmed before backtest: "
            f"{confirmation_status}"
        )
    _require_spec_confirmation_table_hash(audit_payload, spec_audit_path, _hash_file)
    if audit_payload.get("spec_provenance_pass") is not True:
        raise click.ClickException("spec audit spec_provenance_pass must be true before backtest")
    blocking_findings = audit_payload.get("blocking_findings")
    if isinstance(blocking_findings, list) and blocking_findings:
        raise click.ClickException("spec audit has blocking findings")
    if blocking_findings is not None and not isinstance(blocking_findings, list):
        raise click.ClickException("blocking_findings must be a list")
    _reject_blocking_spec_audit_rows(audit_payload)

    audit_spec_hash = _require_json_str(audit_payload, "spec_hash")
    spec_hash = spec.compute_hash()
    if audit_spec_hash != spec_hash:
        raise click.ClickException(f"spec audit hash mismatch: audit={audit_spec_hash}, spec={spec_hash}")


def _require_pre_backtest_runtime_audit(
    spec: StrategySpec,
    runtime_audit_path: Path,
    *,
    spec_audit_path: Path | None,
    effective_data_dir: str | None,
    component_bundle_hashes: set[str] | None = None,
    component_manifests: list[dict] | None = None,
) -> None:
    """Deterministically gate a formal backtest on a pre-run runtime audit."""
    from oxq.spec.compiler import (
        _build_compiled_plan_from_spec_metadata,
        compile_plan,
    )
    from oxq.spec.runtime_audit_schema import validate_runtime_audit_file

    if component_manifests:
        compiled_plan_payload = _build_compiled_plan_from_spec_metadata(
            spec,
            effective_data_dir=effective_data_dir,
            component_manifests=component_manifests,
        )
    else:
        compiled_plan_payload = compile_plan(spec, effective_data_dir=effective_data_dir)
    audit_validation = validate_runtime_audit_file(
        runtime_audit_path,
        spec=spec,
        compiled_plan=compiled_plan_payload,
        require_material_coverage=True,
    )
    if audit_validation["status"] == "fail":
        raise click.ClickException(f"invalid runtime audit: {audit_validation['errors']}")

    audit_payload = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    audit_status = _require_json_str(audit_payload, "status")
    if audit_status != "pass":
        raise click.ClickException(f"runtime audit status must be pass before backtest: {audit_status}")
    if audit_payload.get("runtime_semantics_pass") is not True:
        raise click.ClickException("runtime audit runtime_semantics_pass must be true before backtest")
    _reject_blocking_runtime_audit_rows(audit_payload)

    audit_spec_hash = _require_json_str(audit_payload, "spec_hash")
    spec_hash = spec.compute_hash()
    if audit_spec_hash != spec_hash:
        raise click.ClickException(f"runtime audit hash mismatch: audit={audit_spec_hash}, spec={spec_hash}")
    _require_runtime_audit_hashes(
        audit_payload,
        spec_hash=spec_hash,
        spec_audit_path=spec_audit_path,
        compiled_plan_payload=compiled_plan_payload,
        component_bundle_hashes=component_bundle_hashes,
    )


def _require_component_catalog_before_import(
    component_catalog_path: Path,
    *,
    spec_audit_path: Path,
    component_bundle_hashes: set[str],
) -> None:
    from oxq.core.component_catalog import _catalog_hash, _stable_hash

    try:
        catalog_payload = json.loads(component_catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"component catalog is not valid JSON: {component_catalog_path}: {exc.msg}") from exc
    if not isinstance(catalog_payload, dict):
        raise click.ClickException("component catalog must be an object")
    catalog_hash = _require_json_str(catalog_payload, "catalog_hash")
    computed_catalog_hash = _catalog_hash(catalog_payload)
    if computed_catalog_hash != catalog_hash:
        raise click.ClickException(f"component catalog hash mismatch: stored={catalog_hash}, actual={computed_catalog_hash}")
    recipe_catalog_hash = _require_json_str(catalog_payload, "recipe_catalog_hash")
    computed_recipe_catalog_hash = _stable_hash(catalog_payload.get("recipes", []))
    if computed_recipe_catalog_hash != recipe_catalog_hash:
        raise click.ClickException(
            f"recipe catalog hash mismatch: stored={recipe_catalog_hash}, actual={computed_recipe_catalog_hash}"
        )

    audit_payload = json.loads(spec_audit_path.read_text(encoding="utf-8"))
    audit_catalog_hash = _require_json_str(audit_payload, "catalog_hash")
    if audit_catalog_hash != catalog_hash:
        raise click.ClickException(f"catalog hash mismatch: audit={audit_catalog_hash}, catalog={catalog_hash}")

    catalog_hashes = _catalog_component_bundle_hashes(catalog_payload)
    missing = set(component_bundle_hashes).difference(catalog_hashes)
    if missing:
        raise click.ClickException(
            "component bundle hash mismatch between authorized manifests and component catalog: "
            f"missing={sorted(missing)}, manifest={sorted(component_bundle_hashes)}, catalog={sorted(catalog_hashes)}"
        )


def _require_runtime_audit_hashes(
    audit_payload: dict[str, object],
    *,
    spec_hash: str,
    spec_audit_path: Path | None = None,
    compiled_plan_path: Path | None = None,
    compiled_plan_payload: object | None = None,
    component_bundle_hashes: set[str] | None = None,
) -> None:
    audit_spec_hash = _require_json_str(audit_payload, "spec_hash")
    if audit_spec_hash != spec_hash:
        raise click.ClickException(f"runtime audit hash mismatch: audit={audit_spec_hash}, spec={spec_hash}")
    if spec_audit_path is not None:
        from oxq.spec.compiler import _hash_json_file

        expected_spec_audit_hash = _hash_json_file(spec_audit_path)
        audit_spec_audit_hash = _require_json_str(audit_payload, "spec_audit_hash")
        if audit_spec_audit_hash != expected_spec_audit_hash:
            raise click.ClickException(
                "runtime audit spec_audit_hash mismatch: "
                f"audit={audit_spec_audit_hash}, expected={expected_spec_audit_hash}"
            )
    if compiled_plan_path is not None:
        if not compiled_plan_path.exists():
            raise click.ClickException(f"compiled_plan.json is required for runtime audit verification: {compiled_plan_path}")
        from oxq.spec.compiler import _hash_json_file

        expected_compiled_plan_hash = _hash_json_file(compiled_plan_path)
    elif compiled_plan_payload is not None:
        expected_compiled_plan_hash = _hash_json_payload(compiled_plan_payload)
    else:
        expected_compiled_plan_hash = ""
    if expected_compiled_plan_hash:
        audit_compiled_plan_hash = _require_json_str(audit_payload, "compiled_plan_hash")
        if audit_compiled_plan_hash != expected_compiled_plan_hash:
            raise click.ClickException(
                "runtime audit compiled_plan_hash mismatch: "
                f"audit={audit_compiled_plan_hash}, expected={expected_compiled_plan_hash}"
            )
    expected_component_hashes = sorted(component_bundle_hashes or set())
    if component_bundle_hashes is not None or "component_bundle_hashes" in audit_payload:
        audit_hashes = audit_payload.get("component_bundle_hashes")
        if not isinstance(audit_hashes, list) or not all(isinstance(item, str) for item in audit_hashes):
            raise click.ClickException("runtime audit component_bundle_hashes must list authorized component bundle hashes")
        normalized_audit_hashes = sorted(set(audit_hashes))
        if normalized_audit_hashes != expected_component_hashes:
            raise click.ClickException(
                "runtime audit component_bundle_hashes mismatch: "
                f"audit={normalized_audit_hashes}, expected={expected_component_hashes}"
            )


def _require_runtime_source_hash(audit_payload: dict[str, object], runtime_audit_path: Path) -> None:
    source_path_raw = _require_json_str(audit_payload, "strategy_source_path")
    source_path = Path(source_path_raw)
    if not source_path.is_absolute():
        audit_relative = runtime_audit_path.parent / source_path
        source_path = audit_relative if audit_relative.exists() else Path.cwd() / source_path
    if not source_path.is_file():
        raise click.ClickException(f"runtime audit strategy_source_path file not found: {source_path_raw}")
    expected_hash = f"sha256:{hashlib.sha256(source_path.read_bytes()).hexdigest()}"
    recorded_hash = _require_json_str(audit_payload, "strategy_source_hash")
    if recorded_hash != expected_hash:
        raise click.ClickException(
            f"runtime audit strategy_source_hash mismatch: audit={recorded_hash}, expected={expected_hash}"
        )


def _require_backtest_authorization(
    authorization_path: Path | None,
    *,
    spec: StrategySpec,
    spec_path: Path,
    spec_audit_path: Path | None,
    runtime_audit_path: Path | None,
    component_catalog_path: Path | None,
    component_manifest_paths: tuple[str, ...],
    data_dir: str | None,
    run_out: Path,
) -> None:
    """Require coordinator authorization before an audited formal backtest."""
    if authorization_path is None or not authorization_path.exists():
        raise click.ClickException(
            "backtest_authorization.json is required before an audited formal backtest"
        )
    if spec_audit_path is None:
        raise click.ClickException("spec_audit.json is required by backtest authorization")
    if runtime_audit_path is None:
        raise click.ClickException("runtime_audit.json is required by backtest authorization")
    if component_catalog_path is None:
        raise click.ClickException("component_catalog.json is required by backtest authorization")
    try:
        payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"backtest_authorization.json is not valid JSON: {authorization_path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise click.ClickException("backtest_authorization.json must be an object")
    status = _require_json_str(payload, "status")
    if status != "authorized":
        raise click.ClickException(f"backtest authorization status must be authorized: {status}")

    _require_authorized_path(payload, "strategy_spec", spec_path, authorization_path)
    _require_authorized_path(payload, "spec_audit", spec_audit_path, authorization_path)
    _require_authorized_path(payload, "runtime_audit", runtime_audit_path, authorization_path)
    _require_authorized_path(payload, "component_catalog", component_catalog_path, authorization_path)
    _require_authorized_path(payload, "run_out", run_out, authorization_path)
    _require_authorized_path(
        payload,
        "data_dir",
        Path(_resolve_effective_data_dir(spec, data_dir)),
        authorization_path,
    )
    _require_authorized_manifest_paths(payload, component_manifest_paths, authorization_path)

    from oxq.spec.compiler import _hash_json_file

    expected = {
        "spec_hash": spec.compute_hash(),
        "spec_audit_hash": _hash_json_file(spec_audit_path),
        "runtime_audit_hash": _hash_json_file(runtime_audit_path),
    }
    for field, value in expected.items():
        recorded = _require_json_str(payload, field)
        if recorded != value:
            raise click.ClickException(
                f"backtest authorization {field} mismatch: authorization={recorded}, expected={value}"
            )

    from oxq.spec.runtime_audit_schema import validate_strategy_source_presentation

    presentation_validation = validate_strategy_source_presentation(
        payload,
        authorization_path=authorization_path,
        runtime_audit_path=runtime_audit_path,
        run_out=run_out,
    )
    if presentation_validation["status"] == "fail":
        raise click.ClickException(
            f"invalid strategy_source_presentation: {presentation_validation['errors']}"
        )


def _require_authorized_path(
    payload: dict[str, object],
    field: str,
    expected: Path,
    authorization_path: Path,
) -> None:
    raw = _require_json_str(payload, field)
    if not _authorization_path_matches(raw, expected, authorization_path):
        raise click.ClickException(
            f"backtest authorization {field} mismatch: authorization={raw}, expected={expected}"
        )


def _require_authorized_manifest_paths(
    payload: dict[str, object],
    component_manifest_paths: tuple[str, ...],
    authorization_path: Path,
) -> None:
    raw_paths = payload.get("component_manifests")
    if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
        raise click.ClickException("backtest authorization component_manifests must be a list of strings")
    recorded = {
        str(_authorization_resolve_path(Path(item), authorization_path))
        for item in raw_paths
    }
    expected = {str(_resolve_path_for_compare(Path(item))) for item in component_manifest_paths}
    if recorded != expected:
        raise click.ClickException(
            "backtest authorization component_manifests mismatch: "
            f"authorization={sorted(recorded)}, expected={sorted(expected)}"
        )


def _authorization_path_matches(raw_path: str, expected: Path, authorization_path: Path) -> bool:
    expected_path = _resolve_path_for_compare(expected)
    recorded = Path(raw_path)
    candidates = [recorded] if recorded.is_absolute() else [authorization_path.parent / recorded, Path.cwd() / recorded]
    return any(_resolve_path_for_compare(candidate) == expected_path for candidate in candidates)


def _authorization_resolve_path(path: Path, authorization_path: Path) -> Path:
    if path.is_absolute():
        return _resolve_path_for_compare(path)
    auth_relative = authorization_path.parent / path
    if auth_relative.exists():
        return _resolve_path_for_compare(auth_relative)
    return _resolve_path_for_compare(Path.cwd() / path)


def _resolve_path_for_compare(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _normalize_spec_for_run(spec: StrategySpec) -> StrategySpec:
    """Normalize a spec with the same serialization boundary as run artifacts."""
    return StrategySpec.from_dict(spec.to_dict())


def _attach_provenance_artifacts(
    run_path: Path,
    *,
    spec_audit_path: Path,
    runtime_audit_path: Path,
    component_catalog_path: Path | None,
) -> None:
    from oxq.core.component_catalog import _catalog_hash, _stable_hash
    from oxq.spec.runtime_audit_schema import validate_runtime_audit_file

    if component_catalog_path is None:
        raise click.ClickException("component_catalog.json is required for formal run provenance")
    artifact_hashes_path = run_path / "artifact_hashes.json"
    if not artifact_hashes_path.exists():
        raise click.ClickException(f"missing artifact_hashes.json in run directory: {run_path}")
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    if not isinstance(artifact_hashes, dict):
        raise click.ClickException("artifact_hashes.json must be an object")

    run_spec = StrategySpec.from_yaml(run_path / "strategy_spec.yaml")
    _require_pre_backtest_spec_audit(run_spec, spec_audit_path)
    runtime_validation = validate_runtime_audit_file(runtime_audit_path)
    if runtime_validation["status"] == "fail":
        raise click.ClickException(f"invalid runtime audit: {runtime_validation['errors']}")

    audit_payload = json.loads(spec_audit_path.read_text(encoding="utf-8"))
    runtime_payload = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
    runtime_status = _require_json_str(runtime_payload, "status")
    if runtime_status != "pass":
        raise click.ClickException(f"runtime audit status must be pass before attaching provenance: {runtime_status}")
    catalog_payload = json.loads(component_catalog_path.read_text(encoding="utf-8"))
    run_spec_hash = (run_path / "spec_hash.txt").read_text(encoding="utf-8").strip()
    if run_spec.compute_hash() != run_spec_hash:
        raise click.ClickException(
            f"run spec hash mismatch: spec={run_spec.compute_hash()}, artifact={run_spec_hash}"
        )
    _require_runtime_audit_hashes(
        runtime_payload,
        spec_hash=run_spec_hash,
        spec_audit_path=spec_audit_path,
        compiled_plan_path=run_path / "compiled_plan.json",
        component_bundle_hashes=_run_component_bundle_hashes(run_path),
    )
    _require_runtime_source_hash(runtime_payload, runtime_audit_path)
    if runtime_payload.get("runtime_semantics_pass") is not True:
        raise click.ClickException("runtime audit runtime_semantics_pass must be true before attaching provenance")
    _reject_blocking_runtime_audit_rows(runtime_payload)

    catalog_hash = _require_json_str(catalog_payload, "catalog_hash")
    computed_catalog_hash = _catalog_hash(catalog_payload)
    if computed_catalog_hash != catalog_hash:
        raise click.ClickException(f"component catalog hash mismatch: stored={catalog_hash}, actual={computed_catalog_hash}")
    audit_catalog_hash = _require_json_str(audit_payload, "catalog_hash")
    if audit_catalog_hash != catalog_hash:
        raise click.ClickException(f"catalog hash mismatch: audit={audit_catalog_hash}, catalog={catalog_hash}")
    recipe_catalog_hash = _require_json_str(catalog_payload, "recipe_catalog_hash")
    computed_recipe_catalog_hash = _stable_hash(catalog_payload.get("recipes", []))
    if computed_recipe_catalog_hash != recipe_catalog_hash:
        raise click.ClickException(
            f"recipe catalog hash mismatch: stored={recipe_catalog_hash}, actual={computed_recipe_catalog_hash}"
        )
    _require_run_component_bundles_in_catalog(run_path, catalog_payload)

    _publish_provenance_artifacts(
        run_path,
        spec_audit_content=spec_audit_path.read_bytes(),
        runtime_audit_content=runtime_audit_path.read_bytes(),
        conversation_hash=_require_json_str(audit_payload, "conversation_hash"),
        catalog_hash=catalog_hash,
        recipe_catalog_hash=recipe_catalog_hash,
    )


def _publish_provenance_artifacts(
    run_path: Path,
    *,
    spec_audit_content: bytes,
    runtime_audit_content: bytes | None,
    conversation_hash: str,
    catalog_hash: str,
    recipe_catalog_hash: str,
) -> str:
    from oxq.run_digests import publish_run_artifacts

    artifacts = {
        "spec_audit.json": spec_audit_content,
        "conversation_hash.txt": (conversation_hash + "\n").encode(),
        "component_catalog_hash.txt": (catalog_hash + "\n").encode(),
        "recipe_catalog_hash.txt": (recipe_catalog_hash + "\n").encode(),
    }
    canonical_json = {"spec_audit.json"}
    if runtime_audit_content is not None:
        artifacts["runtime_audit.json"] = runtime_audit_content
        canonical_json.add("runtime_audit.json")
    return publish_run_artifacts(run_path, artifacts, canonical_json=canonical_json)


def _hash_json_payload(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _require_spec_confirmation_table_hash(
    audit_payload: dict[str, object],
    spec_audit_path: Path,
    hash_file,
) -> None:
    table = audit_payload.get("spec_confirmation_table")
    if not isinstance(table, dict):
        raise click.ClickException("spec audit spec_confirmation_table must be present before backtest")
    raw_path = table.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise click.ClickException("spec audit spec_confirmation_table.path must be present before backtest")
    recorded_hash = table.get("hash")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise click.ClickException("spec audit spec_confirmation_table.hash must be present before backtest")
    table_path = _resolve_audit_artifact_path(raw_path, spec_audit_path)
    if not table_path.exists():
        raise click.ClickException(f"spec confirmation table not found: {raw_path}")
    actual_hash = hash_file(table_path)
    actual_full_hash = f"sha256:{hashlib.sha256(table_path.read_bytes()).hexdigest()}"
    if recorded_hash not in {actual_hash, actual_full_hash}:
        raise click.ClickException(
            "spec confirmation table hash mismatch: "
            f"audit={recorded_hash}, actual={actual_hash}"
        )


def _resolve_audit_artifact_path(raw_path: str, audit_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    audit_relative = audit_path.parent / path
    if audit_relative.exists():
        return audit_relative
    return Path.cwd() / path


def _resolve_effective_data_dir(spec: StrategySpec, data_dir: str | None) -> str:
    from oxq.data.loaders import resolve_data_dir

    raw_data_dir = data_dir or (spec.data.data_dir or None)
    return str(resolve_data_dir(Path(raw_data_dir) if raw_data_dir else None).resolve())


def _require_run_component_bundles_in_catalog(run_path: Path, catalog_payload: object) -> None:
    run_hashes = _run_component_bundle_hashes(run_path)
    catalog_hashes = _catalog_component_bundle_hashes(catalog_payload)
    if not run_hashes:
        return
    missing = run_hashes.difference(catalog_hashes)
    if missing:
        raise click.ClickException(
            "component bundle hash mismatch between run artifacts and component catalog: "
            f"missing={sorted(missing)}, run={sorted(run_hashes)}, catalog={sorted(catalog_hashes)}"
        )


def _run_component_bundle_hashes(run_path: Path) -> set[str]:
    from oxq.core.component_manifest import compute_component_bundle_hash

    hashes: set[str] = set()
    summary_path = run_path / "component_manifests.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"component_manifests.json is not valid JSON: {summary_path}: {exc.msg}") from exc
        if not isinstance(summary, list):
            raise click.ClickException("component_manifests.json must be a list")
        for index, item in enumerate(summary):
            if not isinstance(item, dict):
                raise click.ClickException(f"component_manifests.json[{index}] must be an object")
            recorded = item.get("bundle_hash")
            if not isinstance(recorded, str) or not recorded:
                raise click.ClickException(f"component_manifests.json[{index}].bundle_hash is required")
            manifest_path = _resolve_run_component_manifest_path(run_path, item, len(summary))
            if manifest_path is not None:
                actual = _verified_component_bundle_hash(manifest_path, recorded)
                if actual != recorded:
                    raise click.ClickException(
                        f"component bundle {index} hash mismatch: stored={recorded}, actual={actual}"
                    )
            hashes.add(recorded)
    manifest_path = run_path / "component_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"component_manifest.json is not valid JSON: {manifest_path}: {exc.msg}") from exc
        if isinstance(manifest, dict) and isinstance(manifest.get("bundle_hash"), str) and manifest["bundle_hash"]:
            recorded = manifest["bundle_hash"]
            try:
                actual = compute_component_bundle_hash(manifest_path)
            except ValueError as exc:
                raise click.ClickException(f"component bundle could not be verified: {exc}") from exc
            if actual != recorded:
                raise click.ClickException(f"component bundle hash mismatch: stored={recorded}, actual={actual}")
            hashes.add(recorded)
    bundle_hash_path = run_path / "component_bundle_hash.txt"
    if bundle_hash_path.exists():
        digest = bundle_hash_path.read_text(encoding="utf-8").strip()
        if digest:
            if hashes and digest not in hashes:
                raise click.ClickException(
                    "component_bundle_hash.txt mismatch: "
                    f"stored={digest}, verified_component_bundles={sorted(hashes)}"
                )
            hashes.add(digest)
    return hashes


def _resolve_run_component_manifest_path(run_path: Path, item: dict[str, object], summary_count: int) -> Path | None:
    archived_path = item.get("archived_manifest_path")
    if isinstance(archived_path, str) and archived_path:
        return _safe_run_relative_component_manifest(run_path, archived_path)

    legacy_manifest = run_path / "component_manifest.json"
    if summary_count == 1 and legacy_manifest.exists():
        return legacy_manifest

    manifest_path = item.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        return None
    resolved = Path(manifest_path)
    if not resolved.is_absolute():
        resolved = run_path / resolved
    return resolved if resolved.exists() else None


def _safe_run_relative_component_manifest(run_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise click.ClickException(f"archived component manifest path is unsafe: {raw_path}")
    candidate = run_path / path
    if candidate.is_symlink():
        raise click.ClickException(f"archived component manifest path must not be a symlink: {raw_path}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(run_path.resolve()):
        raise click.ClickException(f"archived component manifest path escapes run directory: {raw_path}")
    if not candidate.exists():
        raise click.ClickException(f"archived component manifest not found: {candidate}")
    return candidate


def _verified_component_bundle_hash(manifest_path: Path, recorded: str) -> str:
    from oxq.core.component_manifest import compute_component_bundle_hash

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"component manifest is not valid JSON: {manifest_path}: {exc.msg}") from exc
    if not isinstance(manifest, dict):
        raise click.ClickException(f"component manifest must be an object: {manifest_path}")
    manifest_hash = manifest.get("bundle_hash")
    if manifest_hash != recorded:
        raise click.ClickException(
            f"component bundle manifest hash mismatch: stored={recorded}, manifest={manifest_hash}"
        )
    try:
        return compute_component_bundle_hash(manifest_path)
    except ValueError as exc:
        raise click.ClickException(f"component bundle could not be verified: {exc}") from exc


def _component_bundle_hashes(manifests: list[dict]) -> set[str]:
    hashes: set[str] = set()
    for manifest in manifests:
        digest = manifest.get("bundle_hash")
        if isinstance(digest, str) and digest:
            hashes.add(digest)
    return hashes


def _catalog_component_bundle_hashes(catalog_payload: object) -> set[str]:
    if not isinstance(catalog_payload, dict):
        raise click.ClickException("component catalog must be an object")
    hashes: set[str] = set()
    for section in ("indicators", "signals", "portfolios", "rules"):
        entries = catalog_payload.get(section)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            is_workspace_component = entry.get("source") == "workspace_extension"
            if is_workspace_component and isinstance(entry.get("bundle_hash"), str) and entry["bundle_hash"]:
                hashes.add(entry["bundle_hash"])
    return hashes


def _default_spec_audit_path(spec_path: Path) -> Path | None:
    candidate = spec_path.parent / "spec_audit.json"
    return candidate if candidate.exists() else None


def _default_runtime_audit_path(spec_path: Path) -> Path | None:
    candidate = spec_path.parent / "runtime_audit.json"
    return candidate if candidate.exists() else None


def _default_backtest_authorization_path(runtime_audit_path: Path | None) -> Path | None:
    if runtime_audit_path is None:
        return None
    return runtime_audit_path.parent / "backtest_authorization.json"


def _default_component_catalog_path(spec_path: Path) -> Path | None:
    candidate = spec_path.parent / "component_catalog.json"
    return candidate if candidate.exists() else None


def _resolve_backtest_output_dir(out: str | None) -> str:
    workspace = _read_workspace_config(Path.cwd() / ".open-xquant" / "workspace.yaml")
    if not workspace:
        return out or "runs/auto"
    workflow = workspace.get("workflow")
    configured = workflow.get("default_output_dir") if isinstance(workflow, dict) else None
    if not _is_version_governed_workspace(workspace):
        if out:
            return out
        if not isinstance(configured, str) or not configured:
            return "runs/auto"
        return configured

    active_version = _workspace_active_version(workspace)
    if not active_version:
        raise click.ClickException(
            "version-governed workspace requires current.json active_version; "
            "run `oxq research init` or repair current.json"
        )
    governed_output = _workspace_active_backtest_output(workspace, active_version)
    governed_default = _workspace_default_backtest_output_template(workspace)
    if isinstance(configured, str) and configured in {
        "versions/{active_version}/09_backtests",
        "runs/auto",
        "runs/auto/runs/runs/{active_version}",
        "runs",
        "runs/{active_version}",
    }:
        configured = governed_output
    elif configured == governed_default:
        configured = configured.replace("{active_version}", active_version)
    elif isinstance(configured, str) and "{active_version}" in configured:
        configured = configured.replace("{active_version}", active_version)

    candidate = out or (configured if isinstance(configured, str) and configured else governed_output)
    candidate_path = Path(candidate)
    if ".." in candidate_path.parts:
        raise click.ClickException(
            "backtest output must stay within the active version phase_paths.09_backtests"
        )
    workspace_root = Path.cwd().resolve()
    resolved_candidate = (
        candidate_path.resolve(strict=False)
        if candidate_path.is_absolute()
        else (Path.cwd() / candidate_path).resolve(strict=False)
    )
    governed_path = (Path.cwd() / governed_output).resolve(strict=False)
    try:
        governed_path.relative_to(workspace_root)
        resolved_candidate.relative_to(workspace_root)
        resolved_candidate.relative_to(governed_path)
    except ValueError as exc:
        raise click.ClickException(
            "backtest output must stay within the active version phase_paths.09_backtests"
        ) from exc
    return candidate


def _require_complete_governed_workspace() -> dict[str, Path] | None:
    workspace_root = Path.cwd().resolve()
    workspace = _read_workspace_config(workspace_root / ".open-xquant" / "workspace.yaml")
    if not _is_version_governed_workspace(workspace):
        return None

    from oxq.spec.audit_schema import _active_governed_provenance_paths

    phase_paths, errors = _active_governed_provenance_paths(workspace_root)
    if errors:
        first = errors[0]
        raise click.ClickException(f"{first['path']}: {first['message']}")
    if phase_paths is None:
        raise click.ClickException("version-governed workspace governance is incomplete")
    return phase_paths


def _require_governed_formal_gate_paths(
    *,
    spec_path: Path,
    spec_audit_path: Path | None,
    runtime_audit_path: Path | None,
    component_catalog_path: Path | None,
    authorization_path: Path | None,
    component_manifest_paths: tuple[str, ...],
    run_out: Path,
) -> None:
    workspace_root = Path.cwd().resolve()
    workspace = _read_workspace_config(workspace_root / ".open-xquant" / "workspace.yaml")
    if not _is_version_governed_workspace(workspace):
        return
    _require_complete_governed_workspace()

    active_version = _workspace_active_version(workspace)
    versions_dir = _workspace_versions_dir(workspace)
    version_dir = _resolve_active_version_dir(workspace_root, versions_dir, active_version)
    manifest_path = version_dir / "version_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise click.ClickException(
            "active version_manifest.json must contain a valid JSON object"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("version_id") != active_version:
        raise click.ClickException(
            "active version_manifest.json must match current.json active_version"
        )
    phase_paths = manifest.get("phase_paths")
    if not isinstance(phase_paths, dict):
        raise click.ClickException("active version_manifest.json phase_paths must be an object")

    resolved_phases: dict[str, Path] = {}
    for phase in ("04_spec_build", "06_spec_audit", "07_compile_preview", "08_runtime_audit", "09_backtests"):
        raw_phase_path = phase_paths.get(phase)
        if not isinstance(raw_phase_path, str) or not raw_phase_path:
            raise click.ClickException(
                f"active version_manifest.json requires phase_paths.{phase}"
            )
        resolved_phases[phase] = _resolve_version_phase_path(
            workspace_root,
            version_dir,
            phase,
            raw_phase_path,
        )

    artifact_bindings = (
        ("strategy spec", spec_path, "04_spec_build", "strategy_spec.yaml"),
        ("spec audit", spec_audit_path, "06_spec_audit", "spec_audit.json"),
        ("component catalog", component_catalog_path, "04_spec_build", "component_catalog.json"),
        ("runtime audit", runtime_audit_path, "08_runtime_audit", "runtime_audit.json"),
        ("backtest authorization", authorization_path, "08_runtime_audit", "backtest_authorization.json"),
    )
    for label, actual_path, phase, filename in artifact_bindings:
        if actual_path is None:
            raise click.ClickException(f"{label} is required for a governed formal backtest")
        phase_path = resolved_phases[phase]
        expected_path = (phase_path / filename).resolve(strict=False)
        try:
            expected_path.relative_to(phase_path)
        except ValueError as exc:
            raise click.ClickException(
                f"{label} must stay within active version phase_paths.{phase}"
            ) from exc
        actual_resolved = _resolve_path_for_compare(actual_path)
        if actual_resolved != expected_path:
            raise click.ClickException(
                f"{label} must be active version {active_version} "
                f"phase_paths.{phase}/{filename}"
            )

    expected_run_out = resolved_phases["09_backtests"]
    if _resolve_path_for_compare(run_out) != expected_run_out:
        raise click.ClickException(
            f"run output must be active version {active_version} phase_paths.09_backtests"
        )

    components_dir = _configured_workspace_path(workspace, "components_dir") or Path("components")
    components_root = (workspace_root / components_dir).resolve(strict=False)
    try:
        components_root.relative_to(workspace_root)
    except ValueError as exc:
        raise click.ClickException("workspace paths.components_dir must stay within the workspace") from exc
    for manifest_path in component_manifest_paths:
        resolved_manifest = _resolve_path_for_compare(Path(manifest_path))
        try:
            resolved_manifest.relative_to(components_root)
        except ValueError as exc:
            raise click.ClickException(
                "component manifest must resolve inside workspace paths.components_dir"
            ) from exc

    assert spec_audit_path is not None
    spec_audit_payload = _read_required_json_object(spec_audit_path, "spec audit")
    table = spec_audit_payload.get("spec_confirmation_table")
    table_path_raw = table.get("path") if isinstance(table, dict) else None
    table_path = _resolve_governed_nested_reference(
        table_path_raw,
        workspace_root=workspace_root,
        artifact_parent=spec_audit_path.parent,
        label="confirmation table",
    )
    try:
        table_path.relative_to(resolved_phases["06_spec_audit"])
    except ValueError as exc:
        raise click.ClickException(
            "confirmation table must stay within active version phase_paths.06_spec_audit"
        ) from exc

    assert runtime_audit_path is not None
    runtime_audit_payload = _read_required_json_object(runtime_audit_path, "runtime audit")
    compiled_plan_path = _resolve_governed_nested_reference(
        runtime_audit_payload.get("compiled_plan_path"),
        workspace_root=workspace_root,
        artifact_parent=runtime_audit_path.parent,
        label="compiled plan",
    )
    expected_compiled_plan = (resolved_phases["07_compile_preview"] / "compiled_plan.json").resolve(strict=False)
    if compiled_plan_path != expected_compiled_plan:
        raise click.ClickException(
            "compiled plan must be active version "
            f"{active_version} phase_paths.07_compile_preview/compiled_plan.json"
        )


def _read_required_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"{label} must contain a valid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"{label} must contain a valid JSON object: {path}")
    return payload


def _resolve_governed_nested_reference(
    raw_path: object,
    *,
    workspace_root: Path,
    artifact_parent: Path,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise click.ClickException(f"{label} path must be a non-empty string")
    path = Path(raw_path)
    if ".." in path.parts:
        raise click.ClickException(f"{label} path must not contain traversal")
    if path.is_absolute():
        resolved = path.resolve(strict=False)
    else:
        workspace_candidate = workspace_root / path
        resolved = (
            workspace_candidate.resolve(strict=False)
            if workspace_candidate.exists()
            else (artifact_parent / path).resolve(strict=False)
        )
    if not resolved.is_file():
        raise click.ClickException(f"{label} file not found: {raw_path}")
    return resolved


def _workspace_active_version(workspace: dict) -> str:
    paths = workspace.get("paths")
    if not isinstance(paths, dict):
        paths = {}
    current_path = _workspace_root_manifest_path(paths, "current_manifest", "current.json")
    current = _read_json_object(current_path)
    active_version = current.get("active_version")
    if not isinstance(active_version, str) or not active_version:
        return ""
    if not _WORKSPACE_VERSION_RE.fullmatch(active_version):
        raise click.ClickException(f"workspace current.json active_version is unsafe: {active_version}")
    return active_version


def _workspace_versions_dir(workspace: dict) -> Path:
    paths = workspace.get("paths")
    raw_value = "versions"
    if isinstance(paths, dict) and isinstance(paths.get("versions_dir"), str):
        raw_value = paths["versions_dir"]
    path = Path(raw_value)
    if not raw_value or path.is_absolute() or ".." in path.parts:
        raise click.ClickException("workspace paths.versions_dir must be a safe relative path")
    workspace_root = Path.cwd().resolve()
    if _path_has_symlink_component(Path.cwd() / path, workspace_root):
        raise click.ClickException("workspace paths.versions_dir must not contain symlink components")
    candidate = (Path.cwd() / path).resolve(strict=False)
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        raise click.ClickException("workspace paths.versions_dir must stay within the workspace")
    return path


def _configured_workspace_path(workspace: dict, key: str) -> Path | None:
    paths = workspace.get("paths")
    raw_value = paths.get(key) if isinstance(paths, dict) else None
    if not isinstance(raw_value, str) or not raw_value:
        return None
    path = Path(raw_value)
    if path.is_absolute() or ".." in path.parts:
        raise click.ClickException(f"workspace paths.{key} must be a safe relative path")
    try:
        (Path.cwd() / path).resolve(strict=False).relative_to(Path.cwd().resolve())
    except ValueError as exc:
        raise click.ClickException(f"workspace paths.{key} must stay within the workspace") from exc
    return path


def _workspace_default_backtest_output_template(workspace: dict) -> str:
    return f"{_workspace_versions_dir(workspace).as_posix()}/{{active_version}}/09_backtests"


def _workspace_active_backtest_output(workspace: dict, active_version: str) -> str:
    versions_dir = _workspace_versions_dir(workspace)
    version_dir = _resolve_active_version_dir(Path.cwd(), versions_dir, active_version)
    manifest_path = version_dir / "version_manifest.json"
    fallback = (version_dir / "09_backtests").relative_to(Path.cwd()).as_posix()
    if not manifest_path.exists():
        _resolve_version_phase_path(
            Path.cwd(),
            version_dir,
            "09_backtests",
            fallback,
        )
        return fallback
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise click.ClickException("active version_manifest.json must contain a valid JSON object") from exc
    if not isinstance(manifest, dict) or manifest.get("version_id") != active_version:
        raise click.ClickException("active version_manifest.json must match current.json active_version")
    phase_paths = manifest.get("phase_paths")
    raw_output = phase_paths.get("09_backtests") if isinstance(phase_paths, dict) else None
    if not isinstance(raw_output, str) or not raw_output:
        raise click.ClickException("active version_manifest.json requires phase_paths.09_backtests")
    _resolve_version_phase_path(
        Path.cwd(),
        version_dir,
        "09_backtests",
        raw_output,
    )
    return Path(raw_output).as_posix()


def _workspace_root_manifest_path(paths: dict, key: str, filename: str) -> Path:
    raw_value = paths.get(key) or filename
    if not isinstance(raw_value, str) or not raw_value:
        raw_value = filename
    path = Path(raw_value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != filename:
        raise click.ClickException(f"workspace paths.{key} must be {filename} at the workspace root")
    return Path.cwd() / filename


def _read_workspace_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise click.ClickException(f"workspace config is invalid: {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise click.ClickException(f"workspace config is invalid: {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise click.ClickException(f"workspace config must be a YAML object: {path}")
    return payload


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@main.group()
def strategy():
    """Manage compiled strategies."""


@strategy.command()
@click.argument("spec_file", type=click.Path(exists=True))
@click.option(
    "--data-dir",
    default=None,
    help=(
        "Directory for market data files. Use the same value as the formal "
        "backtest run when writing compile preview artifacts."
    ),
)
@click.option(
    "--component-manifest",
    "component_manifest",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Workspace component manifest to load before compile.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Write deterministic compile preview artifacts to this directory.",
)
def compile(spec_file: str, data_dir: str | None, component_manifest: tuple[str, ...], out: str | None):
    """Compile a strategy spec into an executable strategy.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    from oxq.spec.compiler import _build_strategy_py_artifact, compile_plan, compile_strategy

    loaded_component_manifests = _load_component_manifests(component_manifest)
    spec = StrategySpec.from_yaml(spec_file)
    validation = validate_spec(spec)
    if validation.status == "fail":
        click.echo("Spec validation failed:")
        for e in validation.errors:
            click.echo(f"  [{e['severity']}] {e['check']}: {e['message']}")
        raise SystemExit(1)

    strategy_obj = compile_strategy(spec)
    click.echo(f"Strategy '{strategy_obj.name}' compiled successfully.")
    click.echo(f"  Universe:  {spec.universe.type} ({len(spec.universe.symbols)} symbols)")
    click.echo(f"  Signals:   {list(spec.signal.rules.keys())}")
    click.echo(f"  Portfolio: {spec.portfolio.type}")
    click.echo(f"  Hash:      {spec.compute_hash()}")
    if out:
        out_dir = Path(out)
        effective_data_dir = _resolve_effective_data_dir(spec, data_dir)
        plan = compile_plan(spec, effective_data_dir=effective_data_dir)
        source_spec = Path(spec_file)
        compiled_plan_hash = _hash_json_payload(plan)
        _publish_compile_preview(
            out_dir,
            {
                "compiled_plan.json": (
                    json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n"
                ).encode(),
                "spec_hash.txt": (spec.compute_hash() + "\n").encode(),
                "strategy_spec.yaml": source_spec.read_bytes(),
                "strategy.py": _build_strategy_py_artifact(
                    spec,
                    plan,
                    spec.compute_hash(),
                    compiled_plan_hash,
                ).encode(),
            },
            loaded_component_manifests,
        )
        click.echo(f"  Compile preview: {out_dir / 'compiled_plan.json'}")
        click.echo(f"  Python source preview: {out_dir / 'strategy.py'}")
        click.echo(f"  Effective data dir: {effective_data_dir}")
        click.echo("  Note: effective data_dir is included in compiled_plan.json and its hash.")


@main.group()
def registry():
    """Inspect deterministic component registry artifacts."""


@registry.command(name="export")
@click.option("--out", "-o", required=True, type=click.Path(dir_okay=False), help="Output component catalog JSON path.")
@click.option(
    "--component-manifest",
    "component_manifest",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Workspace component manifest to load and include in the catalog.",
)
def registry_export(out: str, component_manifest: tuple[str, ...]):
    """Export registered components and canonical recipes.

    This command performs no semantic strategy matching. It writes the current
    registry/catalog artifact for Agents and Studio gates to consume.
    """
    from oxq.core.component_catalog import build_component_catalog, component_catalog_json

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifests = _load_component_manifests(component_manifest)
    catalog = build_component_catalog(manifests)
    output_path.write_text(component_catalog_json(catalog), encoding="utf-8")
    click.echo(f"Component catalog written to {output_path}")
    click.echo(f"Catalog hash: {catalog['catalog_hash']}")


@main.group(name="component-manifest")
def component_manifest_group():
    """Validate and hash workspace-local component extension manifests."""


@component_manifest_group.command(name="hash")
@click.argument("manifest_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def component_manifest_hash(manifest_file: str, as_json: bool):
    """Compute a component extension bundle hash."""
    from oxq.core.component_manifest import compute_component_bundle_hash

    digest = compute_component_bundle_hash(manifest_file)
    if as_json:
        click.echo(json.dumps({"component_bundle_hash": digest}, indent=2))
    else:
        click.echo(digest)


@component_manifest_group.command(name="validate")
@click.argument("manifest_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def component_manifest_validate(manifest_file: str, as_json: bool):
    """Validate a component extension manifest hash and importability."""
    from oxq.core.component_manifest import component_manifest_summary, load_component_manifest, scoped_component_registries

    try:
        with scoped_component_registries():
            load_component_manifest(manifest_file, verify_hash=True)
            result = component_manifest_summary(manifest_file)
        result["importable"] = True
    except Exception as exc:
        result = {
            "status": "fail",
            "manifest": str(Path(manifest_file).resolve()),
            "importable": False,
            "errors": [{"message": str(exc)}],
        }
    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        if result.get("bundle_hash"):
            click.echo(f"Bundle hash: {result['bundle_hash']}")
        for error in result.get("errors", []):
            click.echo(f"  {error['message']}")
    if result["status"] == "fail":
        raise SystemExit(1)


@main.group(name="spec-audit")
def spec_audit():
    """Validate Agent-authored spec audit artifacts."""


@spec_audit.command(name="validate")
@click.argument("audit_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(exists=True, dir_okay=False),
    help="strategy_spec.yaml for strict effective field confirmation coverage.",
)
@click.option(
    "--component-catalog",
    "component_catalog_path",
    type=click.Path(exists=True, dir_okay=False),
    help="component_catalog.json used by the audited spec gate.",
)
@click.option(
    "--mapping-contract",
    "mapping_contract_path",
    type=click.Path(exists=True, dir_okay=False),
    help="spec_mapping_contract.json used by the builder-to-auditor handoff.",
)
@click.option(
    "--strict-confirmed",
    is_flag=True,
    help="Require every effective strategy spec field to have a confirmed audit row.",
)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def spec_audit_validate(
    audit_file: str,
    spec_path: str | None,
    component_catalog_path: str | None,
    mapping_contract_path: str | None,
    strict_confirmed: bool,
    as_json: bool,
):
    """Validate spec_audit.json schema without semantic language judgment."""
    from oxq.spec.audit_schema import validate_spec_audit_file

    result = validate_spec_audit_file(
        audit_file,
        spec_path=spec_path,
        component_catalog_path=component_catalog_path,
        require_confirmed_coverage=strict_confirmed,
        verify_confirmation_table=strict_confirmed,
        mapping_contract_path=mapping_contract_path,
        require_formal_provenance=strict_confirmed,
    )
    try:
        audit_payload = json.loads(Path(audit_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        audit_payload = {}
    if mapping_contract_path:
        from oxq.spec.mapping_contract import (
            validate_mapping_contract_file,
            validate_mapping_contract_for_builder_pass_file,
        )
        from oxq.spec.schema import StrategySpec

        mapping_spec = StrategySpec.from_yaml(spec_path) if spec_path else None
        mapping_result = validate_mapping_contract_file(mapping_contract_path, spec=mapping_spec)
        if mapping_result["status"] == "fail":
            result["status"] = "fail"
            result["errors"].extend(
                {
                    "path": f"mapping_contract.{error['path']}",
                    "message": error["message"],
                }
                for error in mapping_result["errors"]
            )
        audit_all_pass = (
            isinstance(audit_payload, dict)
            and audit_payload.get("status") == "pass"
            and audit_payload.get("audit_conclusion") == "all_pass"
        )
        if audit_all_pass:
            idea_brief_path = audit_payload.get("strategy_idea_brief")
            builder_pass_result = validate_mapping_contract_for_builder_pass_file(
                mapping_contract_path,
                spec=mapping_spec,
                idea_brief_path=idea_brief_path if isinstance(idea_brief_path, str) else None,
            )
            if builder_pass_result["status"] == "fail":
                result["status"] = "fail"
                result["errors"].extend(
                    {
                        "path": f"mapping_contract.builder_pass.{error['path']}",
                        "message": error["message"],
                    }
                    for error in builder_pass_result["errors"]
                )
    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        for error in result["errors"]:
            click.echo(f"  {error['path']}: {error['message']}")
    if result["status"] == "fail":
        raise SystemExit(1)


@main.group(name="runtime-audit")
def runtime_audit():
    """Validate Agent-authored runtime audit artifacts."""


@runtime_audit.command(name="validate")
@click.argument("audit_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def runtime_audit_validate(audit_file: str, as_json: bool):
    """Validate runtime_audit.json schema without semantic language judgment."""
    from oxq.spec.runtime_audit_schema import validate_runtime_audit_file

    result = validate_runtime_audit_file(audit_file)
    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        for error in result["errors"]:
            click.echo(f"  {error['path']}: {error['message']}")
    if result["status"] == "fail":
        raise SystemExit(1)


@main.group()
def audit():
    """Audit backtest runs for reproducibility and research bias."""


@audit.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--publish",
    "publish_result",
    is_flag=True,
    help="Atomically publish and bind reproducibility_audit.json in RUN_DIR.",
)
def reproducibility(run_dir: str, as_json: bool, publish_result: bool):
    """Run reproducibility audit on a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    from oxq.audit import audit_reproducibility

    if publish_result:
        run_path = Path(run_dir).resolve()
        with run_digest_transaction(run_path):
            result = audit_reproducibility(run_path)
            _publish_audit_result(run_path, "reproducibility_audit.json", result)
    else:
        result = audit_reproducibility(run_dir)

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, indent=2))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        click.echo(f"Fatal: {result['fatal_count']}, Warnings: {result['warning_count']}")
        for c in result["checks"]:
            icon = "PASS" if c["status"] == "pass" else ("INFO" if c["status"] == "info" else "FAIL")
            click.echo(f"  [{c['severity']}] {icon} {c['id']}: {c['message']}")

    if result["status"] == "fail":
        raise SystemExit(1)


@audit.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--publish",
    "publish_result",
    is_flag=True,
    help="Atomically publish and bind research_bias_audit.json in RUN_DIR.",
)
def research(run_dir: str, as_json: bool, publish_result: bool):
    """Run research bias audit on a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    from oxq.audit import audit_research

    if publish_result:
        run_path = Path(run_dir).resolve()
        with run_digest_transaction(run_path):
            result = audit_research(run_path)
            _publish_audit_result(run_path, "research_bias_audit.json", result)
    else:
        result = audit_research(run_dir)

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, indent=2))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        click.echo(f"Fatal: {result['fatal_count']}, Warnings: {result['warning_count']}")
        for c in result["checks"]:
            icon = "PASS" if c["status"] == "pass" else ("INFO" if c["status"] == "info" else "FAIL")
            click.echo(f"  [{c['severity']}] {icon} {c['id']}: {c['message']}")

    if result["status"] == "fail":
        raise SystemExit(1)


def _publish_audit_result(run_path: Path, artifact_name: str, result: dict) -> None:
    content = (json.dumps(result, indent=2) + "\n").encode()
    try:
        publish_run_artifacts(run_path, {artifact_name: content})
    except RunDigestError as exc:
        raise click.ClickException(str(exc)) from exc


@main.group()
def report():
    """Generate research reports from backtest runs."""


@report.group(name="asset")
def report_asset():
    """Manage report assets for a backtest run."""


@report_asset.command(name="add")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--id", "asset_id", required=True, help="Stable asset id")
@click.option("--title", required=True, help="Human-readable asset title")
@click.option("--caption", default="", help="Optional asset caption")
@click.option("--section", default="results", show_default=True, help="Report section")
@click.option("--order", default=100, show_default=True, type=int, help="Sort order within section")
@click.option("--source-script", default=None, type=click.Path(exists=True, dir_okay=False), help="Plotting script path")
@click.option("--source-artifact", multiple=True, help="Input run artifact used to create this asset")
def report_asset_add(
    run_dir: str,
    file_path: str,
    asset_id: str,
    title: str,
    caption: str,
    section: str,
    order: int,
    source_script: str | None,
    source_artifact: tuple[str, ...],
):
    """Register a figure or attachment as a report asset."""
    from oxq.report.assets import add_report_asset

    try:
        asset = add_report_asset(
            run_dir,
            file_path,
            asset_id=asset_id,
            title=title,
            caption=caption,
            section=section,
            order=order,
            source_script=source_script,
            source_artifacts=list(source_artifact),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Added report asset {asset.id}")
    click.echo(f"  Kind: {asset.kind}")
    click.echo(f"  Path: {asset.path}")
    click.echo(f"  Hash: {asset.sha256}")


@report_asset.command(name="add-batch")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("items_json", type=click.Path(exists=True, dir_okay=False))
def report_asset_add_batch(run_dir: str, items_json: str):
    """Register multiple report assets from a JSON array."""
    from oxq.report.assets import add_report_assets

    try:
        raw = json.loads(Path(items_json).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("report asset batch JSON must be an array")
        assets = add_report_assets(run_dir, raw)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Added {len(assets)} report assets")
    for asset in assets:
        click.echo(f"  {asset.id}")
        click.echo(f"    Kind: {asset.kind}")
        click.echo(f"    Path: {asset.path}")
        click.echo(f"    Hash: {asset.sha256}")


@report_asset.command(name="list")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def report_asset_list(run_dir: str):
    """List registered report assets."""
    from oxq.report.assets import list_report_assets

    assets = list_report_assets(run_dir)
    if not assets:
        click.echo("No report assets registered.")
        return

    for asset in assets:
        click.echo(f"{asset.id}")
        click.echo(f"  Kind: {asset.kind}")
        click.echo(f"  Title: {asset.title}")
        click.echo(f"  Path: {asset.path}")
        click.echo(f"  Hash: {asset.sha256}")


@report.command(name="qa")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def report_qa(run_dir: str, as_json: bool):
    """Run deterministic QA checks on final Markdown and HTML reports."""
    from oxq.report.qa import run_report_qa

    try:
        result = run_report_qa(run_dir, include_advisory_checks=False)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        facts = result.facts
        click.echo(f"Status: {result.status.upper()}")
        click.echo(f"Fatal: {result.fatal_count}, Warnings: {result.warning_count}")
        click.echo(f"Configured end date: {facts.configured_end_date or 'N/A'}")
        click.echo(f"Effective last trading day: {facts.effective_last_trading_day or 'N/A'}")
        click.echo("Semantic report review: use review-research-report")
        for finding in result.findings:
            click.echo(f"  [{finding.severity}] {finding.id}: {finding.message}")

    if result.status == "fail":
        raise SystemExit(1)


@main.group()
def experiment():
    """Manage experiment registry."""


@experiment.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--registry", "-r", default=None, help="Experiment registry file")
def add(run_dir: str, registry: str | None):
    """Add a backtest run to the experiment registry.

    RUN_DIR is the path to a run directory.
    """
    from oxq.observe.experiment_registry import add_experiment

    if not (Path(run_dir) / "metrics.json").exists():
        click.echo("Error: metrics.json not found in run directory")
        raise SystemExit(1)

    workspace = _read_workspace_config(Path.cwd() / ".open-xquant" / "workspace.yaml")
    registry_path = (
        Path(registry)
        if registry is not None
        else (_configured_workspace_path(workspace, "experiment_registry") or Path("experiments.jsonl"))
    )
    version_root = None
    backtest_phase_dir = None
    version_id = None
    if _is_version_governed_workspace(workspace):
        version_id = _workspace_active_version(workspace)
        if not version_id:
            raise click.ClickException(
                "version-governed workspace requires current.json with a safe active_version"
            )
        version_dir = _resolve_active_version_dir(
            Path.cwd(),
            _workspace_versions_dir(workspace),
            version_id,
        )
        if not (version_dir / "version_manifest.json").is_file():
            raise click.ClickException(
                "version-governed workspace requires the active version_manifest.json"
            )
        backtest_phase_dir = Path.cwd() / _workspace_active_backtest_output(workspace, version_id)
    entry = add_experiment(
        run_dir,
        registry_path=registry_path,
        version_root=version_root,
        backtest_phase_dir=backtest_phase_dir,
        version_id=version_id,
    )
    if "error" in entry:
        click.echo(f"Error: {entry['error']}")
        raise SystemExit(1)

    click.echo(f"Experiment added to {registry_path}")
    click.echo(f"  Experiment ID: {entry['experiment_id']}")
    click.echo(f"  Strategy:      {entry['strategy_id']}")


@main.group()
def robustness():
    """Run robustness tests on backtest runs."""


@robustness.command(name="run")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def run_robustness_cmd(run_dir: str, as_json: bool):
    """Run robustness tests on a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    import oxq.robustness

    result = oxq.robustness.run_robustness(run_dir)

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, indent=2, default=str))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        click.echo(f"Baseline Sharpe: {_format_optional_float(result.get('baseline_sharpe'))}")
        click.echo("")
        for t in result["tests"]:
            icon = "PASS" if t["status"] == "pass" else ("FAIL" if t["status"] == "fail" else "WARN")
            click.echo(f"  [{t['status'].upper()}] {icon} {t['name']}: {t.get('message', '')}")
            if "baseline_sharpe" in t:
                click.echo(
                    "         Baseline: "
                    f"{_format_optional_float(t.get('baseline_sharpe'))} → "
                    f"Perturbed: {_format_optional_float(t.get('perturbed_sharpe'))}"
                )

    if result.get("status") in {"error", "fragile"}:
        raise SystemExit(1)


def _format_optional_float(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{parsed:.4f}" if math.isfinite(parsed) else "N/A"


main.add_command(agent_group)
main.add_command(doctor)
main.add_command(research_group)
