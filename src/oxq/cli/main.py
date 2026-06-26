"""oxq CLI — Agentic Quant Research Kernel command-line interface."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import click
import yaml

from oxq.cli.agent import agent as agent_group
from oxq.cli.doctor import doctor
from oxq.cli.research import research as research_group
from oxq.spec.schema import StrategySpec, make_strategy_id
from oxq.spec.validator import validate as validate_spec


@click.group()
def main():
    """oxq — Agentic Quant Research Kernel CLI."""


@main.group()
def spec():
    """Manage strategy specs."""


@spec.command()
@click.argument("description")
@click.option("--out", "-o", default="strategy_spec.yaml", help="Output file path")
def init(description: str, out: str):
    """Initialize a new strategy spec from a natural language description.

    DESCRIPTION is a brief strategy idea in natural language.
    """
    strategy_id = make_strategy_id(description)
    template = StrategySpec.template(strategy_id=strategy_id, hypothesis=description)

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(template.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False), encoding="utf-8")

    click.echo(f"Spec template written to {output_path}")
    click.echo(f"Strategy ID: {strategy_id}")
    click.echo("Next: edit the file, then run `oxq spec validate`")


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


def _load_component_manifests(manifest_paths: tuple[str, ...]) -> list[dict]:
    """Load workspace component manifests and annotate them for catalog export."""
    if not manifest_paths:
        return []
    from oxq.core.component_manifest import load_component_manifest

    manifests: list[dict] = []
    for raw_path in manifest_paths:
        loaded = load_component_manifest(raw_path, verify_hash=True)
        loaded["_manifest_path"] = str(Path(raw_path).resolve())
        manifests.append(loaded)
    return manifests


def _write_run_component_manifest_artifacts(run_dir: Path, manifests: list[dict]) -> None:
    from oxq.spec.compiler import _append_run_digest, _hash_file, _hash_json_file

    summary = [
        {
            "manifest_path": manifest.get("_manifest_path", ""),
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
        for manifest in manifests
    ]
    (run_dir / "component_manifests.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if len(manifests) == 1:
        manifest_copy = dict(manifests[0])
        manifest_copy.pop("_manifest_path", None)
        (run_dir / "component_manifest.json").write_text(
            json.dumps(manifest_copy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (run_dir / "component_bundle_hash.txt").write_text(
            str(manifests[0].get("bundle_hash", "")) + "\n",
            encoding="utf-8",
        )

    artifact_hashes_path = run_dir / "artifact_hashes.json"
    artifact_hashes = json.loads(artifact_hashes_path.read_text(encoding="utf-8"))
    artifact_hashes["component_manifests.json"] = _hash_json_file(run_dir / "component_manifests.json")
    if (run_dir / "component_manifest.json").exists():
        artifact_hashes["component_manifest.json"] = _hash_json_file(run_dir / "component_manifest.json")
    if (run_dir / "component_bundle_hash.txt").exists():
        artifact_hashes["component_bundle_hash.txt"] = _hash_file(run_dir / "component_bundle_hash.txt")
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_run_digest(run_dir, _hash_json_file(artifact_hashes_path))


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


@backtest.command()
@click.argument("spec_file", type=click.Path())
@click.option("--out", "-o", default="runs/auto", help="Output directory for run artifacts")
@click.option("--data-dir", default=None, help="Directory for market data files")
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
    "--component-manifest",
    "component_manifest",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Workspace component manifest to load before validation, compile, and run.",
)
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def run(
    spec_file: str,
    out: str,
    data_dir: str | None,
    spec_audit: str | None,
    runtime_audit: str | None,
    component_manifest: tuple[str, ...],
    as_json: bool,
):
    """Run a backtest from a strategy spec file.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    from oxq.spec.compiler import compile_run

    spec_path = Path(spec_file)
    if not spec_path.exists():
        message = f"strategy spec file not found: {spec_file}"
        if as_json:
            click.echo(json.dumps(_backtest_json_failure("spec_file_missing", message), indent=2))
            raise SystemExit(1)
        raise click.ClickException(message)

    try:
        loaded_component_manifests = _load_component_manifests(component_manifest)
        spec = StrategySpec.from_yaml(spec_file)
    except Exception as e:
        if as_json:
            click.echo(json.dumps(_backtest_json_failure("parse_error", str(e)), indent=2))
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

    pre_run_audit_path = Path(spec_audit) if spec_audit is not None else _default_spec_audit_path(spec_path)
    if pre_run_audit_path is not None:
        try:
            _require_pre_backtest_spec_audit(spec, pre_run_audit_path)
        except click.ClickException as e:
            if as_json:
                click.echo(
                    json.dumps(
                        _backtest_json_failure("spec_audit_failed", e.message, warnings=validation.warnings),
                        indent=2,
                    )
                )
                raise SystemExit(1)
            raise
    pre_run_runtime_audit_path = (
        Path(runtime_audit) if runtime_audit is not None else _default_runtime_audit_path(spec_path)
    )
    if pre_run_audit_path is not None and pre_run_runtime_audit_path is None:
        message = "runtime_audit.json is required when a spec audit gates a formal backtest"
        if as_json:
            click.echo(
                json.dumps(_backtest_json_failure("runtime_audit_missing", message, warnings=validation.warnings), indent=2)
            )
            raise SystemExit(1)
        raise click.ClickException(message)
    if pre_run_runtime_audit_path is not None:
        try:
            _require_pre_backtest_runtime_audit(
                spec,
                pre_run_runtime_audit_path,
                spec_audit_path=pre_run_audit_path,
                effective_data_dir=data_dir,
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

    if not as_json:
        click.echo(f"Running backtest for '{spec.strategy_id}'...")
    try:
        result, run_dir = compile_run(spec, data_dir=data_dir, out_dir=out)
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
    required=False,
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
def attach_provenance(run_dir: str, spec_audit: str, runtime_audit: str | None, component_catalog: str, as_json: bool):
    """Attach pre-run provenance artifacts while preserving run digests."""
    from oxq.audit import audit_reproducibility
    from oxq.core.component_catalog import _catalog_hash, _stable_hash
    from oxq.spec.audit_schema import validate_spec_audit_file
    from oxq.spec.compiler import _append_run_digest, _hash_file, _hash_json_file
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

    audit_validation = validate_spec_audit_file(spec_audit)
    if audit_validation["status"] == "fail":
        raise click.ClickException(f"invalid spec audit: {audit_validation['errors']}")

    audit_payload = json.loads(Path(spec_audit).read_text(encoding="utf-8"))
    catalog_payload = json.loads(Path(component_catalog).read_text(encoding="utf-8"))
    audit_status = _require_json_str(audit_payload, "status")
    if audit_status != "pass":
        raise click.ClickException(f"spec audit status must be pass before attaching provenance: {audit_status}")
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
    runtime_payload: dict[str, object] | None = None
    if runtime_audit is not None:
        runtime_validation = validate_runtime_audit_file(runtime_audit)
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
            compiled_plan_path=run_path / "compiled_plan.json",
        )

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

    (run_path / "spec_audit.json").write_text(Path(spec_audit).read_text(encoding="utf-8"), encoding="utf-8")
    attached = [
        "spec_audit.json",
        "conversation_hash.txt",
        "component_catalog_hash.txt",
        "recipe_catalog_hash.txt",
    ]
    if runtime_audit is not None:
        (run_path / "runtime_audit.json").write_text(Path(runtime_audit).read_text(encoding="utf-8"), encoding="utf-8")
        attached.insert(1, "runtime_audit.json")
    (run_path / "conversation_hash.txt").write_text(conversation_hash + "\n", encoding="utf-8")
    (run_path / "component_catalog_hash.txt").write_text(catalog_hash + "\n", encoding="utf-8")
    (run_path / "recipe_catalog_hash.txt").write_text(recipe_catalog_hash + "\n", encoding="utf-8")

    artifact_hashes.update(
        {
            "spec_audit.json": _hash_json_file(run_path / "spec_audit.json"),
            "conversation_hash.txt": _hash_file(run_path / "conversation_hash.txt"),
            "component_catalog_hash.txt": _hash_file(run_path / "component_catalog_hash.txt"),
            "recipe_catalog_hash.txt": _hash_file(run_path / "recipe_catalog_hash.txt"),
        }
    )
    if runtime_audit is not None:
        artifact_hashes["runtime_audit.json"] = _hash_json_file(run_path / "runtime_audit.json")
    artifact_hashes_path.write_text(json.dumps(artifact_hashes, indent=2) + "\n", encoding="utf-8")
    artifact_hashes_digest = _hash_json_file(artifact_hashes_path)
    _append_run_digest(run_path, artifact_hashes_digest)

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

    audit_validation = validate_spec_audit_file(spec_audit_path)
    if audit_validation["status"] == "fail":
        raise click.ClickException(f"invalid spec audit: {audit_validation['errors']}")

    audit_payload = json.loads(spec_audit_path.read_text(encoding="utf-8"))
    audit_status = _require_json_str(audit_payload, "status")
    if audit_status != "pass":
        raise click.ClickException(f"spec audit status must be pass before backtest: {audit_status}")
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
) -> None:
    """Deterministically gate a formal backtest on a pre-run runtime audit."""
    from oxq.spec.compiler import compile_plan
    from oxq.spec.runtime_audit_schema import validate_runtime_audit_file

    audit_validation = validate_runtime_audit_file(runtime_audit_path)
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
        compiled_plan_payload=compile_plan(spec, effective_data_dir=effective_data_dir),
    )


def _require_runtime_audit_hashes(
    audit_payload: dict[str, object],
    *,
    spec_hash: str,
    spec_audit_path: Path | None = None,
    compiled_plan_path: Path | None = None,
    compiled_plan_payload: object | None = None,
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


def _hash_json_payload(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def _require_run_component_bundles_in_catalog(run_path: Path, catalog_payload: object) -> None:
    run_hashes = _run_component_bundle_hashes(run_path)
    catalog_hashes = _catalog_component_bundle_hashes(catalog_payload)
    if not run_hashes and not catalog_hashes:
        return
    if run_hashes != catalog_hashes:
        raise click.ClickException(
            "component bundle hash mismatch between run artifacts and component catalog: "
            f"run={sorted(run_hashes)}, catalog={sorted(catalog_hashes)}"
        )


def _run_component_bundle_hashes(run_path: Path) -> set[str]:
    hashes: set[str] = set()
    summary_path = run_path / "component_manifests.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, list):
            raise click.ClickException("component_manifests.json must be a list")
        for item in summary:
            if isinstance(item, dict) and isinstance(item.get("bundle_hash"), str) and item["bundle_hash"]:
                hashes.add(item["bundle_hash"])
    manifest_path = run_path / "component_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and isinstance(manifest.get("bundle_hash"), str) and manifest["bundle_hash"]:
            hashes.add(manifest["bundle_hash"])
    bundle_hash_path = run_path / "component_bundle_hash.txt"
    if bundle_hash_path.exists():
        digest = bundle_hash_path.read_text(encoding="utf-8").strip()
        if digest:
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


@main.group()
def strategy():
    """Manage compiled strategies."""


@strategy.command()
@click.argument("spec_file", type=click.Path(exists=True))
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
def compile(spec_file: str, component_manifest: tuple[str, ...], out: str | None):
    """Compile a strategy spec into an executable strategy.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    from oxq.spec.compiler import compile_plan, compile_strategy

    _load_component_manifests(component_manifest)
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
        out_dir.mkdir(parents=True, exist_ok=True)
        plan = compile_plan(spec)
        (out_dir / "compiled_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (out_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")
        click.echo(f"  Compile preview: {out_dir / 'compiled_plan.json'}")


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
    from oxq.core.component_manifest import component_manifest_summary, load_component_manifest

    try:
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
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON.")
def spec_audit_validate(audit_file: str, as_json: bool):
    """Validate spec_audit.json schema without semantic language judgment."""
    from oxq.spec.audit_schema import validate_spec_audit_file

    result = validate_spec_audit_file(audit_file)
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
def reproducibility(run_dir: str, as_json: bool):
    """Run reproducibility audit on a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    from oxq.audit import audit_reproducibility

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
def research(run_dir: str, as_json: bool):
    """Run research bias audit on a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    from oxq.audit import audit_research

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
@click.option("--registry", "-r", default="experiments.jsonl", help="Experiment registry file")
def add(run_dir: str, registry: str):
    """Add a backtest run to the experiment registry.

    RUN_DIR is the path to a run directory.
    """
    from oxq.observe.experiment_registry import add_experiment

    if not (Path(run_dir) / "metrics.json").exists():
        click.echo("Error: metrics.json not found in run directory")
        raise SystemExit(1)

    entry = add_experiment(run_dir, registry_path=registry)
    if "error" in entry:
        click.echo(f"Error: {entry['error']}")
        raise SystemExit(1)

    click.echo(f"Experiment added to {registry}")
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
