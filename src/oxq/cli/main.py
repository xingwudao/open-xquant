"""oxq CLI — Agentic Quant Research Kernel command-line interface."""

from __future__ import annotations

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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def validate(spec_file: str, as_json: bool):
    """Validate a strategy spec file.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    try:
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


@main.group()
def backtest():
    """Run backtests from strategy specs."""


def _backtest_artifact_paths(run_dir: Path) -> dict[str, str]:
    artifacts = {
        "strategy_spec_yaml": str(run_dir / "strategy_spec.yaml"),
        "environment_json": str(run_dir / "environment.json"),
        "data_manifest_json": str(run_dir / "data_manifest.json"),
        "execution_assumptions_json": str(run_dir / "execution_assumptions.json"),
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
@click.argument("spec_file", type=click.Path(exists=True))
@click.option("--out", "-o", default="runs/auto", help="Output directory for run artifacts")
@click.option("--data-dir", default=None, help="Directory for market data files")
@click.option("--json", "as_json", is_flag=True, help="Output machine-readable JSON")
def run(spec_file: str, out: str, data_dir: str | None, as_json: bool):
    """Run a backtest from a strategy spec file.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    from oxq.spec.compiler import compile_run

    try:
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


@main.group()
def strategy():
    """Manage compiled strategies."""


@strategy.command()
@click.argument("spec_file", type=click.Path(exists=True))
def compile(spec_file: str):
    """Compile a strategy spec into an executable strategy.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    from oxq.spec.compiler import compile_strategy

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


@report.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--out", "-o", default=None, help="Output file path (default: run_dir/research_report.md)")
def write(run_dir: str, out: str | None):
    """Generate a research report from a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    from oxq.report import generate_report

    report_md = generate_report(run_dir)
    output_path = Path(out) if out else Path(run_dir) / "research_report.md"
    output_path.write_text(report_md, encoding="utf-8")
    click.echo(f"Report written to {output_path}")


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
